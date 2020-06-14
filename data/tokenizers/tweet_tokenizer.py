# -*- coding: utf-8 -*-
import logging
from typing import List, Optional, Tuple

from overrides import overrides
from fairseq.data.encoders.fastbpe import fastBPE
from fairseq.data import Dictionary

from allennlp.data.tokenizers.token import Token
from allennlp.data.tokenizers.tokenizer import Tokenizer

from whisper.common.utils import sanitize_wordpiece

logger = logging.getLogger(__name__)


class Args:
    def __init__(self, bpe_codes_path):
        self.bpe_codes = bpe_codes_path


@Tokenizer.register("tweet_bert")
class TweetBertTokenizer(Tokenizer):

    def __init__(
        self,
        model_path: str,
    ) -> None:

        self.bpe = fastBPE(Args(model_path + "/bpe.codes"))
        self.vocab = Dictionary()
        self.vocab.add_from_file(f"{model_path}/dict.txt")
        self._tokenizer_lowercases = False
        self.sequence_pair_start_tokens = [Token(text="<s>", text_id=0, type_id=0)]
        self.sequence_pair_mid_tokens = [Token(text="</s>", text_id=2, type_id=0), Token(text="</s>", text_id=2, type_id=0)]
        self.sequence_pair_end_tokens = [Token(text="</s>", text_id=2, type_id=0)]

    @overrides
    def tokenize(self, text: str) -> List[Token]:
        """
        This method only handles a single sentence (or sequence) of text.
        """
        subwords = self.bpe.encode(text)
        token_ids = self.vocab.encode_line(subwords, append_eos=True)
        token_ids.clamp_max_(64000)
        token_offsets = self._estimate_character_indices(text, token_ids)
        tokens = []
        for subword, token_id, offsets in zip(subwords.split(" "), token_ids, token_offsets):
            if offsets is None or offsets[0] >= offsets[1]:
                start = None
                end = None
            else:
                start, end = offsets
            tokens.append(
                Token(text=subword, text_id=token_id, type_id=0, idx=start, idx_end=end)
            )

        return tokens

    def _estimate_character_indices(
        self, text: str, token_ids: List[int]
    ) -> List[Optional[Tuple[int, int]]]:
        """
        The huggingface tokenizers produce tokens that may or may not be slices from the
        original text.  Differences arise from lowercasing, Unicode normalization, and other
        kinds of normalization, as well as special characters that are included to denote
        various situations, such as "##" in BERT for word pieces from the middle of a word, or
        "Ġ" in RoBERTa for the beginning of words not at the start of a sentence.

        This code attempts to calculate character offsets while being tolerant to these
        differences. It scans through the text and the tokens in parallel, trying to match up
        positions in both. If it gets out of sync, it backs off to not adding any token
        indices, and attempts to catch back up afterwards. This procedure is approximate.
        Don't rely on precise results, especially in non-English languages that are far more
        affected by Unicode normalization.
        """

        token_texts = [sanitize_wordpiece(self.vocab[int(token_id)]) for token_id in token_ids]
        token_offsets: List[Optional[Tuple[int, int]]] = [None] * len(token_ids)
        if self._tokenizer_lowercases:
            text = text.lower()
            token_texts = [t.lower() for t in token_texts]

        min_allowed_skipped_whitespace = 3
        allowed_skipped_whitespace = min_allowed_skipped_whitespace

        text_index = 0
        token_index = 0
        while text_index < len(text) and token_index < len(token_ids):
            token_text = token_texts[token_index]
            token_start_index = text.find(token_text, text_index)

            # Did we not find it at all?
            if token_start_index < 0:
                token_index += 1
                # When we skip a token, we increase our tolerance, so we have a chance of catching back up.
                allowed_skipped_whitespace += 1 + min_allowed_skipped_whitespace
                continue

            # Did we jump too far?
            non_whitespace_chars_skipped = sum(
                1 for c in text[text_index:token_start_index] if not c.isspace()
            )
            if non_whitespace_chars_skipped > allowed_skipped_whitespace:
                # Too many skipped characters. Something is wrong. Ignore this token.
                token_index += 1
                # When we skip a token, we increase our tolerance, so we have a chance of catching back up.
                allowed_skipped_whitespace += 1 + min_allowed_skipped_whitespace
                continue
            allowed_skipped_whitespace = min_allowed_skipped_whitespace

            token_offsets[token_index] = (
                token_start_index,
                token_start_index + len(token_text),
            )
            text_index = token_start_index + len(token_text)
            token_index += 1
        return token_offsets

    def add_special_tokens(
        self, tokens1: List[Token], tokens2: Optional[List[Token]] = None
    ) -> List[Token]:
        # Make sure we don't change the input parameters
        import copy

        tokens1 = copy.deepcopy(tokens1)
        tokens2 = copy.deepcopy(tokens2)

        # We add special tokens and also set token type ids.
        if tokens2 is None:
            import copy

            tokens1 = copy.deepcopy(tokens1)
            return [Token(text="<s>", text_id=0, type_id=0)] + tokens1 + [Token(text="</s>", text_id=2, type_id=0)]
        else:
            return (
                [Token(text="<s>", text_id=0, type_id=0)]
                + tokens1
                + [Token(text="</s>", text_id=2, type_id=0), Token(text="</s>", text_id=2, type_id=0)]
                + tokens2
                + [Token(text="</s>", text_id=2, type_id=0)]
            )

    def num_special_tokens_for_sequence(self) -> int:
        return 2

    def num_special_tokens_for_pair(self) -> int:
        return 4
