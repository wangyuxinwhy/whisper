# -*- coding: utf-8 -*-
from typing import Dict, List, Optional, Tuple
import logging
import torch
from allennlp.common.util import pad_sequence_to_length

from overrides import overrides

from allennlp.data.vocabulary import Vocabulary
from allennlp.data.tokenizers.token import Token
from allennlp.data.token_indexers.token_indexer import TokenIndexer, IndexedTokenList

logger = logging.getLogger(__name__)


@TokenIndexer.register("tweet_bert")
class TweetBertIndexer(TokenIndexer):
    """
    This `TokenIndexer` assumes that Tokens already have their indexes in them (see `text_id` field).
    We still require `model_name` because we want to form allennlp vocabulary from pretrained one.
    This `Indexer` is only really appropriate to use if you've also used a
    corresponding :class:`PretrainedTransformerTokenizer` to tokenize your input.  Otherwise you'll
    have a mismatch between your tokens and your vocabulary, and you'll get a lot of UNK tokens.

    Registered as a `TokenIndexer` with name "pretrained_transformer".

    # Parameters

    model_name : `str`
        The name of the `transformers` model to use.
    namespace : `str`, optional (default=`tags`)
        We will add the tokens in the pytorch_transformer vocabulary to this vocabulary namespace.
        We use a somewhat confusing default value of `tags` so that we do not add padding or UNK
        tokens to this namespace, which would break on loading because we wouldn't find our default
        OOV token.
    max_length : `int`, optional (default = None)
        If not None, split the document into segments of this many tokens (including special tokens)
        before feeding into the embedder. The embedder embeds these segments independently and
        concatenate the results to get the original document representation. Should be set to
        the same value as the `max_length` option on the `PretrainedTransformerEmbedder`.
    """

    def __init__(
        self, namespace: str = "tags", **kwargs
    ) -> None:
        super().__init__(**kwargs)
        self._namespace = namespace

        self._num_added_start_tokens = 1
        self._num_added_end_tokens = 1

    @overrides
    def count_vocab_items(self, token: Token, counter: Dict[str, Dict[str, int]]):
        # If we only use pretrained models, we don't need to do anything here.
        pass

    @overrides
    def tokens_to_indices(self, tokens: List[Token], vocabulary: Vocabulary) -> IndexedTokenList:

        indices, type_ids = self._extract_token_and_type_ids(tokens)
        # The mask has 1 for real tokens and 0 for padding tokens. Only real tokens are attended to.
        output: IndexedTokenList = {
            "token_ids": indices,
            "mask": [True] * len(indices),
            "type_ids": type_ids,
        }

        return output

    def _extract_token_and_type_ids(
        self, tokens: List[Token]
    ) -> Tuple[List[int], Optional[List[int]]]:
        """
        Roughly equivalent to `zip(*[(token.text_id, token.type_id) for token in tokens])`,
        with some checks.
        """
        indices: List[int] = []
        type_ids: List[int] = []
        for token in tokens:
            if getattr(token, "text_id", None) is not None:
                # `text_id` being set on the token means that we aren't using the vocab, we just use
                # this id instead. Id comes from the pretrained vocab.
                # It is computed in PretrainedTransformerTokenizer.
                indices.append(token.text_id)
            else:
                raise KeyError(
                    "Using PretrainedTransformerIndexer but field text_id is not set"
                    f" for the following token: {token.text}"
                )

            if type_ids is not None and getattr(token, "type_id", None) is not None:
                type_ids.append(token.type_id)
            else:
                type_ids.append(0)

        return indices, type_ids


    @overrides
    def get_empty_token_list(self) -> IndexedTokenList:
        output: IndexedTokenList = {"token_ids": [], "mask": [], "type_ids": []}
        if self._max_length is not None:
            output["segment_concat_mask"] = []
        return output

    @overrides
    def as_padded_tensor_dict(
        self, tokens: IndexedTokenList, padding_lengths: Dict[str, int]
    ) -> Dict[str, torch.Tensor]:
        tensor_dict = {}
        for key, val in tokens.items():
            if key == "type_ids":
                padding_value = 0
                mktensor = torch.LongTensor
            elif key == "mask" or key == "wordpiece_mask":
                padding_value = False
                mktensor = torch.BoolTensor
            elif len(val) > 0 and isinstance(val[0], bool):
                padding_value = False
                mktensor = torch.BoolTensor
            else:
                padding_value = 1
                mktensor = torch.LongTensor

            tensor = mktensor(
                pad_sequence_to_length(
                    val, padding_lengths[key], default_value=lambda: padding_value
                )
            )

            tensor_dict[key] = tensor
        return tensor_dict
