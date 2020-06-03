# -*- coding: utf-8 -*-
from typing import Iterable, Optional

import pandas as pd
from allennlp.common.util import sanitize_wordpiece

from allennlp.data import Tokenizer, TokenIndexer, Instance, DatasetReader, Token
from allennlp.data.token_indexers import PretrainedTransformerIndexer
from allennlp.data.tokenizers import PretrainedTransformerTokenizer
from allennlp.data.fields import TextField, LabelField, SpanField, MetadataField
from allennlp.common.file_utils import cached_path

from whisper.common.rc_utils import char_span_to_token_span


@DatasetReader.register("tweet_sentiment")
class TweetSentimentDatasetReader(DatasetReader):
    def __init__(
        self,
        lazy: bool = False,
        cache_directory: Optional[str] = None,
        max_instances: Optional[int] = None,
        tokenizer: Optional[Tokenizer] = None,
        tokenindexer: Optional[TokenIndexer] = None,
        sentiment_first: bool = False,
    ) -> None:
        super().__init__(
            lazy=lazy, cache_directory=cache_directory, max_instances=max_instances
        )
        self._tokenizer = tokenizer or PretrainedTransformerTokenizer(
            model_name="bert-base-uncased"
        )
        self._tokenindexer = tokenindexer or PretrainedTransformerIndexer(
            model_name="bert-base-uncased"
        )
        self._sentiment_first = sentiment_first

    def _read(self, file_path: str) -> Iterable[Instance]:
        file_path = cached_path(file_path)
        df = pd.read_csv(file_path)
        for record in df.to_dict("records"):
            if record["selected_text"]:
                text = record["text"]
                if not isinstance(text, str):
                    continue
                elif text.strip() == "":
                    continue
                else:
                    yield self.text_to_instance(
                        " " + text.strip(),
                        record["sentiment"],
                        record["textID"],
                        record["selected_text"],
                    )

    def text_to_instance(
        self,
        text: str,
        sentiment: str,
        text_id: Optional[str] = None,
        selected_text: Optional[str] = None,
    ) -> Instance:
        fields = {}
        text_tokens = self._tokenizer.tokenize(text)
        sentiment_tokens = self._tokenizer.tokenize(sentiment)
        # add special tokens

        if self._sentiment_first:
            text_with_sentiment_tokens = self._tokenizer.add_special_tokens(
                sentiment_tokens, text_tokens
            )
            fields["text_with_sentiment"] = TextField(
                text_with_sentiment_tokens, {"tokens": self._tokenindexer}
            )
            text_start = (
                len(self._tokenizer.sequence_pair_start_tokens)
                + len(sentiment_tokens)
                + len(self._tokenizer.sequence_pair_mid_tokens)
            )
            fields["text_span"] = SpanField(
                text_start,
                len(text_with_sentiment_tokens) - 1,
                fields["text_with_sentiment"],
            )
        else:
            text_with_sentiment_tokens = self._tokenizer.add_special_tokens(
                text_tokens, sentiment_tokens
            )
            fields["text_with_sentiment"] = TextField(
                text_with_sentiment_tokens, {"tokens": self._tokenindexer}
            )
            text_start = len(self._tokenizer.sequence_pair_start_tokens)
            fields["text_span"] = SpanField(
                text_start,
                text_start + len(text_tokens) - 1,
                fields["text_with_sentiment"],
            )
        text_tokens = self._tokenizer.add_special_tokens(text_tokens)

        fields["text"] = TextField(text_tokens, {"tokens": self._tokenindexer})
        fields["sentiment"] = LabelField(sentiment)

        additional_metadata = {}
        if selected_text is not None:
            context = text
            answer = selected_text
            additional_metadata["selected_text"] = selected_text
            first_answer_offset = context.find(answer)

            def tokenize_slice(start: int, end: int) -> Iterable[Token]:
                text_to_tokenize = context[start:end]
                if start - 1 >= 0 and context[start - 1].isspace():
                    prefix = (
                        "a "
                    )  # must end in a space, and be short so we can be sure it becomes only one token
                    wordpieces = self._tokenizer.tokenize(prefix + text_to_tokenize)
                    for wordpiece in wordpieces:
                        if wordpiece.idx is not None:
                            wordpiece.idx -= len(prefix)
                    return wordpieces[1:]
                else:
                    return self._tokenizer.tokenize(text_to_tokenize)

            tokenized_context = []
            token_start = 0
            for i, c in enumerate(context):
                if c.isspace():
                    for wordpiece in tokenize_slice(token_start, i):
                        if wordpiece.idx is not None:
                            wordpiece.idx += token_start
                        tokenized_context.append(wordpiece)
                    token_start = i + 1
            for wordpiece in tokenize_slice(token_start, len(context)):
                if wordpiece.idx is not None:
                    wordpiece.idx += token_start
                tokenized_context.append(wordpiece)

            if first_answer_offset is None:
                (token_answer_span_start, token_answer_span_end) = (-1, -1)
            else:
                (
                    token_answer_span_start,
                    token_answer_span_end,
                ), _ = char_span_to_token_span(
                    [
                        (t.idx, t.idx + len(sanitize_wordpiece(t.text)))
                        if t.idx is not None
                        else None
                        for t in tokenized_context
                    ],
                    (first_answer_offset, first_answer_offset + len(answer)),
                )
            fields["selected_text_span"] = SpanField(
                token_answer_span_start,
                token_answer_span_end,
                fields["text_with_sentiment"],
            )

        if text_id is not None:
            additional_metadata["text_id"] = text_id

        # make the metadata
        metadata = {
            "text": text,
            "sentiment": sentiment,
            "text_with_sentiment_tokens": text_with_sentiment_tokens,
        }
        if additional_metadata:
            metadata.update(additional_metadata)
        fields["metadata"] = MetadataField(metadata)

        return Instance(fields)
