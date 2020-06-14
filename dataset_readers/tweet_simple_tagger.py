# -*- coding: utf-8 -*-
# -*- coding: utf-8 -*-
from typing import Iterable, Optional

import pandas as pd
from allennlp.common.util import sanitize_wordpiece

from allennlp.data import Tokenizer, TokenIndexer, Instance, DatasetReader, Token
from allennlp.data.token_indexers import PretrainedTransformerIndexer
from allennlp.data.tokenizers import PretrainedTransformerTokenizer
from allennlp.data.fields import TextField, LabelField, SequenceLabelField, MetadataField
from allennlp.common.file_utils import cached_path

from whisper.common.rc_utils import char_span_to_token_span


@DatasetReader.register("tweet_simple_tagger")
class TweetTaggerDatasetReader(DatasetReader):
    def __init__(
        self,
        lazy: bool = False,
        cache_directory: Optional[str] = None,
        max_instances: Optional[int] = None,
        tokenizer: Optional[Tokenizer] = None,
        tokenindexer: Optional[TokenIndexer] = None,
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

    def _read(self, file_path: str) -> Iterable[Instance]:
        file_path = cached_path(file_path)
        df = pd.read_csv(file_path)
        for record in df.to_dict("records"):
            text = record["text"]
            if not isinstance(text, str):
                continue
            elif text.strip() == "":
                continue
            else:
                yield self.text_to_instance(
                    " " + text.strip(),
                    record["sentiment"],
                    record.get("selected_text"),
                )

    def text_to_instance(
        self,
        text: str,
        sentiment: str,
        selected_text: Optional[str] = None,
    ) -> Instance:
        fields = {}
        text_tokens = self._tokenizer.tokenize(text)
        sentiment_tokens = self._tokenizer.tokenize(sentiment)
        # add special tokens

        text_with_sentiment_tokens = self._tokenizer.add_special_tokens(
            text_tokens, sentiment_tokens
        )
        tokens_field = TextField(
            text_with_sentiment_tokens, {"tokens": self._tokenindexer}
        )
        fields["tokens"] = tokens_field

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
            tags = ["O"] * len(tokens_field)
            for i in range(token_answer_span_start, token_answer_span_end+1):
                tags[i] = "I"
            fields["tags"] = SequenceLabelField(
                tags, tokens_field
            )


        # make the metadata
        metadata = {
            "text": text,
            "sentiment": sentiment,
            "words": text,
            "text_with_sentiment_tokens": text_with_sentiment_tokens,
        }
        if additional_metadata:
            metadata.update(additional_metadata)
        fields["metadata"] = MetadataField(metadata)

        return Instance(fields)

    def span_to_str(self, text, span_start, span_end):
        text_tokens = self._tokenizer.tokenize(text)
        text_tokens = self._tokenizer.add_special_tokens(text_tokens)
        return span_tokens_to_text(text, text_tokens, span_start, span_end)


def span_tokens_to_text(source_text, tokens, span_start, span_end):
    text_with_sentiment_tokens = tokens
    predicted_start = span_start
    predicted_end = span_end

    while (
        predicted_start >= 0
        and text_with_sentiment_tokens[predicted_start].idx is None
    ):
        predicted_start -= 1
    if predicted_start < 0:
        character_start = 0
    else:
        character_start = text_with_sentiment_tokens[predicted_start].idx

    while (
        predicted_end < len(text_with_sentiment_tokens)
        and text_with_sentiment_tokens[predicted_end].idx is None
    ):
        predicted_end -= 1

    if predicted_end >= len(text_with_sentiment_tokens):
        print(text_with_sentiment_tokens)
        print(len(text_with_sentiment_tokens))
        print(span_end)
        print(predicted_end)
        character_end = len(source_text)
    else:
        end_token = text_with_sentiment_tokens[predicted_end]
        if end_token.idx == 0:
            character_end = (
                end_token.idx + len(sanitize_wordpiece(end_token.text)) + 1
            )
        else:
            character_end = end_token.idx + len(sanitize_wordpiece(end_token.text))

    best_span_string = source_text[character_start:character_end].strip()
    return best_span_string
