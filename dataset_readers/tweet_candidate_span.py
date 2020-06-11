# -*- coding: utf-8 -*-
from typing import Iterable, Optional

import pandas as pd

from allennlp.data import Instance, DatasetReader
from allennlp.data.token_indexers import PretrainedTransformerIndexer
from allennlp.data.tokenizers import PretrainedTransformerTokenizer
from allennlp.data.fields import TextField, LabelField, MetadataField
from allennlp.common.file_utils import cached_path
from allennlp.common.util import sanitize_wordpiece

from whisper.common.utils import extract_config_from_archive
from whisper.data.fields import SpanPairsField


@DatasetReader.register("tweet_candidate_span")
class TweetCandidateSpanDatasetReader(DatasetReader):
    def __init__(
        self,
        lazy: bool = False,
        cache_directory: Optional[str] = None,
        max_instances: Optional[int] = None,
        min_num_candidate: int = 3,
        max_num_candidate: int = 5,
        transformer_model_name_or_archive_path: str = "bert-base-uncased",
    ) -> None:
        super().__init__(
            lazy=lazy, cache_directory=cache_directory, max_instances=max_instances
        )
        if "tar.gz" in transformer_model_name_or_archive_path:
            config = extract_config_from_archive(transformer_model_name_or_archive_path)
            model_name = config.as_dict()["dataset_reader"]["tokenizer"]["model_name"]
        else:
            model_name = transformer_model_name_or_archive_path
        self._tokenizer = PretrainedTransformerTokenizer(
            model_name=model_name, add_special_tokens=False
        )
        self._tokenindexer = PretrainedTransformerIndexer(
            model_name=model_name
        )
        self._min_num_candidate = min_num_candidate
        self._max_num_candidate = max_num_candidate

    def _read(self, file_path: str) -> Iterable[Instance]:
        file_path = cached_path(file_path)
        df = pd.read_json(file_path, lines=True)
        for record in df.to_dict("records"):
            if record["selected_text"]:
                text = record["text"]
                if not isinstance(text, str):
                    continue
                elif text.strip() == "":
                    continue
                elif len(record["candidate_spans"]) < self._min_num_candidate:
                    continue
                else:
                    yield self.text_to_instance(
                        " " + text.strip(),
                        record["sentiment"],
                        record["candidate_spans"],
                        record["textID"],
                        record.get("selected_text"),
                        record.get("selected_text_span"),
                    )

    def text_to_instance(
        self,
        text: str,
        sentiment: str,
        candidate_spans: list,
        text_id: Optional[str] = None,
        selected_text: Optional[str] = None,
        selected_text_span: Optional[tuple] = None,
    ) -> Instance:
        fields = {}
        text_tokens = self._tokenizer.tokenize(text)
        sentiment_tokens = self._tokenizer.tokenize(sentiment)
        text_with_sentiment_tokens = self._tokenizer.add_special_tokens(
            text_tokens, sentiment_tokens
        )
        fields["text_with_sentiment"] = TextField(
            text_with_sentiment_tokens, {"tokens": self._tokenindexer}
        )
        candidate_spans = [tuple(i) for i in candidate_spans[:self._max_num_candidate]]
        additional_metadata = {}
        if selected_text_span is not None:
            selected_text_span = tuple(selected_text_span)
            additional_metadata["selected_text_span"] = selected_text_span
            if selected_text_span not in candidate_spans:
                candidate_spans.append(selected_text_span)
                fields["label"] = LabelField(len(candidate_spans)-1, skip_indexing=True)
                have_truth = False
            else:
                fields["label"] = LabelField(candidate_spans.index(selected_text_span), skip_indexing=True)
                have_truth = True
            additional_metadata["have_truth"] = have_truth
            additional_metadata["candidate_num"] = len(candidate_spans)
        fields["candidate_span_pairs"] = SpanPairsField(candidate_spans, fields["text_with_sentiment"])
        metadata = {
            "text": text,
            "sentiment": sentiment,
            "selected_text": selected_text,
            "text_with_sentiment_tokens": text_with_sentiment_tokens}
        if text_id is not None:
            metadata["text_id"] = text_id
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
