# -*- coding: utf-8 -*-
from typing import Iterable, Optional

import pandas as pd

from allennlp.data import Instance, DatasetReader
from allennlp.data.token_indexers import PretrainedTransformerIndexer
from allennlp.data.tokenizers import PretrainedTransformerTokenizer
from allennlp.data.fields import TextField, LabelField, MetadataField
from allennlp.common.file_utils import cached_path

from whisper.common.utils import extract_config_from_archive
from whisper.data.fields import SpanPairsField


@DatasetReader.register("tweet_candidate_span")
class TweetCandidateSpanDatasetReader(DatasetReader):
    def __init__(
        self,
        lazy: bool = False,
        cache_directory: Optional[str] = None,
        max_instances: Optional[int] = None,
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
                else:
                    yield self.text_to_instance(
                        " " + text.strip(),
                        record["sentiment"],
                        record["candidate_spans"],
                        record["textID"],
                        record["selected_text_span"],
                    )

    def text_to_instance(
        self,
        text: str,
        sentiment: str,
        candidate_spans: list,
        text_id: Optional[str] = None,
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
        candidate_spans = [tuple(i) for i in candidate_spans]
        additional_metadata = {}
        if selected_text_span is not None:
            selected_text_span = tuple(selected_text_span)
            additional_metadata["selected_text_span"] = selected_text_span
            if selected_text_span not in candidate_spans:
                candidate_spans.append(selected_text_span)
                fields["label"] = LabelField(len(candidate_spans)-1, skip_indexing=True)
            else:
                fields["label"] = LabelField(candidate_spans.index(selected_text_span), skip_indexing=True)
        fields["candidate_span_pairs"] = SpanPairsField(candidate_spans, fields["text_with_sentiment"])
        metadata = {"text": text, "sentiment": sentiment}
        if text_id is not None:
            metadata = {"text_id": text_id}
        if additional_metadata:
            metadata.update(additional_metadata)
        fields["metadata"] = MetadataField(metadata)
        return Instance(fields)
