# -*- coding: utf-8 -*-
# -*- coding: utf-8 -*-
from typing import Iterable, Optional

import pandas as pd

from allennlp.data import Tokenizer, TokenIndexer, Instance, DatasetReader
from allennlp.data.token_indexers import PretrainedTransformerIndexer
from allennlp.data.tokenizers import PretrainedTransformerTokenizer
from allennlp.data.fields import TextField, LabelField
from allennlp.common.file_utils import cached_path


@DatasetReader.register("just_sentiment")
class JustSentimentDatasetReader(DatasetReader):
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
        text_with_sentiment_tokens = self._tokenizer.add_special_tokens(
            text_tokens, sentiment_tokens
        )
        fields["tokens"] = TextField(
            text_with_sentiment_tokens, {"tokens": self._tokenindexer}
        )
        fields["label"] = LabelField(sentiment)
        return Instance(fields)
