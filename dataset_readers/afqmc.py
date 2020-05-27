# -*- coding: utf-8 -*-
import json
from typing import Iterable, Optional

from allennlp.data import Tokenizer, TokenIndexer, Instance, DatasetReader
from allennlp.data.token_indexers import SingleIdTokenIndexer
from allennlp.data.fields import TextField, LabelField
from allennlp.common.file_utils import cached_path

from whisper.data.tokenizers import JiebaTokenizer, PretrainedTransformerZhTokenizer


@DatasetReader.register("afqmc")
class AfqmcDatasetReader(DatasetReader):
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
        self._tokenizer = tokenizer or JiebaTokenizer()
        self._tokenindexer = tokenindexer or SingleIdTokenIndexer()

    def _read(self, file_path: str) -> Iterable[Instance]:
        filepath = cached_path(file_path)
        with open(filepath) as f:
            records = []
            for line in f:
                records.append(json.loads(line.strip()))

        for record in records:
            yield self.text_to_instance(
                record["sentence1"], record["sentence2"], record["label"]
            )

    def text_to_instance(
        self, sentence1: str, sentence2: str, label: Optional[str] = None
    ) -> Instance:
        sentence1_tokens = self._tokenizer.tokenize(sentence1)
        sentence2_tokens = self._tokenizer.tokenize(sentence2)
        fields = {}
        sentence1_field = TextField(
            sentence1_tokens, token_indexers={"tokens": self._tokenindexer}
        )
        sentence2_field = TextField(
            sentence2_tokens, token_indexers={"tokens": self._tokenindexer}
        )
        fields["sentence1"] = sentence1_field
        fields["sentence2"] = sentence2_field
        if label is not None:
            label_field = LabelField(label)
            fields["label"] = label_field
        return Instance(fields=fields)


@DatasetReader.register("afqmc_for_transformer")
class AfqmcForTransformerDatasetReader(DatasetReader):
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
        self._tokenizer = tokenizer or PretrainedTransformerZhTokenizer()
        self._tokenindexer = tokenindexer or SingleIdTokenIndexer()

    def _read(self, file_path: str) -> Iterable[Instance]:
        filepath = cached_path(file_path)
        with open(filepath) as f:
            records = []
            for line in f:
                records.append(json.loads(line.strip()))

        for record in records:
            yield self.text_to_instance(
                record["sentence1"], record["sentence2"], record["label"]
            )

    def text_to_instance(
        self, sentence1: str, sentence2: str, label: Optional[str] = None
    ) -> Instance:
        sentence1_tokens = self._tokenizer.tokenize(sentence1)
        sentence2_tokens = self._tokenizer.tokenize(sentence2)
        fields = {}
        sentence1_field = TextField(
            sentence1_tokens, token_indexers={"tokens": self._tokenindexer}
        )
        sentence2_field = TextField(
            sentence2_tokens, token_indexers={"tokens": self._tokenindexer}
        )
        fields["sentence1"] = sentence1_field
        fields["sentence2"] = sentence2_field
        if label is not None:
            label_field = LabelField(label)
            fields["label"] = label_field
        return Instance(fields=fields)
