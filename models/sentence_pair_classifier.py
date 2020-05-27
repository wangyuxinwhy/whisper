# -*- coding: utf-8 -*-
from typing import Optional, Tuple, Dict

import torch

from allennlp.models.model import Model
from allennlp.data import Vocabulary, TextFieldTensors
from allennlp.modules import TextFieldEmbedder, Seq2VecEncoder, Seq2SeqEncoder
from allennlp.nn import InitializerApplicator
from allennlp.nn.util import get_text_field_mask, get_token_ids_from_text_field_tensors
from allennlp.training.metrics import CategoricalAccuracy

from whisper.modules import LinearPairVecToVec


@Model.register("sentence_pair_classifier")
class SentencePairClassifier(Model):
    def __init__(
        self,
        vocab: Vocabulary,
        text_field_embedder: TextFieldEmbedder,
        seq2vec_encoder: Seq2VecEncoder,
        seq2seq_encoder: Optional[Seq2SeqEncoder] = None,
        dropout: float = None,
        num_labels: int = None,
        label_namespace: str = "labels",
        namespace: Tuple[str] = ("sentence1", "sentence2"),
        initializer: InitializerApplicator = InitializerApplicator(),
        **kwargs,
    ) -> None:
        super().__init__(vocab=vocab, **kwargs)
        self._text_field_embedder = text_field_embedder
        self._seq2vec_encoder = seq2vec_encoder
        if seq2seq_encoder is not None:
            self._seq2seq_encoder = seq2seq_encoder
        else:
            self._seq2seq_encoder = None

        if dropout:
            self._dropout = torch.nn.Dropout(dropout)
        else:
            self._dropout = None

        self._label_namespace = label_namespace
        self._namespace = namespace
        if num_labels is not None:
            self._num_labels = num_labels
        else:
            self._num_labels = vocab.get_vocab_size(namespace=self._label_namespace)
        self._pair_vec_to_vec = LinearPairVecToVec(
            self._seq2vec_encoder.get_output_dim(),
            self._seq2vec_encoder.get_output_dim(),
            self._seq2vec_encoder.get_output_dim(),
        )
        self._classification_layer = torch.nn.Linear(
            self._seq2vec_encoder.get_output_dim(), self._num_labels
        )
        self._accuracy = CategoricalAccuracy()
        self._loss = torch.nn.CrossEntropyLoss()
        initializer(self)

    def forward(
        self,
        sentence1: TextFieldTensors,
        sentence2: TextFieldTensors,
        label: torch.IntTensor,
    ) -> Dict[str, torch.Tensor]:
        embedded_sentence1 = self._text_field_embedder(sentence1)
        embedded_sentence2 = self._text_field_embedder(sentence2)
        sentence1_mask = get_text_field_mask(sentence1)
        sentence2_mask = get_text_field_mask(sentence2)

        if self._seq2seq_encoder:
            embedded_sentence1 = self._seq2seq_encoder(
                embedded_sentence1, mask=sentence1_mask
            )
            embedded_sentence2 = self._seq2seq_encoder(
                embedded_sentence2, mask=sentence2_mask
            )

        embedded_sentence1 = self._seq2vec_encoder(
            embedded_sentence1, mask=sentence1_mask
        )
        embedded_sentence2 = self._seq2vec_encoder(
            embedded_sentence2, mask=sentence2_mask
        )
        pair_vec = self._pair_vec_to_vec(embedded_sentence1, embedded_sentence2)
        if self._dropout:
            pair_vec = self._dropout(pair_vec)

        logits = self._classification_layer(pair_vec)
        probs = torch.softmax(logits, dim=-1)

        output_dict = {
            "logits": logits,
            "probs": probs,
            "sentence1_token_ids": get_token_ids_from_text_field_tensors(sentence1),
            "sentence2_token_ids": get_token_ids_from_text_field_tensors(sentence2),
        }
        if label is not None:
            loss = self._loss(logits, label.long().view(-1))
            output_dict["loss"] = loss
            self._accuracy(logits, label)
        return output_dict

    def get_metrics(self, reset: bool = False) -> Dict[str, float]:
        metrics = {"accuracy": self._accuracy.get_metric(reset)}
        return metrics
