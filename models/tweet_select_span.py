# -*- coding: utf-8 -*-
from typing import Any, Dict, List, Optional

import torch
from torch import nn as nn

from allennlp.models import Model, load_archive
from allennlp.data import Vocabulary
from allennlp.nn import util
from allennlp.modules import Seq2SeqEncoder
from allennlp.modules.span_extractors import SpanExtractor
from allennlp.modules.text_field_embedders import BasicTextFieldEmbedder
from allennlp.modules.token_embedders import PretrainedTransformerEmbedder
from allennlp.training.metrics import CategoricalAccuracy

from whisper.nn.util import get_span_pairs_field_mask, batch_span_jaccard


@Model.register("tweet_select_span")
class TweetSelectSpan(Model):
    def __init__(
        self,
        vocab: Vocabulary,
        span_extractor: SpanExtractor,
        transformer_model_name_or_archive_path: str = "bert-base-uncased",
        encoder: Optional[Seq2SeqEncoder] = None,
        freeze: bool = False,
        smoothing: bool = False,
        drop_out: float = 0.0,
        **kwargs
    ) -> None:
        super().__init__(vocab, **kwargs)
        self._span_extractor = span_extractor

        # text_field_embedder
        if "model.tar.gz" in transformer_model_name_or_archive_path:
            archive = load_archive(transformer_model_name_or_archive_path)
            self._text_field_embedder = archive.extract_module(
                "_text_field_embedder", freeze
            )
        else:
            self._text_field_embedder = BasicTextFieldEmbedder(
                {
                    "tokens": PretrainedTransformerEmbedder(
                        transformer_model_name_or_archive_path
                    )
                }
            )
            if freeze:
                for parameter in self._text_field_embedder.parameters():  # type: ignore
                    parameter.requires_grad_(not freeze)

        # encoder
        if encoder is None:
            self._encoder = None
        else:
            self._encoder = encoder

        # linear
        self._linear = nn.Linear(self._span_extractor.get_output_dim(), 1)

        # drop out
        self._drop_out = nn.Dropout(drop_out)

        # loss
        self._smoothing = smoothing
        if not smoothing:
            self._loss = nn.CrossEntropyLoss()
        else:
            self._loss = nn.KLDivLoss(reduction="batchmean")

        # metric
        self._span_accuracy = CategoricalAccuracy()

    def forward(
        self,
        text_with_sentiment: Dict[str, Dict[str, torch.LongTensor]],
        candidate_span_pairs: torch.LongTensor,
        label: Optional[torch.IntTensor] = None,
        metadata: List[Dict[str, Any]] = None,
    ) -> Dict[str, torch.Tensor]:
        embedded_text_input = self._text_field_embedder(text_with_sentiment)

        mask = util.get_text_field_mask(text_with_sentiment)
        candidate_span_mask = get_span_pairs_field_mask(candidate_span_pairs)
        if self._encoder is not None:
            # shape: (batch_size, seq_length, encoder_dim)
            encoded_text = self._encoder(embedded_text_input, mask)
        else:
            encoded_text = embedded_text_input

        # shape: (batch_size, num_pair, span_extractor_dim)
        candidate_span_vec = self._span_extractor(
            encoded_text, candidate_span_pairs, span_indices_mask=candidate_span_mask
        )

        # shape: (batch_size, num_pair)
        logits = self._linear(candidate_span_vec).squeeze()
        logits.masked_fill_(~candidate_span_mask, 1e-32)
        probs = torch.softmax(logits, -1)
        best_span_index = probs.max(dim=-1).indices
        best_span = util.batched_index_select(
            candidate_span_pairs, best_span_index
        ).squeeze()
        output_dict = {"span_probs": probs, "best_span": best_span, "span_logits": logits}

        if label is not None:
            self._span_accuracy(probs, label)
            if not self._smoothing:
                loss = self._loss(logits, label)
                output_dict["loss"] = loss
            else:
                golden_span = util.batched_index_select(candidate_span_pairs, label)
                span_jaccard = batch_span_jaccard(candidate_span_pairs, golden_span)
                prob_distribution = span_jaccard / span_jaccard.sum(-1, keepdim=True)
                log_probs = torch.log(probs)
                loss = self._loss(log_probs, prob_distribution)
                output_dict["loss"] = loss
        return output_dict

    def get_metrics(self, reset: bool = False) -> Dict[str, float]:
        return {"span_accuracy": self._span_accuracy.get_metric(reset)}
