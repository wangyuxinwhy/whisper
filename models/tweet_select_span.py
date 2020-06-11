# -*- coding: utf-8 -*-
from typing import Any, Dict, List, Optional

import torch
from allennlp.common.util import sanitize_wordpiece
from torch import nn as nn

from allennlp.models import Model, load_archive
from allennlp.data import Vocabulary
from allennlp.nn import util
from allennlp.modules import Seq2SeqEncoder
from allennlp.modules.span_extractors import SpanExtractor
from allennlp.modules.text_field_embedders import BasicTextFieldEmbedder
from allennlp.modules.token_embedders import PretrainedTransformerEmbedder
from allennlp.training.metrics import CategoricalAccuracy

from whisper.training.metrics import Jaccard
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
        self._processed_span_accuracy = CategoricalAccuracy()
        self._candidate_jaccard = Jaccard()

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
            output_dict["best_candidate_span_str"] = []
            logits_temp = logits.data.clone()
            for idx, meta in enumerate(metadata):
                if not meta["have_truth"]:
                    # print(idx, logits_temp, candidate_span_pairs)
                    logits_temp[idx, meta["candidate_num"]-1] = 1e-32
            processed_probs = torch.softmax(logits_temp, -1)
            processed_best_span_index = processed_probs.max(dim=-1).indices
            processed_best_span = util.batched_index_select(
                candidate_span_pairs, processed_best_span_index
            ).squeeze()
            self._processed_span_accuracy(processed_probs, label)

            for idx, meta in enumerate(metadata):
                text_with_sentiment_tokens = meta["text_with_sentiment_tokens"]
                predicted_start, predicted_end = tuple(processed_best_span[idx])
                if predicted_end >= len(text_with_sentiment_tokens):
                    predicted_end = len(text_with_sentiment_tokens) - 1
                best_span_string = self.span_tokens_to_text(
                    meta["text"],
                    text_with_sentiment_tokens,
                    predicted_start,
                    predicted_end,
                )
                output_dict["best_candidate_span_str"].append(best_span_string)
                answers = meta.get("selected_text", "")
                if len(answers) > 0:
                    self._candidate_jaccard(best_span_string, answers)

            self._span_accuracy(probs, label)
            if not self._smoothing:
                loss = self._loss(logits, label)
                output_dict["loss"] = loss
            else:
                golden_span = util.batched_index_select(candidate_span_pairs, label)
                span_jaccard = batch_span_jaccard(candidate_span_pairs, golden_span)
                prob_distribution = span_jaccard / span_jaccard.sum(-1, keepdim=True)
                log_probs = logits - torch.log(torch.exp(logits).sum(-1)).unsqueeze(-1)
                loss = self._loss(log_probs, prob_distribution)

                output_dict["loss"] = loss
        return output_dict

    def get_metrics(self, reset: bool = False) -> Dict[str, float]:
        return {"span_accuracy": self._span_accuracy.get_metric(reset),
                "t_jaccard": self._candidate_jaccard.get_metric(reset),
                "t_accuracy": self._processed_span_accuracy.get_metric(reset)}

    @staticmethod
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