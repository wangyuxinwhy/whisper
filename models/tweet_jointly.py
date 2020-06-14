# -*- coding: utf-8 -*-
import logging
from typing import Any, Dict, List, Optional, Tuple

import torch
from torch import nn as nn
from torch.nn.functional import cross_entropy

from allennlp.nn import util
from allennlp.data import Vocabulary
from allennlp.models.model import Model
from allennlp.modules import Seq2VecEncoder
from allennlp.modules.span_extractors import SpanExtractor, EndpointSpanExtractor
from allennlp.common.util import sanitize_wordpiece
from allennlp.common.checks import ConfigurationError
from allennlp.modules.text_field_embedders import BasicTextFieldEmbedder
from allennlp.modules.token_embedders import PretrainedTransformerEmbedder
from allennlp.modules import FeedForward
from allennlp.training.metrics import BooleanAccuracy, CategoricalAccuracy

from whisper.modules.token_embedders.tweet_bert_embedder import TweetBertEmbedder
from ..nn.util import get_sequence_distance_from_span_endpoint, batch_span_jaccard
from ..common import get_best_span, get_candidate_span
from ..training.metrics import Jaccard, LossLog

logger = logging.getLogger(__name__)


@Model.register("tweet_jointly")
class TweetJointly(Model):
    def __init__(
        self,
        vocab: Vocabulary,
        transformer_model_name: str = "bert-base-uncased",
        feedforward: Optional[FeedForward] = None,
        smoothing: bool = False,
        smooth_alpha: float = 0.7,
        sentiment_task: bool = False,
        sentiment_task_weight: float = 1.0,
        sentiment_classification_with_label: bool = True,
        sentiment_seq2vec: Optional[Seq2VecEncoder] = None,
        candidate_span_task: bool = False,
        candidate_span_task_weight: float = 1.0,
        candidate_delay: int = 30000,
        candidate_span_num: int = 5,
        candidate_classification_layer_units: int = 128,
        candidate_span_extractor: Optional[SpanExtractor] = None,
        candidate_span_with_logits: bool = False,
        dropout: Optional[float] = None,
        **kwargs,
    ) -> None:
        super().__init__(vocab, **kwargs)
        if "BERTweet" not in transformer_model_name:
            self._text_field_embedder = BasicTextFieldEmbedder(
                {"tokens": PretrainedTransformerEmbedder(transformer_model_name)}
            )
        else:
            self._text_field_embedder = BasicTextFieldEmbedder(
                {"tokens": TweetBertEmbedder(transformer_model_name)}
            )
        # span start & end task
        if feedforward is None:
            self._linear_layer = nn.Sequential(
                nn.Linear(self._text_field_embedder.get_output_dim(), 128),
                nn.ReLU(),
                nn.Linear(128, 2),
            )
        else:
            self._linear_layer = feedforward
        self._span_start_accuracy = CategoricalAccuracy()
        self._span_end_accuracy = CategoricalAccuracy()
        self._span_accuracy = BooleanAccuracy()
        self._jaccard = Jaccard()
        self._candidate_delay = candidate_delay
        self._delay = 0

        self._smoothing = smoothing
        self._smooth_alpha = smooth_alpha
        if smoothing:
            self._loss = nn.KLDivLoss(reduction="batchmean")
        else:
            self._loss = nn.CrossEntropyLoss()

        # sentiment task
        self._sentiment_task = sentiment_task
        if self._sentiment_task:
            self._sentiment_classification_accuracy = CategoricalAccuracy()
            self._sentiment_loss_log = LossLog()
            self.register_buffer(
                "sentiment_task_weight", torch.tensor(sentiment_task_weight)
            )
            self._sentiment_classification_with_label = (
                sentiment_classification_with_label
            )
            if sentiment_seq2vec is None:
                raise ConfigurationError(
                    "sentiment task is True, we need a sentiment seq2vec encoder"
                )
            else:
                self._sentiment_encoder = sentiment_seq2vec
                self._sentiment_linear = nn.Linear(
                    self._sentiment_encoder.get_output_dim(),
                    vocab.get_vocab_size("labels"),
                )

        # candidate span task
        self._candidate_span_task = candidate_span_task
        if candidate_span_task:
            assert candidate_span_num > 0
            assert candidate_span_task_weight > 0
            assert candidate_classification_layer_units > 0
            self._candidate_span_num = candidate_span_num
            self.register_buffer(
                "candidate_span_task_weight", torch.tensor(candidate_span_task_weight)
            )
            self._candidate_classification_layer_units = (
                candidate_classification_layer_units
            )
            self._span_classification_accuracy = CategoricalAccuracy()
            self._candidate_loss_log = LossLog()
            self._candidate_span_linear = nn.Linear(
                self._text_field_embedder.get_output_dim(),
                self._candidate_classification_layer_units,
            )

            if candidate_span_extractor is None:
                self._candidate_span_extractor = EndpointSpanExtractor(
                    input_dim=self._candidate_classification_layer_units
                )
            else:
                self._candidate_span_extractor = candidate_span_extractor

            if candidate_span_with_logits:
                self._candidate_with_logits = True
                self._candidate_span_vec_linear = nn.Linear(
                    self._candidate_span_extractor.get_output_dim() + 1, 1
                )
            else:
                self._candidate_with_logits = False
                self._candidate_span_vec_linear = nn.Linear(
                    self._candidate_span_extractor.get_output_dim(), 1
                )

            self._candidate_jaccard = Jaccard()

        if sentiment_task or candidate_span_task:
            self._base_loss_log = LossLog()
        else:
            self._base_loss_log = None

        if dropout is not None:
            self._dropout = nn.Dropout(dropout)
        else:
            self._dropout = None

    def forward(  # type: ignore
        self,
        text: Dict[str, Dict[str, torch.LongTensor]],
        sentiment: torch.IntTensor,
        text_with_sentiment: Dict[str, Dict[str, torch.LongTensor]],
        text_span: torch.IntTensor,
        selected_text_span: Optional[torch.IntTensor] = None,
        metadata: List[Dict[str, Any]] = None,
    ) -> Dict[str, torch.Tensor]:
        # batch_size * text_length * hidden_dims
        embedded_question = self._text_field_embedder(text_with_sentiment)
        if self._dropout is not None:
            embedded_question = self._dropout(embedded_question)
        self._delay += int(embedded_question.size(0))
        # span start & span end task
        logits = self._linear_layer(embedded_question)
        span_start_logits, span_end_logits = logits.split(1, dim=-1)
        span_start_logits = span_start_logits.squeeze(-1)
        span_end_logits = span_end_logits.squeeze(-1)

        possible_answer_mask = torch.zeros_like(
            util.get_token_ids_from_text_field_tensors(text_with_sentiment)
        ).bool()
        for i, (start, end) in enumerate(text_span):
            possible_answer_mask[i, start : end + 1] = True

        span_start_logits = util.replace_masked_values(
            span_start_logits, possible_answer_mask, -1e32
        )
        span_end_logits = util.replace_masked_values(
            span_end_logits, possible_answer_mask, -1e32
        )
        span_start_probs = torch.nn.functional.softmax(span_start_logits, dim=-1)
        span_end_probs = torch.nn.functional.softmax(span_end_logits, dim=-1)
        best_spans = get_best_span(span_start_logits, span_end_logits)
        best_span_scores = torch.gather(
            span_start_logits, 1, best_spans[:, 0].unsqueeze(1)
        ) + torch.gather(span_end_logits, 1, best_spans[:, 1].unsqueeze(1))
        best_span_scores = best_span_scores.squeeze(1)

        output_dict = {
            "span_start_logits": span_start_logits,
            "span_start_probs": span_start_probs,
            "span_end_logits": span_end_logits,
            "span_end_probs": span_end_probs,
            "best_span": best_spans,
            "best_span_scores": best_span_scores,
        }

        loss = torch.tensor(0.0).to(embedded_question.device)
        # sentiment task
        if self._sentiment_task:
            if self._sentiment_classification_with_label:
                global_context_vec = self._sentiment_encoder(embedded_question)
            else:
                embedded_only_text = self._text_field_embedder(text)
                if self._dropout is not None:
                    embedded_only_text = self._dropout(embedded_only_text)
                global_context_vec = self._sentiment_encoder(embedded_only_text)
            sentiment_logits = self._sentiment_linear(global_context_vec)
            sentiment_probs = torch.softmax(sentiment_logits, dim=-1)

            self._sentiment_classification_accuracy(sentiment_probs, sentiment)
            sentiment_loss = cross_entropy(sentiment_logits, sentiment)
            self._sentiment_loss_log(sentiment_loss)
            loss.add_(self.sentiment_task_weight * sentiment_loss)

            predict_sentiment_idx = sentiment_probs.argmax(dim=-1)
            sentiment_predicts = []
            for i in predict_sentiment_idx.tolist():
                sentiment_predicts.append(self.vocab.get_token_from_index(i, "labels"))
            output_dict["sentiment_logits"] = sentiment_logits
            output_dict["sentiment_probs"] = sentiment_probs
            output_dict["sentiment_predicts"] = sentiment_predicts

        # span classification
        if self._candidate_span_task and (self._delay >= self._candidate_delay):
            # shape: (batch_size, passage_length, embedding_dim)
            text_features_for_candidate = self._candidate_span_linear(embedded_question)
            text_features_for_candidate = torch.relu(text_features_for_candidate)
            with torch.no_grad():
                # batch_size * candidate_num * 2
                candidate_span = get_candidate_span(
                    span_start_probs, span_end_probs, self._candidate_span_num
                )
                candidate_span_list = candidate_span.tolist()
                output_dict["candidate_spans"] = candidate_span_list
            if selected_text_span is not None:
                candidate_span, candidate_span_label = self.candidate_span_with_labels(
                    candidate_span, selected_text_span
                )
            else:
                candidate_span_label = None
            # shape: (batch_size, candidate_num, span_extractor_output_dim)
            span_feature_vec = self._candidate_span_extractor(
                text_features_for_candidate, candidate_span
            )

            if self._candidate_with_logits:
                candidate_span_start_logits = torch.gather(
                    span_start_logits, 1, candidate_span[:, :, 0]
                )
                candidate_span_end_logits = torch.gather(
                    span_end_logits, 1, candidate_span[:, :, 1]
                )
                candidate_span_sum_logits = (
                    candidate_span_start_logits + candidate_span_end_logits
                )
                span_feature_vec = torch.cat(
                    (span_feature_vec, candidate_span_sum_logits.unsqueeze(2)), -1
                )
            # batch_size * candidate_num
            span_classification_logits = self._candidate_span_vec_linear(
                span_feature_vec
            ).squeeze()
            span_classification_probs = torch.softmax(span_classification_logits, -1)
            output_dict["span_classification_probs"] = span_classification_probs
            candidate_best_span_idx = span_classification_probs.argmax(dim=-1)
            view_idx = (
                candidate_best_span_idx
                + torch.arange(0, end=candidate_best_span_idx.shape[0]).to(
                    candidate_best_span_idx.device
                )
                * self._candidate_span_num
            )
            candidate_span_view = candidate_span.view(-1, 2)
            candidate_best_spans = candidate_span_view.index_select(0, view_idx)
            output_dict["candidate_best_spans"] = candidate_best_spans.tolist()

            if selected_text_span is not None:
                self._span_classification_accuracy(
                    span_classification_probs, candidate_span_label
                )
                candidate_span_loss = cross_entropy(
                    span_classification_logits, candidate_span_label
                )
                self._candidate_loss_log(candidate_span_loss)
                weighted_loss = self.candidate_span_task_weight * candidate_span_loss
                if candidate_span_loss > 1e2:
                    print(f"candidate loss: {candidate_span_loss}")
                    print(f"span_classification_logits: {span_classification_logits}")
                    print(f"candidate_span_label: {candidate_span_label}")
                loss.add_(weighted_loss)

            candidate_best_spans = candidate_best_spans.detach().cpu().numpy()
            output_dict["best_candidate_span_str"] = []
            for metadata_entry, best_span in zip(metadata, candidate_best_spans):
                text_with_sentiment_tokens = metadata_entry[
                    "text_with_sentiment_tokens"
                ]
                predicted_start, predicted_end = tuple(best_span)
                if predicted_end >= len(text_with_sentiment_tokens):
                    predicted_end = len(text_with_sentiment_tokens) - 1
                best_span_string = self.span_tokens_to_text(
                    metadata_entry["text"],
                    text_with_sentiment_tokens,
                    predicted_start,
                    predicted_end,
                )
                output_dict["best_candidate_span_str"].append(best_span_string)
                answers = metadata_entry.get("selected_text", "")
                if len(answers) > 0:
                    self._candidate_jaccard(best_span_string, answers)

        # Compute the loss for training.
        if selected_text_span is not None:
            span_start = selected_text_span[:, 0]
            span_end = selected_text_span[:, 1]
            span_mask = span_start != -1
            self._span_accuracy(
                best_spans,
                selected_text_span,
                span_mask.unsqueeze(-1).expand_as(best_spans),
            )
            if not self._smoothing:
                start_loss = cross_entropy(span_start_logits, span_start, ignore_index=-1)
                if torch.any(start_loss > 1e9):
                    logger.critical("Start loss too high (%r)", start_loss)
                    logger.critical("span_start_logits: %r", span_start_logits)
                    logger.critical("span_start: %r", span_start)
                    logger.critical("text_with_sentiment: %r", text_with_sentiment)
                    assert False

                end_loss = cross_entropy(span_end_logits, span_end, ignore_index=-1)
                if torch.any(end_loss > 1e9):
                    logger.critical("End loss too high (%r)", end_loss)
                    logger.critical("span_end_logits: %r", span_end_logits)
                    logger.critical("span_end: %r", span_end)
                    assert False
            else:
                sequence_length = span_start_logits.size(1)
                device = span_start.device
                start_distance = get_sequence_distance_from_span_endpoint(sequence_length, span_start)
                start_smooth_probs = torch.exp(start_distance * torch.log(torch.tensor(self._smooth_alpha).to(device)))
                start_smooth_probs = start_smooth_probs * possible_answer_mask
                start_smooth_probs = start_smooth_probs / start_smooth_probs.sum(-1, keepdim=True)
                span_start_log_probs = span_start_logits - torch.log(torch.exp(span_start_logits).sum(-1)).unsqueeze(-1)
                end_distance = get_sequence_distance_from_span_endpoint(sequence_length, span_end)
                end_smooth_probs = torch.exp(end_distance * torch.log(torch.tensor(self._smooth_alpha).to(device)))
                end_smooth_probs = end_smooth_probs * possible_answer_mask
                end_smooth_probs = end_smooth_probs / end_smooth_probs.sum(-1, keepdim=True)
                span_end_log_probs = span_end_logits - torch.log(torch.exp(span_end_logits).sum(-1)).unsqueeze(-1)
                # print(end_smooth_probs)
                # print(start_smooth_probs)
                # print(span_end_log_probs)
                # print(span_start_log_probs)
                start_loss = self._loss(span_start_log_probs, start_smooth_probs)
                end_loss = self._loss(span_end_log_probs, end_smooth_probs)

            span_start_end_loss = (start_loss + end_loss) / 2
            if self._base_loss_log is not None:
                self._base_loss_log(span_start_end_loss)
            loss.add_(span_start_end_loss)
            self._span_start_accuracy(span_start_logits, span_start, span_mask)
            self._span_end_accuracy(span_end_logits, span_end, span_mask)

            output_dict["loss"] = loss

        # compute best span jaccard
        best_spans = best_spans.detach().cpu().numpy()
        output_dict["best_span_str"] = []

        for metadata_entry, best_span in zip(metadata, best_spans):
            text_with_sentiment_tokens = metadata_entry["text_with_sentiment_tokens"]

            predicted_start, predicted_end = tuple(best_span)
            best_span_string = self.span_tokens_to_text(
                metadata_entry["text"],
                text_with_sentiment_tokens,
                predicted_start,
                predicted_end,
            )
            output_dict["best_span_str"].append(best_span_string)

            answers = metadata_entry.get("selected_text", "")
            if len(answers) > 0:
                self._jaccard(best_span_string, answers)

        return output_dict

    # @staticmethod
    # def candidate_span_with_labels(
    #     candidate_span: torch.Tensor, selected_text_span: torch.Tensor
    # ) -> Tuple[torch.Tensor, torch.Tensor]:
    #     correct_span_idx = (candidate_span == selected_text_span.unsqueeze(1)).prod(-1)
    #     candidate_span_adjust = torch.where(
    #         ~(correct_span_idx.unsqueeze(-1) == 1),
    #         candidate_span,
    #         selected_text_span.unsqueeze(1),
    #     )
    #     candidate_span_label = correct_span_idx.argmax(-1)
    #     return candidate_span_adjust, candidate_span_label

    @staticmethod
    def candidate_span_with_labels(
        candidate_span: torch.Tensor, selected_text_span: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        candidate_span_label = batch_span_jaccard(candidate_span, selected_text_span).max(-1).indices
        return candidate_span, candidate_span_label

    @staticmethod
    def get_candidate_span_mask(
        candidate_span: torch.Tensor, passage_length: int
    ) -> torch.Tensor:
        device = candidate_span.device
        batch_size, candidate_num = candidate_span.size()[:-1]
        candidate_span_mask = torch.zeros(batch_size, candidate_num, passage_length).to(
            device
        )
        for i in range(batch_size):
            for j in range(candidate_num):
                span_start, span_end = candidate_span[i][j]
                candidate_span_mask[i][j][span_start : span_end + 1] = 1
        return candidate_span_mask

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
            logger.warning(
                f"Could not map the token '{text_with_sentiment_tokens[span_start].text}' at index "
                f"'{span_start}' to an offset in the original text."
            )
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
            logger.warning(
                f"Could not map the token '{text_with_sentiment_tokens[span_end].text}' at index "
                f"'{span_end}' to an offset in the original text."
            )
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

    def get_metrics(self, reset: bool = False) -> Dict[str, float]:
        jaccard = self._jaccard.get_metric(reset)
        metrics = {
            "start_acc": self._span_start_accuracy.get_metric(reset),
            "end_acc": self._span_end_accuracy.get_metric(reset),
            "span_acc": self._span_accuracy.get_metric(reset),
            "jaccard": jaccard,
        }
        if self._candidate_span_task:
            metrics[
                "candidate_span_acc"
            ] = self._span_classification_accuracy.get_metric(reset)
            metrics["candidate_jaccard"] = self._candidate_jaccard.get_metric(reset)
            metrics["candidate_loss"] = self._candidate_loss_log.get_metric(reset)
        if self._sentiment_task:
            metrics[
                "sentiment_acc"
            ] = self._sentiment_classification_accuracy.get_metric(reset)
            metrics["sentiment_loss"] = self._sentiment_loss_log.get_metric(reset)
        if self._base_loss_log is not None:
            metrics["base_loss"] = self._base_loss_log.get_metric(reset)
        return metrics
