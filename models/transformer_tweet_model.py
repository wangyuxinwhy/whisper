# -*- coding: utf-8 -*-
import logging
from typing import Any, Dict, List, Optional

import torch
from torch import nn as nn
from torch.nn.functional import cross_entropy

from allennlp.common.util import sanitize_wordpiece
from allennlp.modules.text_field_embedders import BasicTextFieldEmbedder
from allennlp.modules.token_embedders import PretrainedTransformerEmbedder
from allennlp.nn.util import get_token_ids_from_text_field_tensors
from allennlp.data import Vocabulary
from allennlp.models.model import Model
from allennlp.nn import util
from allennlp.training.metrics import BooleanAccuracy, CategoricalAccuracy

from ..common import get_best_span
from ..training.metrics import Jaccard


logger = logging.getLogger(__name__)


@Model.register("transformer_tweet")
class TransformerTweet(Model):
    """
    This class implements a reading comprehension model patterned after the proposed model in
    https://arxiv.org/abs/1810.04805 (Devlin et al), with improvements borrowed from the SQuAD model in the
    transformers project.
    It predicts start tokens and end tokens with a linear layer on top of word piece embeddings.
    Note that the metrics that the model produces are calculated on a per-instance basis only. Since there could
    be more than one instance per question, these metrics are not the official numbers on the SQuAD task. To get
    official numbers, run the script in scripts/transformer_qa_eval.py.
    Parameters
    ----------
    vocab : ``Vocabulary``
    transformer_model_name : ``str``, optional (default=``bert-base-cased``)
        This model chooses the embedder according to this setting. You probably want to make sure this is set to
        the same thing as the reader.
    """

    def __init__(
        self,
        vocab: Vocabulary,
        transformer_model_name: str = "bert-base-uncased",
        jointly: bool = False,
        dropout: Optional[float] = None,
        **kwargs,
    ) -> None:
        super().__init__(vocab, **kwargs)
        self._text_field_embedder = BasicTextFieldEmbedder(
            {"tokens": PretrainedTransformerEmbedder(transformer_model_name)}
        )
        self._linear_layer = nn.Linear(self._text_field_embedder.get_output_dim(), 2)
        self.jointly = jointly

        if dropout is not None:
            self._dropout = nn.Dropout(dropout)
        else:
            self._dropout = None

        self._span_start_accuracy = CategoricalAccuracy()
        self._span_end_accuracy = CategoricalAccuracy()
        self._span_accuracy = BooleanAccuracy()
        self._jaccard = Jaccard()

    def forward(  # type: ignore
        self,
        text: Dict[str, Dict[str, torch.LongTensor]],
        sentiment: torch.IntTensor,
        text_with_sentiment: Dict[str, Dict[str, torch.LongTensor]],
        text_span: torch.IntTensor,
        selected_text_span: Optional[torch.IntTensor] = None,
        metadata: List[Dict[str, Any]] = None,
    ) -> Dict[str, torch.Tensor]:

        embedded_question = self._text_field_embedder(text_with_sentiment)
        if self._dropout is not None:
            embedded_question = self._dropout(embedded_question)

        logits = self._linear_layer(embedded_question)
        span_start_logits, span_end_logits = logits.split(1, dim=-1)
        span_start_logits = span_start_logits.squeeze(-1)
        span_end_logits = span_end_logits.squeeze(-1)

        possible_answer_mask = torch.zeros_like(
            get_token_ids_from_text_field_tensors(text_with_sentiment)
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

            start_loss = cross_entropy(span_start_logits, span_start, ignore_index=-1)
            if torch.any(start_loss > 1e9):
                logger.critical("Start loss too high (%r)", start_loss)
                logger.critical("span_start_logits: %r", span_start_logits)
                logger.critical("span_start: %r", span_start)
                assert False

            end_loss = cross_entropy(span_end_logits, span_end, ignore_index=-1)
            if torch.any(end_loss > 1e9):
                logger.critical("End loss too high (%r)", end_loss)
                logger.critical("span_end_logits: %r", span_end_logits)
                logger.critical("span_end: %r", span_end)
                assert False

            loss = (start_loss + end_loss) / 2

            self._span_start_accuracy(span_start_logits, span_start, span_mask)
            self._span_end_accuracy(span_end_logits, span_end, span_mask)

            output_dict["loss"] = loss

        # Compute Jaccard
        if metadata is not None:
            best_spans = best_spans.detach().cpu().numpy()

            output_dict["best_span_str"] = []
            for metadata_entry, best_span, cspan in zip(
                metadata, best_spans, text_span
            ):
                text_with_sentiment_tokens = metadata_entry[
                    "text_with_sentiment_tokens"
                ]

                predicted_start, predicted_end = tuple(best_span)
                while (
                    predicted_start >= 0
                    and text_with_sentiment_tokens[predicted_start].idx is None
                ):
                    predicted_start -= 1
                if predicted_start < 0:
                    logger.warning(
                        f"Could not map the token '{text_with_sentiment_tokens[best_span[0]].text}' at index "
                        f"'{best_span[0]}' to an offset in the original text."
                    )
                    character_start = 0
                else:
                    character_start = text_with_sentiment_tokens[predicted_start].idx

                while (
                    predicted_end < len(text_with_sentiment_tokens)
                    and text_with_sentiment_tokens[predicted_end].idx is None
                ):
                    predicted_end += 1
                if predicted_end >= len(text_with_sentiment_tokens):
                    logger.warning(
                        f"Could not map the token '{text_with_sentiment_tokens[best_span[1]].text}' at index "
                        f"'{best_span[1]}' to an offset in the original text."
                    )
                    character_end = len(metadata_entry["context"])
                else:
                    end_token = text_with_sentiment_tokens[predicted_end]
                    if end_token.idx == 0:
                        character_end = (
                            end_token.idx + len(sanitize_wordpiece(end_token.text)) + 1
                        )
                    else:
                        character_end = end_token.idx + len(
                            sanitize_wordpiece(end_token.text)
                        )

                best_span_string = metadata_entry["text"][
                    character_start:character_end
                ].strip()
                output_dict["best_span_str"].append(best_span_string)

                answers = metadata_entry.get("selected_text", "")
                if len(answers) > 0:
                    self._jaccard(best_span_string, answers)

        return output_dict

    def get_metrics(self, reset: bool = False) -> Dict[str, float]:
        jaccard = self._jaccard.get_metric(reset)
        return {
            "start_acc": self._span_start_accuracy.get_metric(reset),
            "end_acc": self._span_end_accuracy.get_metric(reset),
            "span_acc": self._span_accuracy.get_metric(reset),
            "jaccard": jaccard,
        }
