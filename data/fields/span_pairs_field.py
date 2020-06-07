# -*- coding: utf-8 -*-
from typing import Dict, List, Tuple

from overrides import overrides
import torch

from allennlp.data.fields.sequence_field import SequenceField
from allennlp.common.util import pad_sequence_to_length

Pair = Tuple[int, int]


class SpanPairsField(SequenceField[torch.Tensor]):
    def __init__(self, span_pairs: List[Pair], sequence_field: SequenceField) -> None:
        valid_span_pairs = []
        self.sequence_field = sequence_field

        for span_start, span_end in span_pairs:

            if not isinstance(span_start, int) or not isinstance(span_end, int):
                raise TypeError(
                    f"SpanFields must be passed integer indices. Found span indices: "
                    f"({span_start}, {span_end}) with types "
                    f"({type(span_start)} {type(span_end)})"
                )
            # if span_start > span_end:
            #     raise ValueError(
            #         f"span_start must be less than span_end, "
            #         f"but found ({span_start}, {span_end})."
            #     )
            #
            # if span_end > self.sequence_field.sequence_length() - 1:
            #     raise ValueError(
            #         f"span_end must be <= len(sequence_length) - 1, but found "
            #         f"{span_end} and {self.sequence_field.sequence_length() - 1} respectively."
            #     )
            if (not span_start > span_end) and (not span_end > self.sequence_field.sequence_length() - 1):
                valid_span_pairs.append(tuple((span_start, span_end)))
        self.span_pairs = valid_span_pairs

    @overrides
    def get_padding_lengths(self) -> Dict[str, int]:
        return {"span_pairs": len(self.span_pairs)}

    @overrides
    def as_tensor(self, padding_lengths: Dict[str, int]) -> torch.Tensor:
        desired_length = padding_lengths["span_pairs"]
        padded_span_pairs = pad_sequence_to_length(
            self.span_pairs, desired_length, lambda: (-1, -1)
        )
        tensor = torch.tensor(padded_span_pairs).long()
        return tensor

    @overrides
    def sequence_length(self) -> int:
        return len(self.span_pairs)

    @overrides
    def empty_field(self):
        return SpanPairsField([(-1, -1)], self.sequence_field.empty_field())

    def __str__(self) -> str:
        return f"SpanPairsField with spans: {self.span_pairs[:3]} ..."

    def __eq__(self, other) -> bool:
        if isinstance(other, list) and len(other) == len(self.span_pairs):
            set_self = set(self.span_pairs)
            set_other = set(other)
            return set_self == set_other
        return super().__eq__(other)

    def __len__(self):
        return len(self.span_pairs)
