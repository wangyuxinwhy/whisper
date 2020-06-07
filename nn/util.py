# -*- coding: utf-8 -*-

import torch


def get_span_pairs_field_mask(
    span_pairs_field_tensor: torch.LongTensor
) -> torch.Tensor:
    # shape: (batch_size, num_pair, 2)
    mask = ~(span_pairs_field_tensor[:, :, 0] == -1)
    return mask


def batch_span_jaccard(candidate_spans: torch.Tensor, golden_span: torch.Tensor):
    diff_tensor = candidate_spans - golden_span.unsqueeze(1)
    diff_tensor[:, :, 0] = -diff_tensor[:, :, 0]
    diff_tensor.clamp_max_(0)
    diff_sum = diff_tensor.sum(-1)
    golden_span_width = (golden_span[:, 1] - golden_span[:, 0]) + 1
    candidate_span_width = (candidate_spans[:, :, 1] - candidate_spans[:, :, 0]) + 1
    dividend = diff_sum + golden_span_width.unsqueeze(1)
    divisor = (candidate_span_width - dividend) + golden_span_width.unsqueeze(1)
    span_jaccard = torch.true_divide(dividend, divisor)
    return span_jaccard
