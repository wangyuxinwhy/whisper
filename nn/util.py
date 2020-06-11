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
    dividend = torch.clamp_min_(diff_sum + golden_span_width.unsqueeze(1), 0)
    divisor = (candidate_span_width - dividend) + golden_span_width.unsqueeze(1)
    span_jaccard = torch.true_divide(dividend, divisor)
    return span_jaccard


def get_sequence_distance_from_span_endpoint(sequence_length: int, span_endpoint: torch.Tensor):
    batch_size = span_endpoint.size(0)
    distance_range = torch.arange(0, sequence_length, device=span_endpoint.device).repeat(batch_size, 1)
    distance_part1 = (distance_range - span_endpoint.unsqueeze(1)).clamp_min(0)
    distance_part2 = -(distance_range - span_endpoint.unsqueeze(1)).clamp_max(0)
    distance = distance_part1 + distance_part2
    return distance
