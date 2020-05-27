# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
from torch import Tensor
from torch.nn import Module


class LinearPairVecToVec(Module):
    def __init__(self, vec1_dim: int, vec2_dim: int, output_dim: int) -> None:
        super().__init__()
        self._vec1_dim = vec1_dim
        self._vec2_dim = vec2_dim
        self._output_dim = output_dim
        self.linear1 = nn.Linear(self._vec1_dim, self._output_dim)
        self.linear2 = nn.Linear(self._vec2_dim, self._output_dim)

    def forward(self, vec1: Tensor, vec2: Tensor) -> Tensor:
        affine_vec1 = self.linear1(vec1)
        affine_vec2 = self.linear2(vec2)
        return torch.relu(affine_vec1 + affine_vec2)
