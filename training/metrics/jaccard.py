# -*- coding: utf-8 -*-
from typing import Union, Tuple, Dict, List

from allennlp.training.metrics import Metric

from overrides import overrides


@Metric.register("jaccard")
class Jaccard(Metric):

    def __init__(self):
        self._jaccard = 0.0
        self._count = 0.0

    @overrides
    def __call__(self, prediction_str, target_str):
        a = set(prediction_str.lower().split())
        b = set(target_str.lower().split())
        c = a.intersection(b)
        self._jaccard += float(len(c)) / (len(a) + len(b) - len(c))
        self._count += 1

    def get_metric(
        self, reset: bool
    ) -> Union[float, Tuple[float, ...], Dict[str, float], Dict[str, List[float]]]:
        if self._count > 1e-12:
            score = float(self._jaccard) / float(self._count)
        else:
            score = 0.0
        if reset:
            self.reset()
        return score

    @overrides
    def reset(self):
        self._jaccard = 0.0
        self._count = 0.0
