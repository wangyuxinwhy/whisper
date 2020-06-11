# -*- coding: utf-8 -*-

from typing import Union, Tuple, Dict, List

from allennlp.training.metrics import Metric

from overrides import overrides


@Metric.register("loss_log")
class LossLog(Metric):
    def __init__(self):
        self._loss = 0.0
        self._count = 0

    @overrides
    def __call__(self, loss):
        loss = float(loss)
        self._loss += loss
        self._count += 1

    def get_metric(
        self, reset: bool
    ) -> Union[float, Tuple[float, ...], Dict[str, float], Dict[str, List[float]]]:
        if self._count > 0:
            score = self._loss / float(self._count)
        else:
            score = 0.0
        if reset:
            self.reset()
        return score

    @overrides
    def reset(self):
        self._loss = 0.0
        self._count = 0
