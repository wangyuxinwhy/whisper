# -*- coding: utf-8 -*-
import logging
from typing import Any, Dict, List, Optional, Tuple

import torch
from torch import nn as n

from allennlp.models import Model
from allennlp.data import Vocabulary

@Model.register("tweet_select_span")
class TweetSelectSpan(Model):
    def __init__(
        self,
        vocab: Vocabulary,
        fine_tuned_model_tar_path: str = "",
        freeze_encoder: bool = True,

    ):