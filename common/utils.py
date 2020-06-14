# -*- coding: utf-8 -*-
import json
import os
import shutil
import tarfile
import tempfile
import atexit
from pathlib import Path
from typing import Optional, List
from collections import defaultdict

import torch
import pandas as pd

from allennlp.common import Params
from allennlp.common.file_utils import cached_path
from allennlp.predictors import Predictor
from allennlp.models import load_archive
from tqdm import tqdm

from .rc_utils import get_candidate_span
from .tweet_util import predict_test_data

CONFIG_NAME = "config.json"


def extract_config_from_archive(archive_file: str, overrides: str = "") -> Params:
    # redirect to the cache, if necessary
    resolved_archive_file = cached_path(archive_file)

    if os.path.isdir(resolved_archive_file):
        serialization_dir = resolved_archive_file
    else:
        # Extract archive to temp dir
        tempdir = tempfile.mkdtemp()
        with tarfile.open(resolved_archive_file, "r:gz") as archive:
            archive.extractall(tempdir)
        # Postpone cleanup until exit in case the unarchived contents are needed outside
        # this function.
        atexit.register(_cleanup_archive_dir, tempdir)

        serialization_dir = tempdir

    # Load config
    config = Params.from_file(os.path.join(serialization_dir, CONFIG_NAME), overrides)
    return config


def _cleanup_archive_dir(path: str):
    if os.path.exists(path):
        shutil.rmtree(path)


def gen_cv_json(base_json_path, cv_num=5):
    base_path = Path(base_json_path).resolve()
    for i in range(1, cv_num):
        with open(str(base_path.parent / f"cv{i}.json"), "w") as f:
            text = base_path.read_text()
            text = text.replace("train_cv0", f"train_cv{i}")
            text = text.replace("dev_cv0", f"dev_cv{i}")
            f.write(text)


# def get_cv_metric(cv_models_dir, cv_num=5):
#     dir_path = Path(cv_models_dir).resolve()
#     best_epochs = []
#     best_validation_span_acc = []
#     best_validation_jaccard = []
#     best_validation_loss = []
#     for i in range(cv_num):
#         metrics_json_path = dir_path / f"cv{i}" / "metrics.json"
#         metrics = json.loads(metrics_json_path.read_text())
#         best_epochs.append(metrics["best_epoch"] + 1)
#         best_validation_span_acc.append(metrics["best_validation_span_acc"])
#         best_validation_jaccard.append(metrics["best_validation_jaccard"])
#         best_validation_loss.append(metrics["best_validation_loss"])
#     best_epochs_avg = sum(best_epochs) / len(best_epochs)
#     best_validation_span_acc_avg = sum(best_validation_span_acc) / len(best_validation_span_acc)
#     best_validation_jaccard_avg = sum(best_validation_jaccard) / len(best_validation_jaccard)
#     best_validation_loss_avg = sum(best_validation_loss) / len(best_validation_loss)
#     return {
#         "cv_models_dir": str(cv_models_dir),
#         "best_epochs": best_epochs,
#         "best_validation_span_acc": best_validation_span_acc,
#         "best_validation_jaccard": best_validation_jaccard,
#         "best_validation_loss": best_validation_loss,
#         "best_epochs_avg": best_epochs_avg,
#         "best_validation_span_acc_avg": best_validation_span_acc_avg,
#         "best_validation_jaccard_avg": best_validation_jaccard_avg,
#         "best_validation_loss_avg": best_validation_loss_avg
#     }


def get_cv_metric(cv_models_dir, collect_metrics, cv_num=5):
    dir_path = Path(cv_models_dir).resolve()
    output = defaultdict(list)
    for i in range(cv_num):
        metrics_json_path = dir_path / f"cv{i}" / "metrics.json"
        metrics = json.loads(metrics_json_path.read_text())
        for m in collect_metrics:
            output[m].append(metrics[m])
    return output


def generate_candidate_span(start_logits: torch.Tensor, end_logits: torch.Tensor, num_candidate : int = 10):
    if start_logits.dim() == 1:
        start_logits = start_logits.unsqueeze(0)
    if end_logits.dim() == 1:
        end_logits = end_logits.unsqueeze(0)
    tokens_num = (~(start_logits < -1e30)).sum()
    num_candidate = min(num_candidate, int(torch.arange(0, int(tokens_num)).sum()))
    spans = get_candidate_span(start_logits, end_logits, num_candidate).squeeze(0).tolist()
    return spans


def dataframe_to_json(dataframe: pd.DataFrame, save_path: str):
    dataframe.to_json(save_path, orient="records", lines=True)


def span_topk(best_span, candidate_spans, topk=1):
    if best_span in candidate_spans[:topk]:
        return True
    return False

# from allennlp.predictors import Predictor
# from allennlp.models import load_archive
#
# from whisper.common.rc_utils import get_candidate_span
# from whisper.models import TransformerTweet, TweetJointly
# from whisper.dataset_readers import TweetSentimentDatasetReader
# from whisper.predictors import TweetSentimentPredictor


def ensemble_model_avg_logits(models: List[dict], test_dataframe, weight: Optional[List] = None):
    output_dataframe = test_dataframe.copy()

    cv_outputs = []
    cv_results = []
    start_cv_logits = []
    end_cv_logits = []

    ensemble_start_logits = []
    ensemble_end_logits = []

    for model_dict in tqdm(models):
        archive = load_archive(**model_dict)
        archive.model._delay = 50000
        predictor = Predictor.from_archive(archive, "tweet_sentiment")

        results, outputs = predict_test_data(test_dataframe, predictor)
        start_logits = []
        end_logits = []
        for output in outputs:
            start_logits.append(output["span_start_logits"])
            end_logits.append(output["span_end_logits"])
        start_cv_logits.append(start_logits)
        end_cv_logits.append(end_logits)
        cv_outputs.append(outputs)
        cv_results.append(results)

    if weight is None:
        weight = [1/len(models)] * len(models)

    for i in range(test_dataframe.shape[0]):
        single_sample_start_logits = []
        single_sample_end_logits = []
        for j in start_cv_logits:
            single_sample_start_logits.append(torch.tensor(j[i]))
        for k in end_cv_logits:
            single_sample_end_logits.append(torch.tensor(k[i]))
        stack_j = torch.stack(single_sample_start_logits, dim=1)
        stack_k = torch.stack(single_sample_end_logits, dim=1)
        j_logits = torch.matmul(stack_j, torch.tensor(weight))
        k_logits = torch.matmul(stack_k, torch.tensor(weight))
        ensemble_start_logits.append(j_logits.tolist())
        ensemble_end_logits.append(k_logits.tolist())

    for idx, cv_logits in enumerate(start_cv_logits):
        output_dataframe[f"model{idx+1}_start_logits"] = cv_logits
    output_dataframe["ensemble_start_logits"] = ensemble_start_logits
    for idx, cv_logits in enumerate(end_cv_logits):
        output_dataframe[f"model{idx+1}_end_logits"] = cv_logits
    output_dataframe["ensemble_end_logits"] = ensemble_end_logits
    return output_dataframe, cv_outputs, cv_results


def sanitize_wordpiece(wordpiece: str) -> str:
    """
    Sanitizes wordpieces from BERT, RoBERTa or ALBERT tokenizers.
    """
    if wordpiece.startswith("##"):
        return wordpiece[2:]
    elif wordpiece.startswith("Ġ"):
        return wordpiece[1:]
    elif wordpiece.startswith("▁"):
        return wordpiece[1:]
    elif wordpiece.endswith("@@"):
        return wordpiece[:-2]
    else:
        return wordpiece
