# -*- coding: utf-8 -*-
import os
import shutil
import tarfile
import tempfile
import atexit

from allennlp.common import Params

from allennlp.common.file_utils import cached_path

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
