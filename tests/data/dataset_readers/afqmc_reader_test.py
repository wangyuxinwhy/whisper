# -*- coding: utf-8 -*-
from allennlp.common.util import ensure_list

from whisper.common.testing import WhisperNlpTestCase
from whisper.dataset_readers import AfqmcDatasetReader


class TestAfqmcDatasetReader:
    def test_read_from_file(self):
        reader = AfqmcDatasetReader()
        instances = ensure_list(
            reader.read(WhisperNlpTestCase.FIXTURES_ROOT / "data" / "afqmc.json")
        )
        assert len(instances) == 5

        assert instances[0].fields["sentence1"].tokens[0].text == "双十"
        assert instances[0].fields["sentence2"].tokens[0].text == "里"
        assert instances[0].fields["label"].label == "0"
