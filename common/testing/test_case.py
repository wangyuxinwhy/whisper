# -*- coding: utf-8 -*-
import pathlib


class WhisperNlpTestCase:
    PROJECT_ROOT = (pathlib.Path(__file__).parent / ".." / ".." / "..").resolve()
    MODULE_ROOT = PROJECT_ROOT / "whisper"
    TOOLS_ROOT = MODULE_ROOT / "tools"
    TESTS_ROOT = MODULE_ROOT / "tests"
    FIXTURES_ROOT = TESTS_ROOT / "fixtures"
