# -*- coding: utf-8 -*-
from typing import List, Optional
from overrides import overrides

import jieba

from allennlp.data.tokenizers.token import Token
from allennlp.data.tokenizers.tokenizer import Tokenizer


class JiebaTokenizer(Tokenizer):
    def __init__(self, user_dict: Optional[str] = None) -> None:
        self._tokenizer = jieba.Tokenizer()
        if user_dict is not None:
            self._tokenizer.load_userdict(user_dict)

    @overrides
    def tokenize(self, text: str) -> List[Token]:
        return [Token(t) for t in self._tokenizer.lcut(text)]
