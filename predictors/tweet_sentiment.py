# -*- coding: utf-8 -*-

from overrides import overrides

from allennlp.data import Instance
from allennlp.common.util import JsonDict
from allennlp.predictors import Predictor


@Predictor.register("tweet_sentiment")
class TweetSentimentPredictor(Predictor):

    @overrides
    def _json_to_instance(self, json_dict: JsonDict) -> Instance:

        text = json_dict["text"]
        text = " " + text.strip()
        sentiment = json_dict["sentiment"]
        text_id = json_dict.get("TextID")
        if text_id is None:
            text_id = "<No-Text-id>"
        return self._dataset_reader.text_to_instance(text, sentiment, text_id, json_dict.get("selected_text"))
