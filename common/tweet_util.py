# -*- coding: utf-8 -*-
from tqdm import tqdm

BATCH_SIZE = 32


# just fort tweet dataset
def predict_test_data(dataframe, predictor, batch_size=32):

    if "selected_text" in dataframe.columns:
        is_dev = True
    else:
        is_dev = False

    results = []
    model_predict_outputs = []
    for i in tqdm(list(range(0, dataframe.shape[0], batch_size))):
        records = dataframe.iloc[i : i + batch_size].to_dict("records")

        instances = []
        true_spans = []

        for record in records:
            instance = predictor._json_to_instance(record)
            if is_dev:
                truth_span = (
                    instance.fields["selected_text_span"].span_start,
                    instance.fields["selected_text_span"].span_end,
                )
                true_spans.append(truth_span)
            instances.append(instance)

        batch_output = predictor.predict_batch_instance(instances)
        model_predict_outputs.extend(batch_output)
        for j in range(len(records)):
            record = records[j]
            output = batch_output[j]

            result = {
                "textID": record["textID"],
                "text": record["text"],
                "sentiment": record["sentiment"],
                "best_span": output["best_span"],
                "best_span_str": output["best_span_str"],
                "sentiment_predicts": output.get("sentiment_predicts"),
                "candidate_spans": output.get("candidate_spans"),
                "candidate_best_spans": output.get("candidate_best_spans"),
                "best_candidate_span_str": output.get("best_candidate_span_str"),
                "span_classification_probs": output.get("span_classification_probs"),
            }

            if is_dev:
                result["selected_text"] = record["selected_text"]
                result["selected_text_span"] = true_spans[j]

            results.append(result)
    return results, model_predict_outputs


def simple_jaccard(str1, str2):
    str1_set = set(str1.strip().split(" "))
    str2_set = set(str2.strip().split(" "))
    set3 = str1_set.intersection(str2_set)
    return len(set3) / (len(str1_set) + len(str2_set) - len(set3))
