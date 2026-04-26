import os

from typing import Any
import json
import re
import random

from dotenv import load_dotenv
from joblib import Parallel, delayed
from tqdm_joblib import tqdm_joblib

from utils.retrievers import WeightedUnifiedRetriever
from utils.prompt import predict_by_structured_examples_prompt
from utils.llm import get_llm

random.seed(42)

load_dotenv('env.env')

INFERENCE_LLM_NAME = os.environ.get('GPT_4O_MINI_NAME')
PREPROCESS_LLM_NAME = os.environ.get('GPT_4O_MINI_NAME')

llm = get_llm(INFERENCE_LLM_NAME)

weighted_unified_retriever = WeightedUnifiedRetriever(llm_name=PREPROCESS_LLM_NAME)


def path_to_dataset(dataset_path: str) -> list[dict[str, Any]]:
    with open(dataset_path, mode='r', encoding='utf-8') as f:
        return [json.loads(line) for line in f]


def test_all(full_test_path: str, full_train_path: str, output_path: str, inference_llm_name: str, k: int) -> None:
    def test_loop(data: dict[str, Any]) -> None:
        """
        used for multi thread
        """

        unified_weighted_examples = weighted_unified_retriever.retrieve_samples(data, train_set, k)

        prompt = predict_by_structured_examples_prompt(data['input_text'],
                                                             data['source'],
                                                             data['target'],
                                                             unified_weighted_examples)

        llm_resp, token_cost = llm.response(prompt)
        try:
            json_pattern = r'\{[^\{\}]*\}'
            parsed_resp = json.loads(re.findall(json_pattern, llm_resp)[-1])
            answer = parsed_resp['Answer']

            pred_str = answer
        except:
            pred_str = 'No'

        if pred_str == 'Yes':
            pred = 1
        else:
            pred = 0

        data[f'{inference_llm_name}_pred_resp'] = llm_resp
        data[f'{inference_llm_name}_answer'] = pred_str
        data['pred'] = pred

    test_set = path_to_dataset(full_test_path)
    train_set = path_to_dataset(full_train_path)

    with tqdm_joblib(total=len(test_set)):
        Parallel(n_jobs=10, backend='threading')(delayed(test_loop)(item) for item in test_set)

    with open(output_path, mode='w', encoding='utf-8') as f:
        f.writelines(json.dumps(data, ensure_ascii=False) + '\n' for data in test_set)


if __name__ == '__main__':
    DATASET_NAME = 'CTB'
    # DATASET_NAME = 'ESC - inter'
    # DATASET_NAME = 'ESC - intra'

    METHOD_NAME = 'weighted_unified'

    FULL_TEST_PATH = r'dataset\{dataset_name}\full_test.jsonl'.format(dataset_name=DATASET_NAME)
    FULL_TRAIN_PATH = r'dataset\{dataset_name}\full_train.jsonl'.format(dataset_name=DATASET_NAME)

    OUTPUT_PATH = r'dataset\{dataset_name}\results\{method_name}.jsonl'.format(dataset_name=DATASET_NAME, method_name=METHOD_NAME)

    K = 2

    test_all(FULL_TEST_PATH, FULL_TRAIN_PATH, OUTPUT_PATH, PREPROCESS_LLM_NAME, K)
