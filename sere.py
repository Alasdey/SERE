import os
import shutil

from datetime import datetime
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
from utils.cot_synthesis import synthesize_cot_for_examples, get_cached_cot

random.seed(42)

load_dotenv('env.env')


def path_to_dataset(dataset_path: str) -> list[dict[str, Any]]:
    with open(dataset_path, mode='r', encoding='utf-8') as f:
        return [json.loads(line) for line in f]


def print_prf1(test_set: list[dict[str, Any]]) -> None:
    tp = sum(1 for d in test_set if d['pred'] == 1 and d['ground'] == 1)
    fp = sum(1 for d in test_set if d['pred'] == 1 and d['ground'] == 0)
    fn = sum(1 for d in test_set if d['pred'] == 0 and d['ground'] == 1)

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    print(f'Precision: {precision:.4f}')
    print(f'Recall:    {recall:.4f}')
    print(f'F1:        {f1:.4f}')


def test_all(full_test_path: str, full_train_path: str, output_path: str, inference_llm_name: str,
             preprocess_llm_name: str, k: int,
             use_cot: bool = False, cot_cache_path: str | None = None) -> None:
    llm = get_llm(inference_llm_name)
    weighted_unified_retriever = WeightedUnifiedRetriever(llm_name=preprocess_llm_name)

    def test_loop(data: dict[str, Any]) -> None:
        """
        used for multi thread
        """

        unified_weighted_examples = weighted_unified_retriever.retrieve_samples(data, train_set, k)

        prompt = predict_by_structured_examples_prompt(data['input_text'],
                                                             data['source'],
                                                             data['target'],
                                                             unified_weighted_examples,
                                                             use_cot=use_cot)

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

    if use_cot:
        # Synthesize CoT once for the deduplicated set of training examples that will actually
        # be retrieved as fewshots across the whole test set, then re-attach via the disk/memory
        # cache so every test_loop call reuses the same reasoning per example.
        needed_ids = set()
        needed_examples = []
        for data in test_set:
            for e in weighted_unified_retriever.retrieve_samples(data, train_set, k):
                if e['unique_id'] not in needed_ids:
                    needed_ids.add(e['unique_id'])
                    needed_examples.append(e)

        synthesize_cot_for_examples(needed_examples, llm, cache_path=cot_cache_path)

        for e in train_set:
            e['cot'] = get_cached_cot(e['unique_id'])

    with tqdm_joblib(total=len(test_set)):
        Parallel(n_jobs=10, backend='threading')(delayed(test_loop)(item) for item in test_set)

    results_dir = os.path.dirname(output_path)
    os.makedirs(results_dir, exist_ok=True)
    with open(output_path, mode='w', encoding='utf-8') as f:
        f.writelines(json.dumps(data, ensure_ascii=False) + '\n' for data in test_set)

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    snapshot_dir = f'{results_dir}_{timestamp}'
    shutil.copytree(results_dir, snapshot_dir)

    config = {
        'full_test_path': full_test_path,
        'full_train_path': full_train_path,
        'output_path': output_path,
        'inference_llm_name': inference_llm_name,
        'preprocess_llm_name': preprocess_llm_name,
        'k': k,
        'use_cot': use_cot,
        'cot_cache_path': cot_cache_path,
        'timestamp': timestamp,
    }
    with open(os.path.join(snapshot_dir, 'config.json'), mode='w', encoding='utf-8') as f:
        json.dump(config, f, ensure_ascii=False, indent=2)

    print_prf1(test_set)


if __name__ == '__main__':
    # DATASET_NAME = 'CTB'
    # DATASET_NAME = 'ESC - inter'
    DATASET_NAME = 'ESC - intra'

    METHOD_NAME = 'weighted_unified'

    # INFERENCE_LLM_NAME = os.environ.get('GPT_4O_MINI_NAME')
    # INFERENCE_LLM_NAME = "openai/gpt-4o-mini"
    # INFERENCE_LLM_NAME = "deepseek/deepseek-v4-pro"
    INFERENCE_LLM_NAME = "deepseek/deepseek-v3.2"
    PREPROCESS_LLM_NAME = os.environ.get('GPT_4O_MINI_NAME')

    FULL_TEST_PATH = 'dataset/{dataset_name}/full_test.jsonl'.format(dataset_name=DATASET_NAME)
    FULL_TRAIN_PATH = 'dataset/{dataset_name}/full_train.jsonl'.format(dataset_name=DATASET_NAME)

    OUTPUT_PATH = 'dataset/{dataset_name}/results/{method_name}.jsonl'.format(dataset_name=DATASET_NAME, method_name=METHOD_NAME)

    K = 5

    USE_COT = True
    COT_CACHE_PATH = 'dataset/{dataset_name}/cot_cache.json'.format(dataset_name=DATASET_NAME)

    test_all(FULL_TEST_PATH, FULL_TRAIN_PATH, OUTPUT_PATH, INFERENCE_LLM_NAME, PREPROCESS_LLM_NAME, K,
             use_cot=USE_COT, cot_cache_path=COT_CACHE_PATH)
