from typing import Any
import random


class WeightedUnifiedRetriever:
    def __init__(self, llm_name: str) -> None:
        self.__llm_name = llm_name

    def retrieve_samples(self, data: dict[str, Any], train_dataset: list[dict[str, Any]], k: int) -> list[dict[str, Any]]:
        examples_key = f'filtered_weighted_unified_examples_{self.__llm_name}_structured_pattern'
        examples_id = data[examples_key]

        trainset_map = {data['unique_id']: data for data in train_dataset}
        examples_full = [trainset_map[uid] for uid in examples_id]

        pos_k = k // 2
        neg_k = k - pos_k
        pos_examples_full = [e for e in examples_full if e['ground'] == 1]
        neg_examples_full = [e for e in examples_full if e['ground'] == 0]
        
        pos_examples = pos_examples_full[:pos_k]
        neg_examples = neg_examples_full[:neg_k]
        
        final_examples = pos_examples + neg_examples

        if k < len(final_examples):
            final_examples = final_examples[:k]
        if k > len(final_examples):
            final_examples_ids = {fe['unique_id'] for fe in final_examples}
            remaining_examples = [e for e in examples_full if e['unique_id'] not in final_examples_ids]
            final_examples = final_examples + random.sample(remaining_examples, min(k - len(final_examples), len(remaining_examples)))

        return final_examples
