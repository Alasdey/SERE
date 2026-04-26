import random
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(__file__)))
import json
from typing import Any
import re
from tqdm import tqdm
from itertools import groupby

from joblib import Parallel, delayed
from tqdm_joblib import tqdm_joblib
from neo4j import GraphDatabase

from utils.prompt import pattern_prompt_train_pos, pattern_prompt_inference
from utils.llm import get_llm
from utils.syntactic import tree_edit_distance_similarity, DependencyTree, text_edit_distance_similarity
from utils.conceptnet.vecdb import ConceptnetVecDB


EMBEDDING_MODEL = r'YOUR_EMBEDDING_MODEL_PATH'
NODES_PATH = r'Dnodes.csv PATH FROM utils/conceptnet/to_neo4j.py'
CONCEPTNET_NODES_SAVE_PATH = r'.joblib FILE OUTPUT PATH FROM utils/conceptnet/vecdb.py'
CONCEPTNET_URI = 'CONCEPTNET_URI'
CONCEPTNET_USERNAME = 'CONCEPTNET_USERNAME'
CONCEPTNET_PASSWORD = 'CONCEPTNET_PASSWORD'


RELATION_MAP = {
    "HasContext": ["has context in", "is the context of"],
    "MotivatedByGoal": ["is motivated by the goal of", "motivates the goal of"],
    "FormOf": ["is a form of", "has the form of"],
    "SimilarTo": ["is similar to", "is similar to"],
    "HasA": ["has a", "is owned by"],
    "dbpedia": ["is associated with the DBpedia concept of", "has association from the DBpedia concept of"],
    "HasProperty": ["has the property of", "is a property of"],
    "Causes": ["causes", "is caused by"],
    "NotDesires": ["does not desire", "is not desired by"],
    "HasPrerequisite": ["requires as a prerequisite", "is a prerequisite for"],
    "PartOf": ["is part of", "has as a part"],
    "Antonym": ["is an antonym of", "is an antonym of"],  # Symmetric
    "HasLastSubevent": ["has the last subevent of", "is the last subevent of"],
    "MadeOf": ["is made of", "is the material of"],
    "HasFirstSubevent": ["has the first subevent of", "is the first subevent of"],
    "ReceivesAction": ["receives the action of", "is the action performed on"],
    "RelatedTo": ["is related to", "is related to"],  # Symmetric
    "HasSubevent": ["has the subevent of", "is a subevent of"],
    "DistinctFrom": ["is distinct from", "is distinct from"],  # Symmetric
    "InstanceOf": ["is an instance of", "has an instance of"],
    "DerivedFrom": ["is derived from", "is the origin of"],
    "UsedFor": ["is used for", "uses"],
    "MannerOf": ["is a manner of", "has as a manner"],
    "Desires": ["desires", "is desired by"],
    "IsA": ["is a", "has as a type"],
    "AtLocation": ["is located at", "is the location of"],
    "CapableOf": ["is capable of", "enables"],
    "EtymologicallyRelatedTo": ["is etymologically related to", "is etymologically related to"],  # Symmetric
    "Synonym": ["is a synonym of", "is a synonym of"],  # Symmetric
    "CreatedBy": ["is created by", "creates"],
    "CausesDesire": ["causes the desire for", "is desired because of"],
    "Entails": ["entails", "is entailed by"],
    "DefinedAs": ["is defined as", "defines"],
    "NotHasProperty": ["does not have the property of", "is not a property of"]
}

SYNTACTIC_TREE_DISTANCE_SIMILARITY_CACHE = dict()
CONCEPTNET_PATH_DISTANCE_SIMILARITY_CACHE = dict()


def random_deduplicate_item_score_pair(items: list[tuple[dict[str, Any], float]]) -> list[tuple[dict[str, Any], float]]:
    """根据item[0]['input_text_no_tag']进行分组去重"""
    items = sorted(items, key=lambda item: item[0]['input_text_no_tag'])
    return [random.choice(list(group)) for _, group in groupby(items, key=lambda item: item[0]['input_text_no_tag'])]


def path_to_dataset(dataset_path: str) -> list[dict[str, Any]]:
    with open(dataset_path, mode='r', encoding='utf-8') as f:
        return [json.loads(line) for line in f]


def add_pattern_pos(dataset_path: str, llm_name: str) -> None:
    def loop(data):
        if data['ground'] == 0:
            return

        input_text = data['input_text']
        source = data['source']
        target = data['target']

        prompt = pattern_prompt_train_pos(input_text, source, target)

        llm_resp, token_cost = llm.response(prompt)

        data[resp_key] = llm_resp

        try:
            json_pattern = r'\{[^\{\}]*\}'
            parsed_resp = json.loads(re.findall(json_pattern, llm_resp)[-1])
            pattern = parsed_resp['pattern']
            data[res_key] = pattern
        except:
            data[res_key] = 'No'

    dataset = path_to_dataset(dataset_path)

    llm = get_llm(llm_name)

    resp_key = f'{llm_name}_structured_pattern_resp'
    res_key = f'{llm_name}_structured_pattern'

    with tqdm_joblib(total=len(dataset), desc='processing add_pattern_pos'):
        Parallel(n_jobs=10, backend='threading')(delayed(loop)(data) for data in dataset)

    with open(dataset_path, mode='w', encoding='utf-8') as f:
        f.writelines(json.dumps(data, ensure_ascii=False) + '\n' for data in dataset)


def add_pattern_inference(dataset_path: str, llm_name: str) -> None:
    def loop(data):
        input_text = data['input_text']
        source = data['source']
        target = data['target']

        pattern_prompt = pattern_prompt_inference(input_text, source, target)

        llm_resp, token_cost = llm.response(pattern_prompt)

        data[resp_key] = llm_resp

        try:
            json_pattern = r'\{[^\{\}]*\}'
            parsed_resp = json.loads(re.findall(json_pattern, llm_resp)[-1])
            pattern = parsed_resp['pattern']
            data[res_key] = pattern
        except:
            data[res_key] = 'No'

    dataset = path_to_dataset(dataset_path)

    llm = get_llm(llm_name)
    resp_key = f'{llm_name}_structured_pattern_inference_resp'
    res_key = f'{llm_name}_structured_pattern_inference'

    with tqdm_joblib(total=len(dataset), desc='processing add_pattern_inference'):
        Parallel(n_jobs=10, backend='threading')(delayed(loop)(data) for data in dataset)

    with open(dataset_path, mode='w', encoding='utf-8') as f:
        f.writelines(json.dumps(data, ensure_ascii=False) + '\n' for data in dataset)


def add_conceptnet_node(dataset_path: str) -> None:
    if os.path.exists(CONCEPTNET_NODES_SAVE_PATH):
        conceptnet_vec_db = ConceptnetVecDB.load(CONCEPTNET_NODES_SAVE_PATH)
    else:
        conceptnet_vec_db = ConceptnetVecDB(EMBEDDING_MODEL, NODES_PATH)
        conceptnet_vec_db.init_nodes()
        conceptnet_vec_db.init_nodes_vec()
        conceptnet_vec_db.save(CONCEPTNET_NODES_SAVE_PATH)

    dataset = path_to_dataset(dataset_path)

    for data in tqdm(dataset, desc='processing add_conceptnet_node'):
        source = data['source']
        target = data['target']

        source_node, source_node_score = conceptnet_vec_db.get_node(source)
        target_node, target_node_score = conceptnet_vec_db.get_node(target)

        data['source_node'] = [source_node, source_node_score]
        data['target_node'] = [target_node, target_node_score]

    with open(dataset_path, mode='w', encoding='utf-8') as f:
        f.writelines(json.dumps(data, ensure_ascii=False) + '\n' for data in dataset)


def add_concept_path(dataset_path: str, max_path_len: int = 10,  node_threshold: float = 0.6) -> None:
    def parse_path(path_list: list[dict[str, str] | str], rel_list: list[dict[str, str]]) -> str:
        nodes_list = [item['id'] for item in path_list if isinstance(item, dict)]
        rel_map = {f'{item["start"]}@to@{item["end"]}': item for item in rel_list}

        path_str = f'"{nodes_list[0]}"'
        for idx in range(1, len(nodes_list)):
            direction = f'{nodes_list[idx - 1]}@to@{nodes_list[idx]}'
            if direction in rel_map:
                rel = rel_map[direction]['type']
                path_str += f' {RELATION_MAP[rel][0]} "{nodes_list[idx]}", and "{nodes_list[idx]}"'
            else:
                rel = rel_map[f'{nodes_list[idx]}@to@{nodes_list[idx - 1]}']['type']
                path_str += f' {RELATION_MAP[rel][1]} "{nodes_list[idx]}", and "{nodes_list[idx]}"'

        suffix_idx = path_str.rfind(', and')
        path_str = path_str[:suffix_idx] + '.'
        return path_str

    def loop(data: dict[str, Any]) -> None:
        source_node, source_score = data['source_node']
        target_node, target_score = data['target_node']
        
        if source_score < node_threshold or target_score < node_threshold:
            data['conceptnet_path'] = ''
            return

        source_node = re.sub('"', "'", source_node)
        target_node = re.sub('"', "'", target_node)

        query = f"""
        MATCH p = shortestPath((a)-[*0..{max_path_len}]-(b))
        WHERE a.id = $start_id AND b.id = $end_id
        RETURN p, [rel IN relationships(p) | {{
            type: type(rel),
            start: startNode(rel).id,
            end: endNode(rel).id
        }}] AS relationshipDetails
        """
        with conceptnet_driver.session() as session:
            result = session.run(query, start_id=source_node, end_id=target_node).data()

        if len(result) > 0:
            path = parse_path(result[0]['p'], result[0]['relationshipDetails'])
        else:
            path = ''

        data['conceptnet_path'] = path

    conceptnet_driver = GraphDatabase.driver(CONCEPTNET_URI, auth=(CONCEPTNET_USERNAME, CONCEPTNET_PASSWORD))

    dataset = path_to_dataset(dataset_path)

    with tqdm_joblib(total=len(dataset), desc='processing add_conceptnet_path'):
        Parallel(n_jobs=-1, backend='threading')(delayed(loop)(data) for data in dataset)

    with open(dataset_path, mode='w', encoding='utf-8') as f:
        f.writelines(json.dumps(data, ensure_ascii=False) + '\n' for data in dataset)


def cache_syntactic_tree_edit_distance_similarity(test_data: dict[str, Any], train_data: dict[str, Any], test_dataset_map: dict[int, dict[str, Any]], train_dataset_map: dict[int, dict[str, Any]]) -> float:
    text_test = test_data['input_text_no_tag']
    text_train = train_data['input_text_no_tag']
    tree_test = test_dataset_map[test_data['unique_id']]['dependency_tree']
    tree_train = train_dataset_map[train_data['unique_id']]['dependency_tree']

    if f"{text_test} [SEP] {text_train}" in SYNTACTIC_TREE_DISTANCE_SIMILARITY_CACHE:
        return SYNTACTIC_TREE_DISTANCE_SIMILARITY_CACHE[f"{text_test} [SEP] {text_train}"]
    elif f"{text_train} [SEP] {text_test}" in SYNTACTIC_TREE_DISTANCE_SIMILARITY_CACHE:
        return SYNTACTIC_TREE_DISTANCE_SIMILARITY_CACHE[f"{text_train} [SEP] {text_test}"]
    
    if text_test == text_train:
        similarity = 1
    else:
        similarity = tree_edit_distance_similarity(tree1=tree_test, tree2=tree_train)
    
    SYNTACTIC_TREE_DISTANCE_SIMILARITY_CACHE[f"{text_test} [SEP] {text_train}"] = similarity
    SYNTACTIC_TREE_DISTANCE_SIMILARITY_CACHE[f"{text_train} [SEP] {text_test}"] = similarity
    return similarity
    

def cache_conceptnet_path_edit_distance_similarity(test_data: dict[str, Any], train_data: dict[str, Any]) -> float:
    path_test = test_data['conceptnet_path']
    path_train = train_data['conceptnet_path']

    if f"{path_test} [SEP] {path_train}" in CONCEPTNET_PATH_DISTANCE_SIMILARITY_CACHE:
        return CONCEPTNET_PATH_DISTANCE_SIMILARITY_CACHE[f"{path_test} [SEP] {path_train}"]
    elif f"{path_train} [SEP] {path_test}" in CONCEPTNET_PATH_DISTANCE_SIMILARITY_CACHE:
        return CONCEPTNET_PATH_DISTANCE_SIMILARITY_CACHE[f"{path_train} [SEP] {path_test}"]
    
    if path_test == '' or path_train == '':
        similarity = 0
    elif path_test == path_train:
        similarity = 1
    else:
        similarity = text_edit_distance_similarity(path_test, path_train)

    CONCEPTNET_PATH_DISTANCE_SIMILARITY_CACHE[f"{path_test} [SEP] {path_train}"] = similarity
    CONCEPTNET_PATH_DISTANCE_SIMILARITY_CACHE[f"{path_train} [SEP] {path_test}"] = similarity
    return similarity
    
    
def add_filtered_weighted_unified_examples(full_test_path: str, full_train_path: str, conceptnetpath_weight: float, syntactic_weight: float, k: int, llm_name: str) -> None:
    def get_conceptnet_path_score(data: dict[str, Any]) -> list[tuple[int, float]]:
        scores = Parallel(n_jobs=-1, backend='threading')(delayed(cache_conceptnet_path_edit_distance_similarity)(data, corpus_data) for corpus_data in train_dataset)

        scores = list(zip(scores, train_dataset, [data['input_text_no_tag'] for data in train_dataset]))

        scores = [(item[0], item[1]) for item in scores]  # item[0]: dist; item[1]: corpus_data
        data_score = list()
        for item in scores:
            data_score.append((item[1]['unique_id'], item[0]))
        return data_score

    def get_syntactic_score(data: dict[str, Any]) -> list[tuple[int, float]]:
        scores = Parallel(n_jobs=-1, backend='threading')(delayed(cache_syntactic_tree_edit_distance_similarity)(data, corpus_data, test_dataset_map, train_dataset_map) for corpus_data in train_dataset)

        scores = list(zip(scores, train_dataset, [data['input_text_no_tag'] for data in train_dataset]))

        scores = [(item[0], item[1]) for item in scores]  # item[0]: dist; item[1]: corpus_data
        data_score = list()
        for item in scores:
            data_score.append((item[1]['unique_id'], item[0]))
        return data_score

    def get_pattern_examples(data: dict[str, Any]) -> list[tuple[dict[str, Any], float]]:
        def loop(corpus_data_2_score: tuple[dict[str, Any], float]) -> None | tuple[dict[str, Any], float]:
            if corpus_data_2_score[0]['ground'] == 0:
                return None
            if corpus_data_2_score[0][pattern_label_key].strip().lower() in ('no', 'none'):
                return None
            if corpus_data_2_score[0][pattern_label_key].strip().lower() == llm_pattern.strip().lower():
                return corpus_data_2_score

            return None

        llm_pattern = data[pattern_inference_key]
        if llm_pattern.strip().lower() in ('no', 'none'):
            return []

        # score_weighted_unified: list[tuple[train_data, float(unified_score)]]
        examples = Parallel(n_jobs=-1, backend='threading')(delayed(loop)(corpus_data_2_score) for corpus_data_2_score in score_weighted_unified)
        # examples: list[None | tuple[train_data, float(unified_score)]]

        examples = [item for item in examples if item]  # examples: list[tuple[train_data, float(unified_score)]]
        examples = random_deduplicate_item_score_pair(examples)  # examples: list[tuple[train_data, float(unified_score)]]

        return examples


    test_dataset = path_to_dataset(full_test_path)
    test_dataset =random.sample(test_dataset, min(50, len(test_dataset)))
    test_dataset_map = {data['unique_id']: {**data, 'dependency_tree': DependencyTree(data['input_text_no_tag'])} for data in tqdm(test_dataset, desc='processing add_filtered_weighted_unified_examples | building tree | over test_dataset')}

    train_dataset = path_to_dataset(full_train_path)
    train_dataset_map = {data['unique_id']: {**data, 'dependency_tree': DependencyTree(data['input_text_no_tag'])} for data in tqdm(train_dataset, desc='processing add_filtered_weighted_unified_examples | building tree | over train_dataset')}

    for test_data in tqdm(test_dataset, desc='processing add_filtered_weighted_unified_examples | main process', total=len(test_dataset)):
        # step 1
        data_2_conceptnetpath_score = get_conceptnet_path_score(test_data)  # [(train_data_unique_id, score), (train_data_unique_id, score), ...]

        # step 2
        data_2_syntactic_score = get_syntactic_score(test_data)  # [(train_data_unique_id, score), (train_data_unique_id, score), ...]
        data_2_syntactic_map = {item[0]: item[1] for item in data_2_syntactic_score}

        # step 3
        score_weighted_unified = list()  # list[tuple[train_data, float(unified_score)]]
        for uid, cnp_score in data_2_conceptnetpath_score:
            syn_score = data_2_syntactic_map[uid]
            unified_score = conceptnetpath_weight * cnp_score + syntactic_weight * syn_score
            score_weighted_unified.append((train_dataset_map[uid], unified_score))

        pattern_inference_key = f'{llm_name}_structured_pattern_inference'
        pattern_label_key = f'{llm_name}_structured_pattern'
        res_key = f'filtered_weighted_unified_examples_{llm_name}_structured_pattern'

        pattern_examples = get_pattern_examples(test_data)  # examples: list[tuple[train_data, float(unified_score)]]

        # 排序
        pattern_examples = sorted(pattern_examples, key=lambda item: item[1], reverse=True)
        if len(pattern_examples) > k:
            pattern_examples = pattern_examples[:k]
        elif len(pattern_examples) < k:
            this_examples_ids = {item[0]['unique_id'] for item in pattern_examples}
            
            complement_examples = [item for item in score_weighted_unified if item[0]['unique_id'] not in this_examples_ids]
            complement_examples = sorted(complement_examples, key=lambda item: item[1], reverse=True)
            
            pattern_examples = pattern_examples + complement_examples[:k - len(pattern_examples)]
            pattern_examples = sorted(pattern_examples, key=lambda item: item[1], reverse=True)

        pattern_examples_ids = [item[0]['unique_id'] for item in pattern_examples]

        test_data[res_key] = pattern_examples_ids

    with open(full_test_path, 'w', encoding='utf-8') as f:
        f.writelines(json.dumps(item, ensure_ascii=False) + '\n' for item in test_dataset)


if __name__ == '__main__':
    def main(dataset_name: str) -> None:
        FULL_TEST_PATH = r'dataset\{dataset_name}\full_test.jsonl'.format(dataset_name=dataset_name)
        FULL_TRAIN_PATH = r'dataset\{dataset_name}\full_train.jsonl'.format(dataset_name=dataset_name)

        LLM_NAME = os.environ.get('GPT_4O_MINI_NAME')

        CONCEPTNET_PATH_MAX = 30
        
        NODE_THRESHOLD = 0.6

        CONCEPTNET_PATH_WEIGHT = 0.5
        SYNTACTICS_WEIGHT = 0.5
        K = 10

        add_pattern_pos(FULL_TRAIN_PATH, LLM_NAME)
        add_pattern_inference(FULL_TEST_PATH, LLM_NAME)
        
        add_conceptnet_node(FULL_TEST_PATH)
        add_conceptnet_node(FULL_TRAIN_PATH)
        add_concept_path(FULL_TEST_PATH, CONCEPTNET_PATH_MAX, NODE_THRESHOLD)
        add_concept_path(FULL_TRAIN_PATH, CONCEPTNET_PATH_MAX, NODE_THRESHOLD)

        add_filtered_weighted_unified_examples(FULL_TEST_PATH, FULL_TRAIN_PATH, CONCEPTNET_PATH_WEIGHT, SYNTACTICS_WEIGHT, K, LLM_NAME)


    main('CTB')
    # main('ESC - inter')
    # main('ESC - intra')
