import spacy
import zss
import Levenshtein

import math


DEPENDENCY_WEIGHTS = {
    "acl": 2,           # 名词性从句修饰语（结构性，影响名词性结构）
    "acomp": 1,         # 形容词性补语（补充描述词性，不影响核心结构）
    "advcl": 3,         # 副词性从句修饰语（增强句子整体修饰关系）
    "advmod": 2,        # 副词修饰语（修饰动词，适中权重）
    "agent": 5,         # 行为者（关键性，影响主谓结构）
    "amod": 0,          # 形容词修饰语（描述性，修饰作用较小）
    "appos": 3,         # 同位语修饰语（增强句子表达，影响结构）
    "attr": 1,          # 属性修饰语（补充说明，影响描述）
    "aux": 0,           # 助动词（功能性，不影响核心结构）
    "auxpass": 0,       # 被动语态助动词（功能性，不影响句子核心结构）
    "case": 0,          # 格标记（功能性，不影响句子核心结构）
    "cc": 2,            # 并列连词（连接句子结构，适中权重）
    "ccomp": 3,         # 从句补语（结构性，从句补语有一定影响）
    "compound": 3,      # 复合修饰词（增强句子结构的表达，适中权重）
    "conj": 2,          # 并列词（适中权重，用于并列结构）
    "csubj": 5,         # 从句主语（结构性，影响句子核心语法）
    "csubjpass": 5,     # 被动从句主语（结构性，从句中的主语角色）
    "dative": 0,        # 与格（功能性，不影响核心结构）
    "dep": 0,           # 未分类的依赖项（影响较小，结构性不明确）
    "det": 1,           # 限定词（修饰名词，但不会改变核心结构）
    "dobj": 4,          # 直接宾语（句子核心成分之一，影响结构）
    "expl": 0,          # 表现词（影响较小，结构辅助）
    "intj": 0,          # 感叹词（对结构影响不大）
    "mark": 0,          # 语法标记（如“that”等引导词，影响句子结构关系）
    "meta": 0,          # 元修饰符（主要用于附加信息，影响较小）
    "neg": 2,           # 否定修饰符（否定句意，对结构有一定影响）
    "nounmod": 2,       # 名词修饰词（修饰名词，影响句子结构）
    "npmod": 2,         # 名词短语作为副词修饰语（有助于语法结构）
    "nsubj": 5,         # 名词主语（句子主要结构之一，核心成分）
    "nsubjpass": 5,     # 被动名词主语（核心成分，影响结构）
    "nummod": 1,        # 数词修饰符（修饰数词，结构影响较小）
    "oprd": 3,          # 宾语谓词（影响语法结构，属于核心部分）
    "parataxis": 2,     # 并列句（影响结构连接，适度权重）
    "pcomp": 3,         # 介词短语补语（影响结构，但不是主句核心）
    "pobj": 4,          # 介词宾语（重要的句子结构成分）
    "poss": 1,          # 所有格修饰符（影响较小，属于修饰）
    "preconj": 1,       # 前置关联连词（连接句子，影响结构）
    "predet": 1,        # 前置限定词（修饰名词的关系，结构影响较小）
    "prep": 3,          # 介词修饰（句子中结构性连接，影响较大）
    "prt": 1,           # 小品词（影响结构较小）
    "punct": 0,         # 标点符号（对结构影响极小）
    "quantmod": 1,      # 数量词修饰（修饰量词，结构影响较小）
    "relcl": 3,         # 关系从句修饰（对句子结构有较大影响）
    "ROOT": 5,          # 句子根节点（核心结构，最重要）
    "xcomp": 3,         # 开放性从句补语（影响句子结构）
}

nlp = spacy.load('en_core_web_sm')


class DependencyNode:
    def __init__(self, token=None, label=None) -> None:
        self.__token = token
        
        if token is not None:
            self.label = f"{self.__token.dep_}"
        else:
            self.label = label
        self.children = list()

    def add_child(self, node) -> None:
        self.children.append(node)


class DependencyTree:
    def __init__(self, text: str) -> None:
        self.__text = text
        self.root = None

        self.__build_tree()

    def __build_tree(self) -> None:
        def build(token) -> DependencyNode:
            node = DependencyNode(token)

            for child in token.children:
                node.add_child(build(child))
            return node

        doc = nlp(self.__text)
        sents = list(doc.sents)
        
        if len(sents) == 1:
            self.root = build([sent.root for sent in sents][0])
        else:
            self.root = DependencyNode(label='hyper_root')
            for sent in sents:
                self.root.add_child(build(sent.root))


def tree_edit_distance_similarity(text1: str = None, text2: str = None, tree1: DependencyTree = None, tree2: DependencyTree = None) -> float:
    def get_children(node: DependencyNode) -> list[DependencyNode]:
        return node.children

    def get_label(node: DependencyNode) -> str:
        return node.label

    def label_dist(label1, label2):
        """计算节点之间的距离。考虑距离不同依存关系权重。"""

        if label1 == label2:
            return 0
        else:
            if label1 == '':
                return DEPENDENCY_WEIGHTS.get(label2, 0)
            elif label2 == '':
                return DEPENDENCY_WEIGHTS.get(label1, 0)
            else:
                return DEPENDENCY_WEIGHTS.get(label1, 0) + DEPENDENCY_WEIGHTS.get(label2, 0)

    assert ((text1 is not None) and (text2 is not None)) ^ ((tree1 is not None) and (tree2 is not None))

    if tree1 is None and tree2 is None:
        tree1 = DependencyTree(text1)
        tree2 = DependencyTree(text2)

    distance = zss.distance(tree1.root, tree2.root, get_children=get_children,
                            insert_cost=lambda node: label_dist('', get_label(node)),
                            remove_cost=lambda node: label_dist('', get_label(node)),
                            update_cost=lambda a, b: label_dist(get_label(a), get_label(b)))

    similarity = math.exp(-0.05 * distance)
    return similarity


def text_edit_distance_similarity(text1: str, text2: str) -> float:
    edit_distance = Levenshtein.distance(text1, text2)

    max_len = max(len(text1), len(text2))
    similarity = 1 - edit_distance / max_len if max_len > 0 else 0
    return similarity
