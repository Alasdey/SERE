from tqdm import tqdm

import joblib
from transformers import AutoTokenizer, AutoModel
import torch
from torch.utils.data import DataLoader


class ConceptnetVecDB:
    def __init__(self, embedding_model: str, nodes_path: str) -> None:
        self.__tokenizer = AutoTokenizer.from_pretrained(embedding_model)
        self.__model = AutoModel.from_pretrained(embedding_model).to(torch.device('cuda'))
        self.__batch_size = 32

        self.__nodes = None
        self.__nodes_vec = None
        
        self.__nodes_path = nodes_path

    def init_nodes(self) -> None:
        self.__nodes = list()
        with open(self.__nodes_path, 'r', encoding='utf-8') as f:
            for idx, line in enumerate(f):
                if idx == 0:
                    continue

                self.__nodes.append(line.split(',')[0])

    def __embed(self, input_list: list[str]):
        def mean_pooling(token_embeddings, mask):
            token_embeddings = token_embeddings.masked_fill(~mask[..., None].bool(), 0.)
            sentence_embeddings = token_embeddings.sum(dim=1) / mask.sum(dim=1)[..., None]
            return sentence_embeddings

        all_embeddings = list()
        dataloader = DataLoader(input_list, batch_size=self.__batch_size, shuffle=False)

        if len(input_list) == 1:
            for batch in dataloader:
                inputs = self.__tokenizer(batch, padding=True, return_tensors='pt')
                inputs = {key: value.to(torch.device('cuda')) for key, value in inputs.items()}

                with torch.no_grad():
                    outputs = self.__model(**inputs)

                embedding = mean_pooling(outputs[0], inputs['attention_mask'])  # type(embedding): torch.Tensor  shape: [len(input_list), dim]
                all_embeddings.append(embedding)
        else:
            for batch in tqdm(dataloader, total=len(dataloader), desc='generating node embeddings'):
                inputs = self.__tokenizer(batch, padding=True, return_tensors='pt')
                inputs = {key: value.to(torch.device('cuda')) for key, value in inputs.items()}

                with torch.no_grad():
                    outputs = self.__model(**inputs)

                embedding = mean_pooling(outputs[0], inputs['attention_mask'])  # type(embedding): torch.Tensor  shape: [len(input_list), dim]
                all_embeddings.append(embedding)

        return torch.cat(all_embeddings, dim=0)

    def init_nodes_vec(self) -> None:
        self.__nodes_vec = self.__embed(self.__nodes)

    def get_node(self, event: str) -> tuple[str, float]:
        event_embedding = self.__embed([event])  # shape: [1, dim]
        event_embedding = event_embedding[0:]  # shape: [1, dim]

        cosine_scores = (self.__nodes_vec * event_embedding).sum(1) / (self.__nodes_vec.norm(dim=-1) * event_embedding.norm())  # [len(nodes), dim] * [1, dim]

        max_index = torch.argmax(cosine_scores)
        max_similarity = cosine_scores[max_index].item()

        best_node = self.__nodes[max_index]

        return best_node, max_similarity

    def save(self, save_path: str) -> None:
        joblib.dump(self, save_path)
        print(f"Instance saved to {save_path}")

    @staticmethod
    def load(save_path: str) -> 'ConceptnetVecDB':
        instance = joblib.load(save_path)
        print(f"Instance loaded from {save_path}")
        return instance


if __name__ == '__main__':
    EMBEDDING_MODEL = r'YOUR_EMBEDDING_MODEL_PATH'
    NODES_PATH = r'nodes.csv PATH FROM utils/conceptnet/to_neo4j.py'

    SAVE_PATH = r'.joblib FILE OUTPUT PATH'

    db = ConceptnetVecDB(EMBEDDING_MODEL, NODES_PATH)
    db.save(SAVE_PATH)
