import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from openai import OpenAI, NOT_GIVEN
from dotenv import load_dotenv
import tiktoken


load_dotenv(os.path.join(os.path.dirname(os.path.dirname(__file__)), 'env.env'))


GPT_KEY = os.environ.get('OPENROUTER_API_KEY')
GPT_4O_MINI_NAME = os.environ.get('GPT_4O_MINI_NAME')
GPT_4O_ENCODING_NAME = os.environ.get('GPT_4O_ENCODING_NAME')


def num_tokens_from_string(s: str, encoding_name: str=None, model_name: str=None) -> int:
    assert (encoding_name is not None) ^ (model_name is not None)
    
    if encoding_name is not None:
        encoding = tiktoken.get_encoding(encoding_name)
    else:
        encoding = tiktoken.encoding_for_model(model_name)
        
    num_tokens = len(encoding.encode(s))
    return num_tokens


class GPT4oMini:
    def __init__(self, model_name: str | None = None) -> None:
        self._client = OpenAI(api_key=GPT_KEY, base_url='https://openrouter.ai/api/v1')
        self._model_name = model_name or GPT_4O_MINI_NAME
        self._encoding_name = GPT_4O_ENCODING_NAME

    def response(self, question: str, system_prompt: str = None, in_json: bool = False, **kwargs) -> tuple[str, dict[str, int]]:
        system_prompt = 'You are an assistant.' if system_prompt is None else system_prompt

        response_format = dict()
        if in_json:
            response_format['type'] = 'json_object'
        response_format = response_format if response_format else NOT_GIVEN

        completion = self._client.chat.completions.create(model=self._model_name,
                                                          messages=[
                                                              {"role": "system", "content": system_prompt},
                                                              {"role": "user", "content": question}
                                                          ],
                                                          temperature=0.0,
                                                          response_format=response_format,
                                                          **kwargs)

        resp = completion.choices[0].message.content

        try:
            input_token_cost = num_tokens_from_string(system_prompt + question, self._encoding_name)
            output_token_cost = num_tokens_from_string(resp, self._encoding_name)
        except Exception:
            input_token_cost = 0
            output_token_cost = 0

        return resp, {'input': input_token_cost, 'output': output_token_cost}


def get_llm(name: str) -> object:
    return GPT4oMini(model_name=name)
