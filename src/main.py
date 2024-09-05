import asyncio
from typing import List, Type

from mlp_sdk.abstract import Task
from mlp_sdk.hosting.host import host_mlp_cloud, host
from mlp_sdk.transport.MlpClientSDK import MlpClientSDK
from mlp_sdk.transport.MlpServiceSDK import MlpServiceSDK
from mlp_sdk.types import Span, NamedEntities, TextsCollection, NamedEntitiesCollection
from pydantic import BaseModel, validator
import os
from dotenv import load_dotenv
from openai import AsyncOpenAI

from src.prompt import prompt, user_text

load_dotenv()

# class PredictRequest(TextsCollection):
#     @validator('texts')
#     @classmethod
#     def texts_are_not_too_long(cls, texts: List[str]) -> List[str]:
#         for t in texts:
#             if len(t) > 1000:
#                 raise ValueError(f"Text should not be longer than 1000 characters."
#                                  f"Long text: {t}")
#         return texts


SOURCE_TYPE = "LLM_PER_EXTRACTOR"  # TODO pass it

request_frame = '{"messages":[{"role":"user", "content":"%s"}]}'

api_key = os.getenv("MLP_CLIENT_TOKEN")
assert api_key is not None
openai = AsyncOpenAI(
    api_key=api_key,  # TODO maybe smth else
    base_url="https://caila.io/api/adapters/openai"
)


class NERLLM(Task):
    def __init__(self, config: BaseModel = None, service_sdk: MlpServiceSDK = None) -> None:
        super().__init__(config, service_sdk)

    def init_config_schema(self) -> Type[BaseModel]:
        return BaseModel

    async def predict(self, data: TextsCollection, config: BaseModel) -> NamedEntitiesCollection:
        prompt_w_text = prompt % user_text  # data.texts[0] # TODO all texts
        res = await openai.chat.completions.create(
            messages=[{"role": "user", "content": "hello"}],
            model="just-ai/openai-proxy/gpt-4o-mini"
        )
        content = res.choices[0].message.content
        print(f"<<<< {content}")
        return NamedEntitiesCollection(entities_list=[])

    @property
    def predict_config_schema(self) -> Type[BaseModel]:
        return BaseModel


if __name__ == "__main__":
    host(NERLLM, params=None, port=8082)
    # host_mlp_cloud(NERLLM, BaseModel())
