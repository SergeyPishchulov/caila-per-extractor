import asyncio
from typing import List, Type

from mlp_sdk.abstract import Task
from mlp_sdk.hosting.host import host_mlp_cloud, host
from mlp_sdk.transport.MlpClientSDK import MlpClientSDK
from mlp_sdk.transport.MlpServiceSDK import MlpServiceSDK
from mlp_sdk.types import Span, NamedEntities, TextsCollection, NamedEntitiesCollection, NamedEntity, \
    InflectorTextsCollection
from pydantic import BaseModel, validator
import os
from dotenv import load_dotenv
from openai import AsyncOpenAI

from src.ner import NerLlm
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

class NERLLM(Task):
    def __init__(self, config: BaseModel = None, service_sdk: MlpServiceSDK = None) -> None:
        super().__init__(config, service_sdk)
        self.nerllm = NerLlm()

    def init_config_schema(self) -> Type[BaseModel]:
        return BaseModel

    async def predict(self, data: TextsCollection, config: BaseModel) -> NamedEntitiesCollection:
        texts_results = []
        for text in data.texts:
            found_in_text = await self.nerllm.get_ner_person(text)
            assert isinstance(found_in_text, NamedEntities), type(found_in_text)
            texts_results.append(found_in_text)

        return NamedEntitiesCollection(entities_list=texts_results)

    async def orig_predict(self, data: TextsCollection, config: BaseModel) -> NamedEntitiesCollection:
        result = TextsCollection(texts=[])
        result.texts.append("Done")
        return NamedEntitiesCollection(
            entities_list=[
                NamedEntities(entities=[NamedEntity(
                    entity_type="t",
                    value="1",
                    span=Span(start_index=1, end_index=2),
                    entity="e",
                    source_type="1"
                )])])

    @property
    def predict_config_schema(self) -> Type[BaseModel]:
        return BaseModel


if __name__ == "__main__":
    host(NERLLM, params=BaseModel(), port=8082)
    # host_mlp_cloud(NERLLM, BaseModel())
