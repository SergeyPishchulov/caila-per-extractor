import asyncio
from typing import List

from mlp_sdk.abstract import Task
from mlp_sdk.hosting.host import host_mlp_cloud
from mlp_sdk.transport.MlpClientSDK import MlpClientSDK
from mlp_sdk.transport.MlpServiceSDK import MlpServiceSDK
from mlp_sdk.types import Span, NamedEntities, TextsCollection
from pydantic import BaseModel, validator
import os
from dotenv import load_dotenv
from openai import AsyncOpenAI

from src.prompt import prompt, user_text

load_dotenv()


class PredictRequest(TextsCollection):
    @validator('texts')
    @classmethod
    def texts_are_not_too_long(cls, texts: List[str]) -> List[str]:
        if sum([len(t) for t in texts]) > 1000:
            raise ValueError("All texts should not be longer than 1000 characters")
        return texts


SOURCE_TYPE = "LLM_PER_EXTRACTOR"


class PredictResponse(BaseModel):
    entities_list: List[NamedEntities]


request_frame = '{"messages":[{"role":"user", "content":"%s"}]}'

openai = AsyncOpenAI(
    api_key=os.getenv("MLP_CLIENT_TOKEN"),  # TODO maybe smth else
    base_url="https://caila.io/api/adapters/openai"
)


class SimpleActionExample(Task):
    def __init__(self, config: BaseModel, service_sdk: MlpServiceSDK = None) -> None:
        super().__init__(config, service_sdk)

    async def predict(self, data: PredictRequest, config: BaseModel) -> PredictResponse:
        prompt_w_text = prompt % user_text  # data.texts[0] # TODO all texts
        res = await openai.chat.completions.create(
            messages=[{"role": "user", "content": prompt_w_text}],
            model="just-ai/openai-proxy/yandexgpt"
        )
        content = res.choices[0].message.content
        print(f"<<<< {content}")
        return PredictResponse(entities_list=[])


if __name__ == "__main__":
    asyncio.run()
    # host_mlp_cloud(SimpleActionExample, BaseModel())

