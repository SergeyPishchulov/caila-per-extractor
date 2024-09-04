from typing import List

from mlp_sdk.abstract import Task
from mlp_sdk.hosting.host import host_mlp_cloud
from mlp_sdk.transport.MlpClientSDK import MlpClientSDK
from mlp_sdk.transport.MlpServiceSDK import MlpServiceSDK
from mlp_sdk.types import Span, NamedEntities, TextsCollection
from pydantic import BaseModel, field_validator
import os
from dotenv import load_dotenv

load_dotenv()


class PredictRequest(TextsCollection):

    @field_validator('texts')
    @classmethod
    def texts_are_not_too_long(cls, texts: List[str]) -> List[str]:
        if sum([len(t) for t in texts]) > 1000:
            raise ValueError("All texts should not be longer than 1000 characters")
        return texts


SOURCE_TYPE = "LLM_PER_EXTRACTOR"


class PredictResponse(BaseModel):
    entities_list: List[NamedEntities]


class SimpleActionExample(Task):
    def __init__(self, config: BaseModel, service_sdk: MlpServiceSDK = None) -> None:
        super().__init__(config, service_sdk)

    def predict(self, data: PredictRequest, config: BaseModel) -> PredictResponse:
        sdk = MlpClientSDK()
        sdk.predict(account=os.getenv('CAILA_ACCOUNT'),
                    model="yandexgpt",
                    data="")
        return PredictResponse(response="Hello, " + data.name + "!")


if __name__ == "__main__":
    host_mlp_cloud(SimpleActionExample, BaseModel())
