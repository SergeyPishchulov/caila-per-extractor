from typing import List

from mlp_sdk.abstract import Task
from mlp_sdk.hosting.host import host_mlp_cloud
from mlp_sdk.transport.MlpServiceSDK import MlpServiceSDK
from pydantic import BaseModel, field_validator


class PredictRequest(BaseModel):
    texts: List[str]

    @field_validator('texts')
    @classmethod
    def texts_are_not_too_long(cls, texts: List[str]) -> List[str]:
        if sum([len(t) for t in texts]) > 1000:
            raise ValueError("All texts should not be longer than 1000 characters")
        return texts


class Span(BaseModel):
    start_index: int
    end_index: int


class Entity(BaseModel):
    value: str
    entity_type: str
    span: Span
    entity: str
    source_type: str = "LLM_PER_EXTRACTOR"


class EntityContainer(BaseModel):
    entities: List[Entity]


class PredictResponse(BaseModel):
    entities_list: List[EntityContainer]


class SimpleActionExample(Task):
    def __init__(self, config: BaseModel, service_sdk: MlpServiceSDK = None) -> None:
        super().__init__(config, service_sdk)

    def predict(self, data: PredictRequest, config: BaseModel) -> PredictResponse:
        return PredictResponse(response="Hello, " + data.name + "!")


if __name__ == "__main__":
    host_mlp_cloud(SimpleActionExample, BaseModel())
