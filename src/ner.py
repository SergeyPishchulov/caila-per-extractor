import os
import re
from typing import List

from mlp_sdk.types import NamedEntities, NamedEntity, Span
from openai import AsyncOpenAI

from src import key
from src.prompt import prompt


class NerLlm:
    SOURCE_TYPE = "LLM_PER_EXTRACTOR"

    def __init__(self):
        api_key = key.MLP_CLIENT_TOKEN #os.getenv("MLP_CLIENT_TOKEN")
        assert api_key is not None
        self.openai_client = AsyncOpenAI(
            api_key=api_key,
            base_url="https://caila.io/api/adapters/openai"
        )

    async def get_ner_person(self, text: str) -> NamedEntities:
        if not isinstance(text, str):
            raise TypeError(f"text of wrong type: {type(text)}")

        llm_raw_results = await self.get_llm_raw_result(text)
        return self.parse_llm_raw_result(llm_raw_results, text)

    def parse_llm_raw_result(self, llm_raw_results, text):
        named_entities: List[NamedEntity] = []
        for name in llm_raw_results.split(','):
            name = name.strip()
            starts_ends = [(m.start(), m.end()) for m in re.finditer(name, text)]
            for start, end in starts_ends:
                named_entities.append(NamedEntity(
                    entity_type="PERSON",
                    value=name,
                    span=Span(start_index=start, end_index=end),
                    entity=name,
                    source_type=self.SOURCE_TYPE))

        return NamedEntities(entities=named_entities)

    async def get_llm_raw_result(self, text):
        response = await self.openai_client.chat.completions.create(
            messages=[{"role": "user", "content": self.get_prompt(text)}],
            model="just-ai/openai-proxy/gpt-4o-mini"
        )
        return response.choices[0].message.content


    def get_prompt(self, user_text):
        return prompt % user_text
