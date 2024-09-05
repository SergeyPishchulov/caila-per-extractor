from mlp_sdk.types import NamedEntities, NamedEntity, Span


class NerLlm:
    def __init__(self):
        pass

    def get_ner_person(self, text: str) -> NamedEntities:
        if not isinstance(text, str):
            raise TypeError(f"text of wrong type: {type(text)}")
        res = NamedEntities(entities=[])
        for i in range(2):
            res.entities.append(NamedEntity(
                entity_type="t",
                value="1",
                span=Span(start_index=1, end_index=2),
                entity="e",
                source_type="1"
            ))
        return res
