from src.ner import NerLlm


def test_find_if_name_in_declension():
    ner_llm = NerLlm()
    raw_llm_results = "Пётр Столыпин, Михаил Васильевич Арсеньев, Мария, М. Ю. Лермонтов."
    text = "Одна из пяти сестёр деда Петра Столыпина была женой Михаила Васильевича Арсеньева. Их дочь Мария была матерью М. Ю. Лермонтова."
    entities = [
        e.entity for e in
        ner_llm.parse_llm_raw_result(raw_llm_results, text).entities]
    assert "Петра Столыпина" in entities
    assert "Михаила Васильевича Арсеньева" in entities
    assert "Мария" in entities
    assert "М. Ю. Лермонтова" in entities

