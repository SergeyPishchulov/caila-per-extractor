from src.ner import NerLlm


def test_1():
    ner_llm = NerLlm()
    raw = "Пётр Столыпин, Михаил Васильевич Арсеньев, Мария, М. Ю. Лермонтов."
    text = "Одна из пяти сестёр деда Петра Столыпина была женой Михаила Васильевича Арсеньева. Их дочь Мария была матерью М. Ю. Лермонтова."
    print(ner_llm.parse_llm_raw_result(raw, text))