import pytest

from src.aprox_matcher import AproxMatcher

def get_found_names(found_indices, text):
    return [text[start:end] for (start, end) in found_indices]

@pytest.mark.parametrize("text",
                         ["Петра",
                          "дочь Петра",
                          "дочь Петра.",
                          "Петра дочь",
                          "Петра Столыпина",
                          ])
def test_1_word_name(text):
    string = "Петр"
    found_indices = AproxMatcher().find(string, text=text)
    found_names = get_found_names(found_indices, text)
    assert "Петра" in found_names


@pytest.mark.parametrize("text",
                         ["Петра Столыпина",
                          "дочь Петра Столыпина",
                          "дочь Петра Столыпина.",
                          "Петра Столыпина дочь",
                          ])
def test_2_words(text):
    string = "Петр Столыпин"
    found_indices = AproxMatcher().find(string, text=text)
    found_names = get_found_names(found_indices, text)
    assert "Петра Столыпина" in found_names
