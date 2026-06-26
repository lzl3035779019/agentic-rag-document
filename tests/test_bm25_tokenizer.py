from src.bm25_store import tokenize


def test_tokenize_keeps_english_words():
    assert tokenize("Basecamp benefits include PTO.") == [
        "basecamp",
        "benefits",
        "include",
        "pto",
    ]


def test_tokenize_emits_chinese_character_tokens():
    tokens = tokenize("员工福利包括医疗保险")

    assert "员" in tokens
    assert "工" in tokens
    assert "福" in tokens
    assert "利" in tokens
