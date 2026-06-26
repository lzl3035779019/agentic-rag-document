from src.kb_routing import infer_query_language, resolve_query_kb_id


def test_infer_query_language_detects_chinese_questions():
    assert infer_query_language("员工福利有哪些？") == "zh"


def test_infer_query_language_defaults_english_without_chinese():
    assert infer_query_language("What benefits does the company provide?") == "en"


def test_resolve_query_kb_uses_manual_selection_when_auto_disabled():
    assert (
        resolve_query_kb_id(
            question="员工福利有哪些？",
            manual_kb_id="manual",
            auto_route_enabled=False,
            chinese_kb_id="zh-kb",
            english_kb_id="en-kb",
        )
        == "manual"
    )


def test_resolve_query_kb_routes_chinese_and_english_questions():
    assert (
        resolve_query_kb_id(
            question="员工福利有哪些？",
            manual_kb_id="manual",
            auto_route_enabled=True,
            chinese_kb_id="zh-kb",
            english_kb_id="en-kb",
        )
        == "zh-kb"
    )
    assert (
        resolve_query_kb_id(
            question="What benefits are provided?",
            manual_kb_id="manual",
            auto_route_enabled=True,
            chinese_kb_id="zh-kb",
            english_kb_id="en-kb",
        )
        == "en-kb"
    )


def test_resolve_query_kb_falls_back_to_manual_when_language_target_missing():
    assert (
        resolve_query_kb_id(
            question="员工福利有哪些？",
            manual_kb_id="manual",
            auto_route_enabled=True,
            chinese_kb_id=None,
            english_kb_id="en-kb",
        )
        == "manual"
    )
