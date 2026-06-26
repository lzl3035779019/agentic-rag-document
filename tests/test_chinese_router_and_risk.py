from src.graph import _risk_level
from src.router import rule_route


def test_chinese_comparison_question_routes_complex_by_rule():
    result = rule_route("\u65b9\u6848 A \u548c\u65b9\u6848 B \u6709\u4ec0\u4e48\u533a\u522b\uff1f")

    assert result["route"] == "complex"
    assert result["router"] == "rule"


def test_chinese_single_topic_question_routes_simple_by_rule():
    result = rule_route("\u8fd9\u4e2a\u7cfb\u7edf\u7684\u6838\u5fc3\u6d41\u7a0b\u6709\u54ea\u4e9b\uff1f")

    assert result["route"] == "simple"
    assert result["router"] == "rule"


def test_chinese_broad_scenario_question_routes_complex_by_rule():
    result = rule_route("\u5f00\u59cb\u4e00\u4e2a\u65b0\u9879\u76ee\u524d\u5e94\u8be5\u4e86\u89e3\u4ec0\u4e48\uff1f")

    assert result["route"] == "complex"
    assert result["router"] == "rule"


def test_chinese_destructive_or_permission_question_is_high():
    assert _risk_level("\u6267\u884c\u5220\u9664\u64cd\u4f5c\u524d\u9700\u8981\u6ce8\u610f\u4ec0\u4e48\uff1f") == "high"
    assert _risk_level("\u8bbf\u95ee\u6743\u9650\u5e94\u8be5\u600e\u4e48\u63a7\u5236\uff1f") == "high"


def test_chinese_conditions_and_boundaries_question_is_medium():
    assert _risk_level("\u8fd9\u4e2a\u6d41\u7a0b\u7684\u9002\u7528\u6761\u4ef6\u548c\u8fb9\u754c\u662f\u4ec0\u4e48\uff1f") == "medium"


def test_chinese_general_question_is_low():
    assert _risk_level("Agent \u7684\u5de5\u4f5c\u6a21\u5f0f\u6709\u54ea\u4e9b\uff1f") == "low"
