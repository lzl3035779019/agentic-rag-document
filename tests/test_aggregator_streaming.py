from src.agents.aggregator import aggregate_answers_stream


def test_aggregate_answers_stream_uses_aggregate_answers_task(monkeypatch):
    calls = {}

    def fake_stream(task_name, prompt):
        calls["task_name"] = task_name
        calls["prompt"] = prompt
        return iter(["final"])

    monkeypatch.setattr("src.agents.aggregator.invoke_llm_cached_stream", fake_stream)

    chunks = list(
        aggregate_answers_stream(
            "Compare them.",
            [{"sub_question": "What is A?", "answer": "A is supported."}],
        )
    )

    assert chunks == ["final"]
    assert calls["task_name"] == "aggregate_answers"
    assert calls["prompt"] is not None


def test_aggregate_answers_passes_prompt_to_cached_llm(monkeypatch):
    calls = {}

    def fake_invoke(task_name, prompt):
        calls["task_name"] = task_name
        calls["prompt"] = prompt
        return "final"

    monkeypatch.setattr("src.agents.aggregator.invoke_llm_cached", fake_invoke)

    result = aggregate_answers_stream.__globals__["aggregate_answers"](
        "Compare them.",
        [{"sub_question": "What is A?", "answer": "A is supported."}],
    )

    assert result == "final"
    assert calls["task_name"] == "aggregate_answers"
    assert calls["prompt"] is not None
