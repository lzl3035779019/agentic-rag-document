from src import llm_cache


class _Chunk:
    def __init__(self, content: str):
        self.content = content


class _StreamingLLM:
    def stream(self, prompt):
        yield _Chunk("hello")
        yield _Chunk(" ")
        yield _Chunk("world")


def test_streaming_cache_hit_yields_cached_text(monkeypatch):
    monkeypatch.setattr(llm_cache, "get_cached_response", lambda task_name, prompt_text: "cached answer")

    chunks = list(llm_cache.invoke_llm_cached_stream("task", "prompt"))

    assert "".join(chunks) == "cached answer"


def test_streaming_cache_miss_streams_and_saves(monkeypatch):
    saved = {}

    monkeypatch.setattr(llm_cache, "get_cached_response", lambda task_name, prompt_text: None)
    monkeypatch.setattr(llm_cache, "get_llm", lambda: _StreamingLLM())
    monkeypatch.setattr(
        llm_cache,
        "save_cached_response",
        lambda task_name, prompt_text, response: saved.update(
            {"task_name": task_name, "prompt_text": prompt_text, "response": response}
        ),
    )

    chunks = list(llm_cache.invoke_llm_cached_stream("fast_answer", "prompt text"))

    assert chunks == ["hello", " ", "world"]
    assert saved == {
        "task_name": "fast_answer",
        "prompt_text": "prompt text",
        "response": "hello world",
    }
