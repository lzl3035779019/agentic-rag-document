import os
from contextlib import contextmanager

from dotenv import load_dotenv

load_dotenv()


# def langfuse_enabled() -> bool:
#     return bool(
#         os.getenv("LANGFUSE_PUBLIC_KEY")
#         and os.getenv("LANGFUSE_SECRET_KEY")
#         and os.getenv("LANGFUSE_HOST")
#     )
#
#
# @contextmanager
# def trace_step(name: str, metadata: dict | None = None):
#     print(f"[trace:start] {name}", metadata or {})
#     try:
#         yield
#     finally:
#         print(f"[trace:end] {name}")

from contextlib import contextmanager

from dotenv import load_dotenv
from langfuse import get_client

load_dotenv()

langfuse = get_client()


def check_langfuse():
    if langfuse.auth_check():
        print("Langfuse connected")
    else:
        print("Langfuse auth failed")


@contextmanager
def trace_step(name: str, metadata: dict | None = None):
    with langfuse.start_as_current_observation(
        as_type="span",
        name=name,
    ) as span:
        span.update(input=metadata or {})
        try:
            yield span
        except Exception as exc:
            span.update(
                output={
                    "error": str(exc),
                    "error_type": type(exc).__name__,
                }
            )
            raise
        # finally:
        #     span.update(metadata={"status": "finished"})