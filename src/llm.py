import os

from langchain_ollama import ChatOllama
from langchain_openai import ChatOpenAI

from src.config import LLM_MODEL
from src.loader import load_documents
import dotenv

dotenv.load_dotenv(dotenv_path=os.path.join(os.path.dirname(__file__), ".env"))


def get_llm():
    # return ChatOllama(model=LLM_MODEL, temperature=0)
    llm = ChatOpenAI(
        openai_api_key=os.getenv("QWEN_API_KEY"),
        openai_api_base="https://dashscope.aliyuncs.com/compatible-mode/v1",
        model_name=LLM_MODEL,
        temperature=0,
    )
    return llm


if __name__ == "__main__":
    llm = get_llm()
    response = llm.invoke("用一句话解释什么是 RAG")
    print(response.content)
    docs = load_documents()
    print(f"文档数量: {len(docs)}")
    print(docs[0].page_content[:200])
