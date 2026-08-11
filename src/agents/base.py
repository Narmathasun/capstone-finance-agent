"""
Shared helpers so every agent follows the same
RAG-retrieval -> LLM-processing -> response pattern
without duplicating boilerplate.
"""
from langchain_openai import ChatOpenAI
from config import settings, get_logger
from src.rag.vector_store import similarity_search

logger = get_logger(__name__)

def get_llm(temperature: float = 0.2) -> ChatOpenAI:
    return ChatOpenAI(
        model=settings.OPENAI_MODEL,
        api_key=settings.OPENAI_API_KEY,
        temperature=temperature,
    )


def retrieve_context(query: str, category: str | None = None, k: int = 4):
    try:
        return similarity_search(query, k=k, category=category)
    except Exception as e:
        logger.error(f"RAG retrieval failed: {e}")
        return []  # agents must handle empty context gracefully


def format_context_block(docs: list[dict]) -> str:
    if not docs:
        return "No relevant knowledge-base articles found."
    blocks = []
    for i, d in enumerate(docs, 1):
        blocks.append(f"[{i}] ({d.get('title') or 'untitled'} — {d.get('source')})\n{d['content']}")
    return "\n\n".join(blocks)


def sources_from_docs(docs: list[dict]) -> list[str]:
    seen, out = set(), []
    for d in docs:
        s = d.get("source", "unknown")
        if s not in seen:
            seen.add(s)
            out.append(s)
    return out
