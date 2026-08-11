"""
Ingests the curated financial-education knowledge base into the vector store.

Usage:
    python -m src.rag.ingest

Expects markdown/text articles under src/rag/knowledge_base/<category>/*.md
Each file's metadata (category, title, source) is derived from its path
and an optional YAML frontmatter block:

---
title: What is a Roth IRA?
source: Investopedia (adapted)
category: tax_education
---
<article body...>
"""
import re
from pathlib import Path
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from src.rag.vector_store import get_vectorstore
from config import get_logger

logger = get_logger(__name__)
KB_DIR = Path(__file__).parent / "knowledge_base"

FRONTMATTER_RE = re.compile(r"^---\s*\n(.*?)\n---\s*\n", re.DOTALL)


def _parse_frontmatter(text: str, fallback_category: str, fallback_title: str):
    match = FRONTMATTER_RE.match(text)
    meta = {"category": fallback_category, "title": fallback_title, "source": "internal"}
    body = text
    if match:
        block = match.group(1)
        body = text[match.end():]
        for line in block.splitlines():
            if ":" in line:
                k, v = line.split(":", 1)
                meta[k.strip()] = v.strip()
    return meta, body.strip()


def load_documents() -> list[Document]:
    docs = []
    if not KB_DIR.exists():
        logger.warning(f"Knowledge base dir not found: {KB_DIR}")
        return docs

    for category_dir in KB_DIR.iterdir():
        if not category_dir.is_dir():
            continue
        for article_path in category_dir.glob("*.md"):
            text = article_path.read_text(encoding="utf-8")
            meta, body = _parse_frontmatter(
                text, fallback_category=category_dir.name, fallback_title=article_path.stem
            )
            docs.append(Document(page_content=body, metadata=meta))
    return docs


def ingest(chunk_size: int = 800, chunk_overlap: int = 120):
    docs = load_documents()
    if not docs:
        logger.warning("No documents found to ingest. Add .md files under src/rag/knowledge_base/<category>/")
        return 0

    splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    chunks = splitter.split_documents(docs)

    vs = get_vectorstore()
    vs.add_documents(chunks)
    logger.info(f"Ingested {len(docs)} articles -> {len(chunks)} chunks")
    return len(chunks)


if __name__ == "__main__":
    ingest()
