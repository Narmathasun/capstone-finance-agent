"""
Vector store abstraction.
Dev default: Chroma (local, zero-setup, persists to disk).
Production: Pinecone (managed, scalable) — switch via VECTOR_BACKEND env var.
Both are wrapped behind the same get_retriever() interface so agent
code never needs to know which backend is active.
"""
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma
from config import settings, get_logger

logger = get_logger(__name__)

_embeddings = OpenAIEmbeddings(
    model="text-embedding-3-small", api_key=settings.OPENAI_API_KEY
)

_vectorstore = None


def get_vectorstore():
    global _vectorstore
    if _vectorstore is not None:
        return _vectorstore

    if settings.VECTOR_BACKEND == "pinecone":
        from langchain_pinecone import PineconeVectorStore
        from pinecone import Pinecone, ServerlessSpec

        pc = Pinecone(api_key=settings.PINECONE_API_KEY)
        existing = [idx["name"] for idx in pc.list_indexes()]
        if settings.PINECONE_INDEX_NAME not in existing:
            pc.create_index(
                name=settings.PINECONE_INDEX_NAME,
                dimension=1536,  # text-embedding-3-small
                metric="cosine",
                spec=ServerlessSpec(cloud="aws", region=settings.PINECONE_ENV),
            )
        _vectorstore = PineconeVectorStore(
            index_name=settings.PINECONE_INDEX_NAME, embedding=_embeddings
        )
        logger.info("Using Pinecone vector backend")
    else:
        _vectorstore = Chroma(
            collection_name="finance_kb",
            embedding_function=_embeddings,
            persist_directory=settings.CHROMA_PERSIST_DIR,
        )
        logger.info(f"Using Chroma vector backend ({settings.CHROMA_PERSIST_DIR})")

    return _vectorstore


def get_retriever(k: int = 4, category: str | None = None):
    vs = get_vectorstore()
    search_kwargs = {"k": k}
    if category:
        search_kwargs["filter"] = {"category": category}
    return vs.as_retriever(search_kwargs=search_kwargs)


def similarity_search(query: str, k: int = 4, category: str | None = None):
    retriever = get_retriever(k=k, category=category)
    docs = retriever.invoke(query)
    return [
        {
            "content": d.page_content,
            "source": d.metadata.get("source", "unknown"),
            "category": d.metadata.get("category", "general"),
            "title": d.metadata.get("title", ""),
        }
        for d in docs
    ]
