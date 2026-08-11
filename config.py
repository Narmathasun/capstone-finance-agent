"""
Central configuration. Loads from environment / .env file.
Never hard-code secrets here — this file only reads them.
"""
import os
import logging
from dotenv import load_dotenv

load_dotenv(override=True) 

class Settings:
    # LLM
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
    OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")

    # Market data
    ALPHA_VANTAGE_API_KEY = os.getenv("ALPHA_VANTAGE_API_KEY", "")

    # Vector DB
    VECTOR_BACKEND = os.getenv("VECTOR_BACKEND", "chroma")  # chroma | pinecone | faiss
    CHROMA_PERSIST_DIR = os.getenv("CHROMA_PERSIST_DIR", "./chroma_db")
    PINECONE_API_KEY = os.getenv("PINECONE_API_KEY", "")
    PINECONE_ENV = os.getenv("PINECONE_ENV", "us-east-1")
    PINECONE_INDEX_NAME = os.getenv("PINECONE_INDEX_NAME", "finance-kb")

    NEWS_API_KEY = os.getenv("NEWS_API_KEY", "")

    LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
    CACHE_TTL_SECONDS = int(os.getenv("CACHE_TTL_SECONDS", "300"))

    @classmethod
    def validate(cls):
        """Fail fast and loud if critical secrets are missing."""
        missing = []
        if not cls.OPENAI_API_KEY:
            missing.append("OPENAI_API_KEY")
        if cls.VECTOR_BACKEND == "pinecone" and not cls.PINECONE_API_KEY:
            missing.append("PINECONE_API_KEY")
        if missing:
            raise EnvironmentError(
                f"Missing required environment variables: {', '.join(missing)}. "
                "Copy .env.example to .env and fill them in."
            )

settings = Settings()

logging.basicConfig(
    level=getattr(logging, settings.LOG_LEVEL, logging.INFO),
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)

def get_logger(name: str) -> logging.Logger:
    return logging.getLogger(name)
