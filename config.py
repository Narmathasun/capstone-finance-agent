"""
Central configuration. Loads from environment / .env file.
Never hard-code secrets here — this file only reads them.
"""
import os
import logging
from dotenv import load_dotenv

load_dotenv(override=True)

# ---- Streamlit Cloud secrets bridge ----
# Locally, secrets come from .env (loaded above). On Streamlit Community
# Cloud, secrets are entered via the app dashboard and exposed through
# st.secrets, NOT as OS environment variables. This block copies any
# st.secrets values into os.environ (without overwriting anything .env
# already set) so the rest of this file — and every module that reads
# settings.* — works identically in both environments with no branching.
# Wrapped defensively: this must never break non-Streamlit contexts like
# pytest, the MCP server, or plain CLI scripts, where no Streamlit runtime
# or secrets.toml exists.
try:
    import streamlit.runtime as _st_runtime
    if _st_runtime.exists():
        import streamlit as st
        for _key, _value in st.secrets.items():
            os.environ.setdefault(_key, str(_value))
except Exception:
    pass

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

    # Optional shared-password gate for public deployments (see app.py)
    APP_PASSWORD = os.getenv("APP_PASSWORD", "")

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
