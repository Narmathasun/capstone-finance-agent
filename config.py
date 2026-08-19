"""
Central configuration. Loads from environment / .env file.
Never hard-code secrets here — this file only reads them.

Precedence (highest wins): real OS environment variables > .env file >
config.yaml defaults > hardcoded fallback defaults below. config.yaml is
for NON-SECRET operational defaults only (model name, cache TTL, log
level, etc.) — it's meant to be safely committed to version control.
API keys and other secrets always come from .env (local) or a secrets
manager (cloud), never from config.yaml.
"""
import os
import logging
import yaml
from dotenv import load_dotenv

# ---- YAML config defaults (lowest precedence, safe to commit) ----
# Loaded FIRST, using setdefault, so both a real .env file and any
# already-set shell environment variable can still override any of these.
_CONFIG_YAML_PATH = os.getenv(
    "CONFIG_YAML_PATH", os.path.join(os.path.dirname(__file__), "config.yaml")
)
try:
    if os.path.exists(_CONFIG_YAML_PATH):
        with open(_CONFIG_YAML_PATH, "r") as _f:
            _yaml_config = yaml.safe_load(_f) or {}
        for _key, _value in _yaml_config.items():
            os.environ.setdefault(_key, str(_value))
except Exception as _e:
    # Never let a malformed YAML file take down the whole app — fall back
    # to hardcoded defaults / .env / real env vars as if it weren't there.
    print(f"Warning: could not load {_CONFIG_YAML_PATH}: {_e}")

load_dotenv(override=True)

# ---- Streamlit Cloud secrets bridge ----
# Locally, secrets come from .env (loaded above). On Streamlit Community
# Cloud, secrets are entered via the app dashboard and exposed through
# st.secrets, NOT as OS environment variables. This block copies any
# st.secrets values into os.environ (without overwriting anything .env
# already set) so the rest of this file — and every module that reads
# settings.* — works identically in both environments with no branching.
#
# IMPORTANT: we check whether a secrets.toml file actually exists on disk
# BEFORE touching st.secrets at all. Merely accessing st.secrets when no
# such file exists causes Streamlit to display a "No secrets found"
# warning in the app UI — and since config.py is imported (running this
# code) before app.py's required first command, st.set_page_config(), that
# warning itself counts as a Streamlit command and breaks the "must be the
# first command" rule with a StreamlitSetPageConfigMustBeFirstCommandError.
# Checking file existence first avoids ever triggering that warning when
# running locally with no secrets.toml (the normal case — local dev uses
# .env instead).
try:
    import streamlit.runtime as _st_runtime
    if _st_runtime.exists():
        _secrets_candidates = [
            os.path.expanduser(os.path.join("~", ".streamlit", "secrets.toml")),
            os.path.join(os.getcwd(), ".streamlit", "secrets.toml"),
        ]
        if any(os.path.exists(p) for p in _secrets_candidates):
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

    # Conversation memory / session persistence backend
    # "memory" (default): in-process only, wiped on restart — fine for
    #   ephemeral environments like Streamlit Community Cloud, where local
    #   disk writes don't survive a redeploy anyway.
    # "sqlite": persists to a local .sqlite file — durable across restarts
    #   on a machine/server you control (e.g. running the MCP server or
    #   Streamlit locally, or on a VM/container with a real persistent disk).
    CHECKPOINTER_BACKEND = os.getenv("CHECKPOINTER_BACKEND", "memory")
    SQLITE_CHECKPOINT_PATH = os.getenv("SQLITE_CHECKPOINT_PATH", "./checkpoints.sqlite")

    # Real multi-user accounts (signup + login), opt-in — see src/auth/.
    # When False (default), app.py falls back to the single shared
    # APP_PASSWORD gate above, preserving existing deployment behavior.
    ENABLE_MULTI_USER_AUTH = os.getenv("ENABLE_MULTI_USER_AUTH", "false").lower() == "true"
    USERS_YAML_PATH = os.getenv("USERS_YAML_PATH", "./users.yaml")
    # Signs/encrypts the auth cookie — set a real random secret in .env for
    # any real deployment. The fallback here is fine for local dev only.
    AUTH_COOKIE_KEY = os.getenv("AUTH_COOKIE_KEY", "dev-only-insecure-default-change-me")
    USER_DATA_DIR = os.getenv("USER_DATA_DIR", "./user_data")

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
