"""
Per-user portfolio persistence.

Deliberately kept separate from auth_manager.py / the credentials YAML —
portfolio holdings are application data, not authentication data, and
mixing the two would make the credentials file (which streamlit-
authenticator manages and writes to automatically) fragile to edit by two
different pieces of code. Each user's portfolio is stored as its own small
JSON file under settings.USER_DATA_DIR, keyed by username.
"""
import os
import json
import re
from config import settings, get_logger

logger = get_logger(__name__)

_SAFE_USERNAME_RE = re.compile(r"^[a-zA-Z0-9_.-]+$")


def _portfolio_path(username: str) -> str:
    """
    Validates the username against a strict allow-list pattern before ever
    building a filesystem path from it — this is what stands between a
    username and a path-traversal bug (e.g. a username of '../../etc').
    streamlit-authenticator's own Validator already restricts usernames at
    registration time, but this module doesn't assume that and re-checks
    independently, since it's the code actually touching the filesystem.
    """
    if not username or not _SAFE_USERNAME_RE.match(username):
        raise ValueError(f"Invalid username for file storage: {username!r}")
    os.makedirs(settings.USER_DATA_DIR, exist_ok=True)
    return os.path.join(settings.USER_DATA_DIR, f"{username}_portfolio.json")


def load_user_portfolio(username: str) -> list:
    """Returns [] if the user has no saved portfolio yet — never raises for
    a simply-missing file, only for a genuinely invalid username or
    corrupted file content."""
    path = _portfolio_path(username)
    if not os.path.exists(path):
        return []
    try:
        with open(path, "r") as f:
            data = json.load(f)
        return data.get("holdings", [])
    except (json.JSONDecodeError, OSError) as e:
        logger.error(f"Failed to load portfolio for {username}: {e}")
        return []


def save_user_portfolio(username: str, holdings: list) -> bool:
    path = _portfolio_path(username)
    try:
        with open(path, "w") as f:
            json.dump({"username": username, "holdings": holdings}, f, indent=2)
        return True
    except OSError as e:
        logger.error(f"Failed to save portfolio for {username}: {e}")
        return False
