"""
Multi-user authentication (signup + login), built on streamlit-authenticator.

Credentials are stored in a YAML file (settings.USERS_YAML_PATH) that the
library reads from and writes back to automatically — including new
registrations — so this module's job is just to ensure that file exists
with a valid empty structure on first run, and to hand back a configured
Authenticate instance for app.py to drive the actual login/register UI.

Passwords are hashed by the library itself (bcrypt) before ever touching
disk — this module never sees or stores a plain-text password.
"""
import os
import yaml
import streamlit_authenticator as stauth
from config import settings, get_logger

logger = get_logger(__name__)


def _default_credentials_structure() -> dict:
    return {
        "credentials": {"usernames": {}},
        "cookie": {
            "name": "finance_assistant_auth",
            "key": settings.AUTH_COOKIE_KEY,
            "expiry_days": 30,
        },
    }


def ensure_users_yaml_exists() -> str:
    """
    Creates settings.USERS_YAML_PATH with a valid empty structure if it
    doesn't already exist. Idempotent — never overwrites an existing file,
    so real registered users are never at risk of being wiped by this call.
    Returns the path (for convenience at the call site).
    """
    path = settings.USERS_YAML_PATH
    if not os.path.exists(path):
        parent = os.path.dirname(path)
        if parent:
            os.makedirs(parent, exist_ok=True)
        with open(path, "w") as f:
            yaml.dump(_default_credentials_structure(), f, default_flow_style=False)
        logger.info(f"Created new users.yaml at {path} (no users yet)")
    return path


def get_authenticator() -> "stauth.Authenticate":
    """
    Single entry point app.py uses to get a ready-to-use Authenticate
    instance. Handles ensuring the credentials file exists first.
    """
    path = ensure_users_yaml_exists()
    return stauth.Authenticate(
        credentials=path,
        cookie_name="finance_assistant_auth",
        cookie_key=settings.AUTH_COOKIE_KEY,
        cookie_expiry_days=30,
    )
