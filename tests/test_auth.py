import os
import tempfile
import pytest
from unittest.mock import patch


class TestAuthManager:
    def test_ensure_users_yaml_creates_valid_structure(self):
        from src.auth import auth_manager
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "users.yaml")
            with patch.object(auth_manager.settings, "USERS_YAML_PATH", path):
                result_path = auth_manager.ensure_users_yaml_exists()
                assert result_path == path
                assert os.path.exists(path)

                import yaml
                with open(path) as f:
                    data = yaml.safe_load(f)
                assert data["credentials"]["usernames"] == {}
                assert "cookie" in data
                assert data["cookie"]["name"] == "finance_assistant_auth"

    def test_ensure_users_yaml_is_idempotent(self):
        """Must never overwrite an existing file — that would delete real
        registered users' credentials."""
        from src.auth import auth_manager
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "users.yaml")
            with patch.object(auth_manager.settings, "USERS_YAML_PATH", path):
                auth_manager.ensure_users_yaml_exists()

                import yaml
                with open(path) as f:
                    data = yaml.safe_load(f)
                data["credentials"]["usernames"]["existing_user"] = {"email": "x@example.com"}
                with open(path, "w") as f:
                    yaml.dump(data, f)

                auth_manager.ensure_users_yaml_exists()

                with open(path) as f:
                    data_after = yaml.safe_load(f)
                assert "existing_user" in data_after["credentials"]["usernames"], (
                    "Existing user data was wiped — ensure_users_yaml_exists must be idempotent"
                )

    def test_get_authenticator_returns_configured_instance(self):
        from src.auth import auth_manager
        import streamlit_authenticator as stauth
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "users.yaml")
            with patch.object(auth_manager.settings, "USERS_YAML_PATH", path):
                authenticator = auth_manager.get_authenticator()
                assert isinstance(authenticator, stauth.Authenticate)


class TestUserData:
    def test_missing_user_returns_empty_list(self):
        from src.auth import user_data
        with tempfile.TemporaryDirectory() as tmp:
            with patch.object(user_data.settings, "USER_DATA_DIR", tmp):
                result = user_data.load_user_portfolio("nonexistent_user")
                assert result == []

    def test_save_and_load_round_trip(self):
        from src.auth import user_data
        with tempfile.TemporaryDirectory() as tmp:
            with patch.object(user_data.settings, "USER_DATA_DIR", tmp):
                holdings = [{"ticker": "AAPL", "shares": 10, "cost_basis": 150.0}]
                assert user_data.save_user_portfolio("alice", holdings) is True
                loaded = user_data.load_user_portfolio("alice")
                assert loaded == holdings

    def test_two_users_are_isolated(self):
        from src.auth import user_data
        with tempfile.TemporaryDirectory() as tmp:
            with patch.object(user_data.settings, "USER_DATA_DIR", tmp):
                user_data.save_user_portfolio("alice", [{"ticker": "AAPL", "shares": 1, "cost_basis": 1.0}])
                user_data.save_user_portfolio("bob", [{"ticker": "TSLA", "shares": 2, "cost_basis": 2.0}])
                assert user_data.load_user_portfolio("alice")[0]["ticker"] == "AAPL"
                assert user_data.load_user_portfolio("bob")[0]["ticker"] == "TSLA"

    @pytest.mark.parametrize("bad_username", [
        "../../../etc/passwd",
        "../evil",
        "user/../../etc",
        "user\\..\\..\\windows",
        "",
    ])
    def test_path_traversal_attempts_are_rejected(self, bad_username):
        from src.auth import user_data
        with tempfile.TemporaryDirectory() as tmp:
            with patch.object(user_data.settings, "USER_DATA_DIR", tmp):
                with pytest.raises(ValueError):
                    user_data.load_user_portfolio(bad_username)
                with pytest.raises(ValueError):
                    user_data.save_user_portfolio(bad_username, [])

    def test_valid_usernames_with_special_chars_allowed(self):
        """Usernames with dots/hyphens/underscores are legitimate and
        shouldn't be rejected by the same guard that blocks traversal."""
        from src.auth import user_data
        with tempfile.TemporaryDirectory() as tmp:
            with patch.object(user_data.settings, "USER_DATA_DIR", tmp):
                for username in ["alice.smith", "bob-jones", "user_123"]:
                    assert user_data.save_user_portfolio(username, []) is True

    def test_corrupted_file_degrades_gracefully(self):
        from src.auth import user_data
        with tempfile.TemporaryDirectory() as tmp:
            with patch.object(user_data.settings, "USER_DATA_DIR", tmp):
                bad_path = os.path.join(tmp, "corrupt_portfolio.json")
                with open(bad_path, "w") as f:
                    f.write("{not valid json!!!")
                with patch.object(user_data, "_portfolio_path", return_value=bad_path):
                    result = user_data.load_user_portfolio("corrupt")
                    assert result == []
