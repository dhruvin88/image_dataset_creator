"""Tests for API key config management."""
from __future__ import annotations

import json
import os
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest


class TestGetApiKey:
    def test_env_var_takes_priority(self, monkeypatch, tmp_path):
        monkeypatch.setenv("IDC_UNSPLASH_KEY", "env_key_123")
        from idc import config as cfg

        # Write a config file with a different key
        cfg.CONFIG_DIR.mkdir(parents=True, exist_ok=True)
        cfg.CONFIG_FILE.write_text(json.dumps({"unsplash_key": "file_key"}))

        key = cfg.get_api_key("unsplash")
        assert key == "env_key_123"

    def test_returns_none_when_not_set(self, monkeypatch):
        monkeypatch.delenv("IDC_UNSPLASH_KEY", raising=False)
        monkeypatch.delenv("IDC_PEXELS_KEY", raising=False)

        with patch("idc.config.HAS_KEYRING", False):
            with patch("idc.config.CONFIG_FILE", Path("/nonexistent/config.json")):
                from idc import config as cfg
                key = cfg.get_api_key("pexels")
                assert key is None

    def test_set_and_get_via_config_file(self, tmp_path, monkeypatch):
        monkeypatch.delenv("IDC_PIXABAY_KEY", raising=False)

        with patch("idc.config.HAS_KEYRING", False), \
             patch("idc.config.CONFIG_DIR", tmp_path), \
             patch("idc.config.CONFIG_FILE", tmp_path / "config.json"):
            from idc import config as cfg

            cfg.set_api_key("pixabay", "test_key_xyz")
            result = cfg.get_api_key("pixabay")
            assert result == "test_key_xyz"

    def test_get_all_keys_returns_dict(self):
        from idc import config as cfg

        keys = cfg.get_all_keys()
        assert set(keys.keys()) == {"unsplash", "pexels", "pixabay"}

    def test_keyring_path_returns_key(self, monkeypatch):
        monkeypatch.delenv("IDC_UNSPLASH_KEY", raising=False)

        mock_keyring = MagicMock()
        mock_keyring.get_password.return_value = "keyring_key"

        with patch("idc.config.HAS_KEYRING", True), \
             patch("idc.config.keyring", mock_keyring, create=True):
            from idc import config as cfg
            key = cfg.get_api_key("unsplash")

        assert key == "keyring_key"

    def test_keyring_failure_falls_back_to_config_file(self, tmp_path, monkeypatch):
        monkeypatch.delenv("IDC_UNSPLASH_KEY", raising=False)

        mock_keyring = MagicMock()
        mock_keyring.get_password.side_effect = Exception("keyring unavailable")

        config_file = tmp_path / "config.json"
        config_file.write_text(json.dumps({"unsplash_key": "file_fallback"}))

        with patch("idc.config.HAS_KEYRING", True), \
             patch("idc.config.keyring", mock_keyring, create=True), \
             patch("idc.config.CONFIG_FILE", config_file):
            from idc import config as cfg
            key = cfg.get_api_key("unsplash")

        assert key == "file_fallback"

    def test_set_api_key_uses_keyring_when_available(self, monkeypatch):
        mock_keyring = MagicMock()
        mock_keyring.set_password.return_value = None

        with patch("idc.config.HAS_KEYRING", True), \
             patch("idc.config.keyring", mock_keyring, create=True):
            from idc import config as cfg
            cfg.set_api_key("pexels", "newkey")

        mock_keyring.set_password.assert_called_once_with("idc", "pexels", "newkey")

    def test_set_api_key_keyring_failure_writes_file(self, tmp_path, monkeypatch):
        mock_keyring = MagicMock()
        mock_keyring.set_password.side_effect = Exception("keyring error")

        with patch("idc.config.HAS_KEYRING", True), \
             patch("idc.config.keyring", mock_keyring, create=True), \
             patch("idc.config.CONFIG_DIR", tmp_path), \
             patch("idc.config.CONFIG_FILE", tmp_path / "config.json"):
            from idc import config as cfg
            cfg.set_api_key("pixabay", "fallback_key")

        stored = json.loads((tmp_path / "config.json").read_text())
        assert stored["pixabay_key"] == "fallback_key"
