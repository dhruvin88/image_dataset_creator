from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Dict, Optional

try:
    import keyring

    HAS_KEYRING = True
except ImportError:
    HAS_KEYRING = False

KEYRING_SERVICE = "idc"
CONFIG_DIR = Path.home() / ".idc"
CONFIG_FILE = CONFIG_DIR / "config.json"

SOURCES = ["unsplash", "pexels", "pixabay"]

_ENV_VARS: Dict[str, str] = {
    "unsplash": "IDC_UNSPLASH_KEY",
    "pexels": "IDC_PEXELS_KEY",
    "pixabay": "IDC_PIXABAY_KEY",
}


def get_api_key(source: str) -> Optional[str]:
    """Get API key for a source. Priority: env var > keyring > config file."""
    env_var = _ENV_VARS.get(source)
    if env_var:
        val = os.environ.get(env_var)
        if val:
            return val

    if HAS_KEYRING:
        try:
            val = keyring.get_password(KEYRING_SERVICE, source)
            if val:
                return val
        except Exception:
            pass

    if CONFIG_FILE.exists():
        try:
            with open(CONFIG_FILE) as f:
                config = json.load(f)
            return config.get(f"{source}_key")
        except Exception:
            pass

    return None


def set_api_key(source: str, key: str) -> None:
    """Store API key. Prefers keyring, falls back to config file."""
    if HAS_KEYRING:
        try:
            keyring.set_password(KEYRING_SERVICE, source, key)
            return
        except Exception:
            pass

    CONFIG_DIR.mkdir(parents=True, exist_ok=True)
    config: dict = {}
    if CONFIG_FILE.exists():
        try:
            with open(CONFIG_FILE) as f:
                config = json.load(f)
        except Exception:
            pass
    config[f"{source}_key"] = key
    with open(CONFIG_FILE, "w") as f:
        json.dump(config, f, indent=2)


def get_all_keys() -> Dict[str, Optional[str]]:
    return {source: get_api_key(source) for source in SOURCES}
