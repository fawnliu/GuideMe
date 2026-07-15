"""Load LLM API credentials from a local, gitignored api_setting.yaml.

Every infer/score entry point obtains its API key + host through
resolve_api_credentials(), so credentials live in ONE place at the repo root
instead of being exported from each shell script.

Resolution order (first non-empty wins):
    1. Environment variables LLM_API_KEY / LLM_API_HOST  (handy for CI / one-offs)
    2. api_setting.yaml at the repo root                 (the normal case; gitignored)

To set up: copy api_setting.example.yaml to api_setting.yaml (same repo root)
and fill in your own key/host. api_setting.yaml is listed in .gitignore, so the
secret never gets committed.
"""

import os

import yaml

# repo root = parent of the tools/ package that holds this file
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CONFIG_PATH = os.path.join(_REPO_ROOT, "api_setting.yaml")
EXAMPLE_PATH = os.path.join(_REPO_ROOT, "api_setting.example.yaml")

# accept a couple of spellings so the yaml reads naturally
_KEY_ALIASES = ("api_key", "LLM_API_KEY")
_HOST_ALIASES = ("api_host", "LLM_API_HOST")


def _from_yaml():
    """Return (key, host) read from api_setting.yaml, or ("", "") if absent."""
    if not os.path.exists(CONFIG_PATH):
        return "", ""
    try:
        with open(CONFIG_PATH) as f:
            cfg = yaml.safe_load(f) or {}
    except Exception as exc:  # pragma: no cover - surfaced to the user directly
        raise RuntimeError(f"Failed to parse {CONFIG_PATH}: {exc}") from exc
    if not isinstance(cfg, dict):
        return "", ""
    key = next((str(cfg[k]).strip() for k in _KEY_ALIASES if cfg.get(k)), "")
    host = next((str(cfg[k]).strip() for k in _HOST_ALIASES if cfg.get(k)), "")
    return key, host


def resolve_api_credentials():
    """Return (api_key, api_host); env vars override api_setting.yaml."""
    key = os.environ.get("LLM_API_KEY", "").strip()
    host = os.environ.get("LLM_API_HOST", "").strip()
    if key and host:
        return key, host
    y_key, y_host = _from_yaml()
    return (key or y_key), (host or y_host)
