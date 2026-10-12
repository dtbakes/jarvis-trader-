import os
import sys

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
# The backtest modules (regime, engine_a, engine_b) import siblings as
# top-level modules (`from indicators import ...`), so core/ must be on the path.
sys.path.insert(0, os.path.join(ROOT, "core"))

# core.agent builds an Anthropic client at import time; give it a dummy key so
# imports work in CI. No test makes a real API call.
os.environ.setdefault("ANTHROPIC_API_KEY", "test-key-not-real")

from core import killswitch  # noqa: E402


@pytest.fixture(autouse=True)
def isolated_state(tmp_path, monkeypatch):
    """Every test gets its own kill switch file and no env override."""
    path = str(tmp_path / "killswitch.json")
    monkeypatch.setattr(killswitch, "STATE_PATH", path)
    monkeypatch.delenv("JARVIS_KILL", raising=False)
    return path
