import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core import killswitch


@pytest.fixture(autouse=True)
def isolated_state(tmp_path, monkeypatch):
    """Every test gets its own kill switch file and no env override."""
    path = str(tmp_path / "killswitch.json")
    monkeypatch.setattr(killswitch, "STATE_PATH", path)
    monkeypatch.delenv("JARVIS_KILL", raising=False)
    return path
