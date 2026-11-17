import pytest

from tests.fakes import make_client


@pytest.fixture(autouse=True)
def user_voices(monkeypatch, tmp_path_factory):
    """Registered voices go to a temporary directory, never the user's real one."""
    d = tmp_path_factory.mktemp("user_voices")
    monkeypatch.setenv("LOCAL_TTS_VOICES_DIR", str(d))
    return d


@pytest.fixture
def client(monkeypatch, tmp_path):
    yield from make_client(monkeypatch, tmp_path)
