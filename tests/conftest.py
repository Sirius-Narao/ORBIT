import pytest


@pytest.fixture(autouse=True)
def isolated_orbit_settings(tmp_path, monkeypatch):
    """
    Point ORBIT's settings file at a path inside this test's tmp_path, which
    doesn't exist until a test writes it. That way a developer's real
    ~/.orbit/config.toml (its workspace, defaults or theme) can never leak
    into a test, and a test that writes settings can never touch it.
    """
    path = tmp_path / "orbit_settings" / "config.toml"
    monkeypatch.setenv("ORBIT_CONFIG", str(path))
    return path
