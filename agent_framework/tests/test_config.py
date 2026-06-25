import importlib

import config as config_mod


def test_defaults_when_env_missing(monkeypatch):
    monkeypatch.delenv("MAX_STEPS", raising=False)
    monkeypatch.delenv("DEEPSEEK_API_KEY", raising=False)
    importlib.reload(config_mod)
    assert config_mod.MAX_STEPS == 10
    assert config_mod.MODEL == "deepseek-chat"
    assert config_mod.HOST == "127.0.0.1"
    assert config_mod.PORT == 8000


def test_env_overrides(monkeypatch):
    monkeypatch.setenv("MAX_STEPS", "5")
    monkeypatch.setenv("PORT", "9000")
    importlib.reload(config_mod)
    assert config_mod.MAX_STEPS == 5
    assert config_mod.PORT == 9000

# -- Phase 3: non-numeric env falls back to default --


def test_non_numeric_env_falls_back_to_default(monkeypatch):
    monkeypatch.setenv("MAX_STEPS", "abc")
    monkeypatch.setenv("PORT", "xyz")
    import importlib

    import config as config_mod
    importlib.reload(config_mod)
    assert config_mod.MAX_STEPS == 10
    assert config_mod.PORT == 8000
