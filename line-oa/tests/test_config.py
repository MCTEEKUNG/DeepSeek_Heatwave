from bot.config import load_config, DEFAULT_WEBSITE_URL, DEFAULT_FORECAST_URL


def test_load_config_reads_env(monkeypatch):
    monkeypatch.setenv("LINE_CHANNEL_SECRET", "sek")
    monkeypatch.setenv("LINE_CHANNEL_ACCESS_TOKEN", "tok")
    monkeypatch.setenv("FORECAST_URL", "https://example/x.json")
    monkeypatch.setenv("WEBSITE_URL", "https://example/map")
    cfg = load_config()
    assert cfg.channel_secret == "sek"
    assert cfg.channel_access_token == "tok"
    assert cfg.forecast_url == "https://example/x.json"
    assert cfg.website_url == "https://example/map"


def test_load_config_uses_defaults(monkeypatch):
    for k in ("LINE_CHANNEL_SECRET", "LINE_CHANNEL_ACCESS_TOKEN", "FORECAST_URL", "WEBSITE_URL"):
        monkeypatch.delenv(k, raising=False)
    cfg = load_config()
    assert cfg.channel_secret == ""
    assert cfg.forecast_url == DEFAULT_FORECAST_URL
    assert cfg.website_url == DEFAULT_WEBSITE_URL
