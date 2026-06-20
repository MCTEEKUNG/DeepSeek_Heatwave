import pytest

from bot import forecast_source


class _FakeResp:
    def __init__(self, status, payload):
        self.status_code = status
        self._payload = payload

    def raise_for_status(self):
        if self.status_code >= 400:
            import requests
            raise requests.HTTPError(f"status {self.status_code}")

    def json(self):
        return self._payload


def test_fetch_forecast_ok(monkeypatch):
    monkeypatch.setattr(forecast_source.requests, "get",
                        lambda url, timeout=10: _FakeResp(200, {"provinces": [{"id": 1}]}))
    data = forecast_source.fetch_forecast("https://x")
    assert data["provinces"][0]["id"] == 1


def test_fetch_forecast_http_error(monkeypatch):
    import requests
    monkeypatch.setattr(forecast_source.requests, "get",
                        lambda url, timeout=10: _FakeResp(404, {}))
    with pytest.raises(requests.HTTPError):
        forecast_source.fetch_forecast("https://x")


def test_fetch_forecast_missing_provinces(monkeypatch):
    monkeypatch.setattr(forecast_source.requests, "get",
                        lambda url, timeout=10: _FakeResp(200, {"foo": 1}))
    with pytest.raises(ValueError):
        forecast_source.fetch_forecast("https://x")
