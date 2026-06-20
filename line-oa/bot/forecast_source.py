from __future__ import annotations

import requests


def fetch_forecast(url: str, timeout: int = 10) -> dict:
    resp = requests.get(url, timeout=timeout)
    resp.raise_for_status()
    data = resp.json()
    if "provinces" not in data:
        raise ValueError("forecast JSON ไม่มีคีย์ 'provinces'")
    return data
