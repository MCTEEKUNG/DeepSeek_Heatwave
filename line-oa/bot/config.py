from __future__ import annotations

import os
from dataclasses import dataclass

DEFAULT_FORECAST_URL = (
    "https://raw.githubusercontent.com/MCTEEKUNG/DeepSeek_Heatwave/"
    "main/docs/forecast_provinces.json"
)
DEFAULT_WEBSITE_URL = "https://heat-map-frontend.vercel.app/map"


@dataclass(frozen=True)
class Config:
    channel_secret: str
    channel_access_token: str
    forecast_url: str
    website_url: str


def load_config() -> Config:
    return Config(
        channel_secret=os.environ.get("LINE_CHANNEL_SECRET", ""),
        channel_access_token=os.environ.get("LINE_CHANNEL_ACCESS_TOKEN", ""),
        forecast_url=os.environ.get("FORECAST_URL", DEFAULT_FORECAST_URL),
        website_url=os.environ.get("WEBSITE_URL", DEFAULT_WEBSITE_URL),
    )
