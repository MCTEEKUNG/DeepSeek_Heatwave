"""Archive a published operational forecast JSON by issue_date.

Usage (programmatic):
    from verify.archive import archive_forecast
    dest = archive_forecast(Path("docs/forecast_provinces.json"))
"""
from __future__ import annotations

import json
import logging
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent  # scripts/verify/ -> scripts/ -> repo root
OUT_DIR = ROOT / "outputs" / "operational" / "forecasts"

logger = logging.getLogger(__name__)


def archive_forecast(forecast_json_path: Path) -> Path | None:
    """Copy *forecast_json_path* into the operational forecast archive.

    The archive filename is ``forecast_{issue_date}.json`` where *issue_date*
    is read from ``obj["provinces"][0]["issue_date"]``.

    Returns
    -------
    Path
        Destination path when the file was successfully archived.
    None
        When an archive for this *issue_date* already exists (idempotent
        no-op — the existing file is preserved unchanged).
    """
    try:
        obj = json.loads(forecast_json_path.read_text(encoding="utf-8"))
        issue_date: str = obj["provinces"][0]["issue_date"]
    except (FileNotFoundError, json.JSONDecodeError, KeyError, IndexError) as exc:
        logger.warning("[archive] ไม่สามารถอ่าน issue_date จาก %s: %s", forecast_json_path, exc)
        return None

    dest = OUT_DIR / f"forecast_{issue_date}.json"

    if dest.exists():
        logger.info("[archive] ข้าม — archive สำหรับ %s มีอยู่แล้ว (%s) ไม่เขียนทับ", issue_date, dest)
        return None

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(forecast_json_path, dest)
    logger.info("[archive] archived -> %s", dest)
    return dest
