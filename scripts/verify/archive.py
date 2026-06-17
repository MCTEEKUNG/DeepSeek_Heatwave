"""Archive a published operational forecast JSON by issue_date.

Usage (programmatic):
    from verify.archive import archive_forecast
    dest = archive_forecast(Path("docs/forecast_provinces.json"))
"""
from __future__ import annotations

import json
import logging
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent  # scripts/verify/ -> scripts/ -> repo root
OUT_DIR = ROOT / "outputs" / "operational" / "forecasts"

# Allow `from build_provinces_dataset import ...` and `from verify.observed_labels import ...`
_scripts_dir = str(Path(__file__).resolve().parent.parent)  # scripts/ dir
if _scripts_dir not in sys.path:
    sys.path.insert(0, _scripts_dir)

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


def verify_closed_windows(observed_data_dir: Path) -> int:
    """Score archived forecasts whose target windows have since closed.

    Appends new (issue_date, province_id, lead, probability, base_rate, y_obs)
    rows to outputs/operational/verification/operational_pairs.csv.
    Skips already-scored (issue_date × lead) pairs (idempotent on re-run).

    Parameters
    ----------
    observed_data_dir : Path
        Root dir containing tmax_thailand/ and soil_moisture_thailand/ subdirs.
        Examples: data/raw_backtest/, data/raw_recent/

    Returns
    -------
    int
        Number of new rows appended.
    """
    import pandas as pd
    from build_provinces_dataset import _load_frozen_climatology
    from verify.observed_labels import build_labeled_frame

    # 1. Find archived forecasts
    forecast_files = sorted(OUT_DIR.glob("forecast_*.json"))
    if not forecast_files:
        logger.info("[verify] ไม่พบ archived forecasts ใน %s", OUT_DIR)
        return 0

    # 2. Build observed labels
    tmax_dir = observed_data_dir / "tmax_thailand"
    soil_dir = observed_data_dir / "soil_moisture_thailand"
    clim = _load_frozen_climatology()
    frozen_thr90 = clim["thr90_grid"]

    labeled = build_labeled_frame(tmax_dir, soil_dir, frozen_thr90=frozen_thr90, verbose=False)
    labeled["date"] = pd.to_datetime(labeled["date"])
    latest_date = labeled["date"].max()

    # 3. Load already-scored pairs for idempotency
    op_pairs_path = ROOT / "outputs" / "operational" / "verification" / "operational_pairs.csv"
    already_scored: set[tuple[str, int]] = set()
    if op_pairs_path.exists():
        existing = pd.read_csv(op_pairs_path, usecols=["issue_date", "lead"])
        already_scored = set(zip(
            existing["issue_date"].apply(lambda x: pd.Timestamp(x).strftime("%Y-%m-%d")),
            existing["lead"].astype(int),
        ))

    # 4. Process each archived forecast
    leads = [2, 3, 4, 5, 6]  # must match build_dataset.LEADS
    new_rows: list[dict] = []

    for fc_file in forecast_files:
        try:
            obj = json.loads(fc_file.read_text(encoding="utf-8"))
            provinces_list = obj.get("provinces", [])
            if not provinces_list:
                continue
            issue_date_str = pd.Timestamp(provinces_list[0]["issue_date"]).strftime("%Y-%m-%d")
            issue_ts = pd.Timestamp(issue_date_str)
        except (json.JSONDecodeError, OSError, KeyError, IndexError, ValueError) as e:
            logger.warning("[verify] ไม่สามารถอ่าน %s: %s", fc_file, e)
            continue

        for L in leads:
            window_close = issue_ts + pd.Timedelta(days=7 * L + 6)
            if window_close > latest_date:
                continue  # window not yet closed
            if (issue_date_str, L) in already_scored:
                continue  # already scored this (issue_date × lead)

            # Build probability lookup for this lead from archived JSON
            prob_map: dict[int, tuple[float, float]] = {}
            for prov in provinces_list:
                pid = int(prov["id"])
                for fc in prov.get("forecasts", []):
                    if int(fc["lead_weeks"]) == L:
                        prob_map[pid] = (
                            float(fc["probability"]),
                            float(fc.get("climatology_base_rate", float("nan"))),
                        )
                        break

            # Get observed labels for this issue_date × lead
            label_col = f"y_rm_l{L}"
            day_rows = labeled[(labeled["date"] == issue_ts) & labeled[label_col].notna()]

            for _, row in day_rows.iterrows():
                pid = int(row["province_id"])
                if pid not in prob_map:
                    continue  # province not in archived forecast
                prob, br = prob_map[pid]
                new_rows.append({
                    "issue_date": issue_date_str,
                    "province_id": pid,
                    "lead": int(L),
                    "probability": prob,
                    "base_rate": br,
                    "y_obs": float(row[label_col]),
                })

    if not new_rows:
        logger.info("[verify] ไม่มี closed window ที่ยังไม่ได้ score")
        return 0

    # 5. Append to operational_pairs.csv
    new_df = pd.DataFrame(
        new_rows,
        columns=["issue_date", "province_id", "lead", "probability", "base_rate", "y_obs"],
    )
    op_pairs_path.parent.mkdir(parents=True, exist_ok=True)
    if op_pairs_path.exists():
        new_df.to_csv(op_pairs_path, mode="a", header=False, index=False)
    else:
        new_df.to_csv(op_pairs_path, index=False)

    logger.info("[verify] เพิ่ม %d แถวใน %s", len(new_rows), op_pairs_path)
    return len(new_rows)
