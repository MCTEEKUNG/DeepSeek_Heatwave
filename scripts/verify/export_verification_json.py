"""Export verification.json for the frontend accuracy screen.

Reads pre-computed CSVs from outputs/operational/verification/ and writes
docs/verification.json in the schema expected by services/forecastService.ts.

Run:
  python scripts/verify/export_verification_json.py
  python scripts/verify/export_verification_json.py --pairs outputs/operational/verification/operational_pairs.csv
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent.parent
VERIFY_DIR = ROOT / "outputs" / "operational" / "verification"
OUT_JSON = ROOT / "docs" / "verification.json"

# BSS per-week thresholds for hit/near/miss classification
_HIT_BSS = 0.05
_NEAR_BSS = -0.05


def _bss(y: np.ndarray, p: np.ndarray, br: np.ndarray) -> float:
    """Brier Skill Score vs per-row climatology base_rate."""
    bs = float(np.mean((p - y) ** 2))
    bs_clim = float(np.mean((br - y) ** 2))
    return 0.0 if bs_clim == 0 else 1.0 - bs / bs_clim


def _outcome(bss_val: float) -> str:
    if bss_val > _HIT_BSS:
        return "hit"
    if bss_val > _NEAR_BSS:
        return "near"
    return "miss"


def export(
    pairs_path: Path = VERIFY_DIR / "backtest_pairs.csv",
    scorecard_path: Path = VERIFY_DIR / "scorecard.csv",
    reliability_path: Path = VERIFY_DIR / "reliability.csv",
    out_path: Path = OUT_JSON,
) -> dict:
    """Build verification.json from pre-computed CSVs and write to out_path.

    Skips any section whose source CSV is missing — produces a partial JSON
    rather than failing, so the frontend can still show whatever is available.
    Returns the dict that was written.
    """
    result: dict = {}

    # --- BSS overall + per-lead (scorecard.csv) ---
    if scorecard_path.exists():
        sc = pd.read_csv(scorecard_path)

        row_all = sc[(sc["stratum"] == "all") & (sc["metric"] == "bss")]
        if not row_all.empty:
            v = float(row_all.iloc[0]["value"])
            if not math.isnan(v):
                result["bss"] = round(v, 4)

        per_lead = []
        for lead_int in [2, 3, 4, 5, 6]:
            row_l = sc[(sc["stratum"] == f"lead_{lead_int}") & (sc["metric"] == "bss")]
            if not row_l.empty:
                v = float(row_l.iloc[0]["value"])
                if not math.isnan(v):
                    per_lead.append({"lead": lead_int, "bss": round(v, 4)})
        if per_lead:
            result["per_lead"] = per_lead
    else:
        print(f"[ข้าม] ไม่พบ {scorecard_path} — ข้าม bss / per_lead")

    # --- Calibration (reliability.csv, stratum='all') ---
    if reliability_path.exists():
        rel = pd.read_csv(reliability_path)
        rel_all = rel[(rel["stratum"] == "all") & (rel["count"] > 0)].copy()
        if not rel_all.empty:
            calib = [
                {"predicted": round(float(r["mean_pred"]), 4),
                 "observed": round(float(r["obs_freq"]), 4)}
                for _, r in rel_all.sort_values("mean_pred").iterrows()
                if not (math.isnan(r["mean_pred"]) or math.isnan(r["obs_freq"]))
            ]
            if calib:
                result["calibration"] = calib
    else:
        print(f"[ข้าม] ไม่พบ {reliability_path} — ข้าม calibration")

    # --- Period + week-by-week track record (pairs CSV) ---
    if pairs_path.exists():
        pairs = pd.read_csv(pairs_path, parse_dates=["issue_date"])
        pairs = pairs.dropna(subset=["y_obs"])   # ยังไม่มี label จริง → ข้าม
        if not pairs.empty:
            dates = sorted(pairs["issue_date"].dt.date.unique())
            result["period"] = {
                "start": str(dates[0]),
                "end": str(dates[-1]),
                "weeks": len(dates),
            }
            weeks_list = []
            for d in dates:
                sub = pairs[pairs["issue_date"].dt.date == d]
                y = sub["y_obs"].to_numpy(float)
                p = sub["probability"].to_numpy(float)
                br = sub["base_rate"].to_numpy(float)
                bss_d = _bss(y, p, br)
                weeks_list.append({"target_date": str(d), "outcome": _outcome(bss_d)})
            result["weeks"] = weeks_list
    else:
        print(f"[ข้าม] ไม่พบ {pairs_path} — ข้าม period / weeks")

    result["generated_at"] = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(
        f"[OK] verification.json -> {out_path.relative_to(ROOT)} "
        f"(bss={result.get('bss')}, weeks={len(result.get('weeks', []))})"
    )
    return result


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Export verification.json for frontend accuracy screen")
    ap.add_argument("--pairs", type=Path, default=VERIFY_DIR / "backtest_pairs.csv",
                    help="backtest_pairs.csv หรือ operational_pairs.csv")
    ap.add_argument("--scorecard", type=Path, default=VERIFY_DIR / "scorecard.csv")
    ap.add_argument("--reliability", type=Path, default=VERIFY_DIR / "reliability.csv")
    ap.add_argument("--out", type=Path, default=OUT_JSON)
    a = ap.parse_args()
    try:
        export(a.pairs, a.scorecard, a.reliability, a.out)
    except Exception as exc:
        print(f"[FAIL] export_verification_json: {exc}", file=sys.stderr)
        raise SystemExit(1)
