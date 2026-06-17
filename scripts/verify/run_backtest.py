"""Retrospective backtest driver: run frozen models over 2024-2025 ERA5."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent.parent  # scripts/verify/ -> repo root
sys.path.insert(0, str(ROOT / "scripts"))

from verify.observed_labels import build_labeled_frame, FEATURES_P
from build_provinces_dataset import _load_frozen_climatology
from train_provinces import LEADS
from train_final_provinces import artifact_path
# NOTE: import _load_arts from predict_provinces would pull in too many dependencies
# Define it locally:
import pickle

TMAX_DIR = ROOT / "data" / "raw_backtest" / "tmax_thailand"
SOIL_DIR = ROOT / "data" / "raw_backtest" / "soil_moisture_thailand"
OUT_FILE = ROOT / "outputs" / "operational" / "verification" / "backtest_pairs.csv"


def _load_arts() -> dict:
    # pickle is safe here: artifact_path() resolves to models/ inside this repo,
    # written exclusively by train_final_provinces.py from our own training pipeline.
    # No external / user-supplied pickle files are ever loaded by this function.
    arts = {}
    for L in LEADS:
        with open(artifact_path(L), "rb") as fh:
            arts[L] = pickle.load(fh)
    return arts


def run(verbose: bool = True) -> pd.DataFrame:
    clim = _load_frozen_climatology()
    frozen_thr90 = clim["thr90_grid"]
    prov_base = clim["base_rate"]    # {pid: {lead: float}}

    if verbose:
        print(f"[backtest] โหลด ERA5 จาก {TMAX_DIR} ...")
    labeled = build_labeled_frame(TMAX_DIR, SOIL_DIR, frozen_thr90=frozen_thr90, verbose=verbose)

    arts = _load_arts()
    if verbose:
        print(f"[backtest] โหลดโมเดล lead {LEADS} OK")

    rows = []
    for pid, g in labeled.groupby("province_id"):
        valid = g.dropna(subset=FEATURES_P)   # only rows with complete features
        for _, row in valid.iterrows():
            X = row[FEATURES_P].to_numpy(float).reshape(1, -1)
            issue_date = str(pd.Timestamp(row["date"]).date())
            for L in LEADS:
                y_obs = row.get(f"y_rm_l{L}", float("nan"))
                if pd.isna(y_obs):
                    continue   # skip open / incomplete windows
                a = arts[L]
                p = float(a["calibrator"].transform(a["estimator"].predict_proba(X)[:, 1])[0])
                try:
                    br = float(prov_base[int(pid)][int(L)])
                except KeyError:
                    raise KeyError(f"base_rate ไม่มีจังหวัด {int(pid)} lead {int(L)}")
                rows.append({
                    "issue_date": issue_date,
                    "province_id": int(pid),
                    "lead": int(L),
                    "probability": round(p, 6),
                    "base_rate": round(br, 6),
                    "y_obs": float(y_obs),
                })

    pairs = pd.DataFrame(rows, columns=["issue_date", "province_id", "lead",
                                         "probability", "base_rate", "y_obs"])
    OUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    pairs.to_csv(OUT_FILE, index=False)
    if verbose:
        print(f"[backtest] เขียน {len(pairs)} แถว -> {OUT_FILE}")
        print(f"  lead distribution: {pairs.groupby('lead').size().to_dict()}")
        print(f"  base_rate range: y_obs mean={pairs['y_obs'].mean():.3f}")
    return pairs


if __name__ == "__main__":
    run()
