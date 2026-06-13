"""เทรนโมเดล pooled per-province 'ตัวจบ' ราย lead -> models/heatwave_prov_lead{L}.pkl.

ใช้ขั้นตอนเดียวกับ train_provinces วัดผล (fit_calibrated_model) -> deploy ไม่ drift.
"""
from __future__ import annotations

import pickle
import sys
from datetime import datetime, timezone
from pathlib import Path

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))
from train import fit_calibrated_model
from train_final import PROD_MODEL
from train_provinces import FEATURES_P, LEADS, PRIMARY_TARGET
from build_provinces_dataset import OUT_FILE

MODEL_DIR = ROOT / "models"


def artifact_path(lead: int) -> Path:
    return MODEL_DIR / f"heatwave_prov_lead{lead}.pkl"


def train_final(verbose: bool = True) -> list[Path]:
    df = pd.read_parquet(OUT_FILE)
    MODEL_DIR.mkdir(exist_ok=True)
    written = []
    for lead in LEADS:
        col = f"{PRIMARY_TARGET}_l{lead}"
        sub = df[FEATURES_P + [col]].dropna()
        X = sub[FEATURES_P].to_numpy(float)
        y = sub[col].to_numpy(float)
        fitted = fit_calibrated_model(PROD_MODEL, X, y)
        if fitted is None:
            raise RuntimeError(f"lead {lead}: เทรนไม่ได้ (ข้อมูลสั้น/คลาสเดียว)")
        est, cal = fitted
        art = {"estimator": est, "calibrator": cal, "features": list(FEATURES_P),
               "model_name": f"{PROD_MODEL}_cal", "lead_weeks": lead,
               "base_rate": float(y.mean()), "n_train": int(len(sub)),
               "trained_at": datetime.now(timezone.utc).isoformat()}
        with open(artifact_path(lead), "wb") as fh:
            pickle.dump(art, fh)
        written.append(artifact_path(lead))
        if verbose:
            print(f"  lead {lead}: n={len(sub)}, base_rate={art['base_rate']:.3f} -> {artifact_path(lead).name}")
    return written


if __name__ == "__main__":
    train_final()
