"""ออก forecast รายจังหวัด (lead 2-6) ของวันล่าสุดที่ feature ครบ -> docs/forecast_provinces.json."""
from __future__ import annotations

import json
import pickle
import sys
from datetime import datetime, timezone
from pathlib import Path

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))
from build_provinces_dataset import build_provinces_features
from train_provinces import LEADS, FEATURES_P
from train_final_provinces import artifact_path
from predict import risk_level
from province_grid import load_provinces

OUT_FILE = ROOT / "docs" / "forecast_provinces.json"
SKILL_CSV = ROOT / "outputs" / "analysis" / "provinces_per_province_bss.csv"


def _load_arts():
    arts = {}
    for L in LEADS:
        with open(artifact_path(L), "rb") as fh:
            arts[L] = pickle.load(fh)
    return arts


def build_forecast() -> dict:
    feat_all, _hw = build_provinces_features(verbose=False)
    arts = _load_arts()
    pv = load_provinces().set_index("id")
    skill = pd.read_csv(SKILL_CSV) if SKILL_CSV.exists() else pd.DataFrame()
    provinces_out = []
    for pid, g in feat_all.groupby("province_id"):
        valid = g.dropna(subset=FEATURES_P)
        if valid.empty:
            continue
        row = valid.sort_values("date").iloc[-1]
        X = row[FEATURES_P].to_numpy(float).reshape(1, -1)
        info = pv.loc[pid]
        fcs = []
        for L in LEADS:
            a = arts[L]
            p = float(a["calibrator"].transform(a["estimator"].predict_proba(X)[:, 1])[0])
            th, en, ratio = risk_level(p, a["base_rate"])
            fcs.append({"lead_weeks": L, "probability": round(p, 4),
                        "climatology_base_rate": round(a["base_rate"], 4),
                        "ratio_vs_normal": round(ratio, 2),
                        "risk_level_th": th, "risk_level_en": en})
        provinces_out.append({"id": int(pid), "code": info["code"],
                              "name_th": info["name_th"], "name_en": info["name_en"],
                              "region": info["region"], "lat": float(info["lat"]),
                              "lon": float(info["lon"]),
                              "issue_date": str(pd.Timestamp(row["date"]).date()),
                              "forecasts": fcs})
    out = {"schema_version": 1, "model": arts[LEADS[0]]["model_name"],
           "generated_at": datetime.now(timezone.utc).isoformat(),
           "n_provinces": len(provinces_out), "provinces": provinces_out}
    if not skill.empty:
        out["skill"] = skill.to_dict(orient="records")
    return out


def _selftest() -> None:
    """parity: feature รายจังหวัดที่ build_provinces_features สร้าง = ที่อยู่ใน dataset_provinces.parquet."""
    from build_provinces_dataset import OUT_FILE as DS
    if not DS.exists():
        print("[ข้าม] ไม่มี dataset_provinces.parquet — รัน build ก่อน"); return
    feat_all, _ = build_provinces_features(verbose=False)
    ds = pd.read_parquet(DS)
    key = ["province_id", "date"]
    m = feat_all.merge(ds, on=key, suffixes=("_a", "_b"))
    bad = 0
    for c in FEATURES_P:
        a = m[f"{c}_a"].to_numpy(float); b = m[f"{c}_b"].to_numpy(float)
        ok = np.isclose(a, b, rtol=1e-6, atol=1e-8) | (np.isnan(a) & np.isnan(b))
        bad += int((~ok).sum())
    assert bad == 0, f"parity ไม่ตรง {bad} จุด"
    print(f"[OK] train/serve parity: {len(FEATURES_P)} feature x {len(m)} แถว ตรง dataset")
    print("[OK] self-test ผ่าน")


def predict(verbose: bool = True) -> dict:
    fc = build_forecast()
    OUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    OUT_FILE.write_text(json.dumps(fc, ensure_ascii=False, indent=2), encoding="utf-8")
    if verbose:
        print(f"[OK] {OUT_FILE} | {fc['n_provinces']} จังหวัด | model {fc['model']}")
    return fc


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "test":
        _selftest()
    else:
        predict()
