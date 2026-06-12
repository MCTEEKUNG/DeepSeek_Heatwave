"""
เฟส 3 — วินิจฉัย calibration ของความน่าจะเป็น

ส่วน 4a (อ่าน predictions.csv อย่างเดียว): Brier decomposition (Murphy) + ECE/MCE
  ต่อ (target, lead, model) -> อธิบายเชิงปริมาณว่าทำไม "lgbm ดิบ" BSS ติดลบทั้งที่ AUC ดี:
  REL (reliability) สูง = calibrate แย่ ; RES (resolution) ยังดี = แยกแยะได้
  -> ปัญหาอยู่ที่ "ความน่าจะเป็นเฟ้อ" ไม่ใช่ "แยกแยะไม่ได้" -> ต้อง recalibrate

ส่วน 4b (recalibrate เล็กน้อย จาก dataset.csv): เทียบวิธี calibrate ของ lgbm
  raw (ไม่ทำ) vs Platt vs Isotonic — ใช้ "core/calib split แยกตามเวลา" แบบเดียวกับ
  train.fit_predict_calibrated เป๊ะ (กัน leakage) เพื่อเทียบกันยุติธรรม

outputs/analysis/calibration_decomp.csv , calibration_methods.csv

ใช้งาน:
  python calibration.py                 # 4a ทุกโมเดล + 4b (lgbm บน y_rm ทุก lead)
  python calibration.py --targets y_rm y_rm95 y_af   # 4b ครอบทุก target
  python calibration.py test
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from evaluate import brier, brier_skill_score, predict_seasonal_climatology  # noqa: E402
from cv import RollingOriginCV  # noqa: E402
from train import (FEATURES, GAP, N_SPLITS, TEST_SIZE, CALIB_FRAC,  # noqa: E402
                   make_estimator, PlattCalibrator, DATASET)

import io_utils as io  # noqa: E402
from stats import brier_decomposition, ece, mce  # noqa: E402


class IsotonicCalibrator:
    """Isotonic recalibration — interface เดียวกับ train.PlattCalibrator (.fit/.transform).

    out_of_bounds='clip': ความน่าจะเป็นใน test ที่หลุดช่วงของ calib block จะถูก clamp
    (ไม่ใช่ NaN) — จำเป็นเพราะ isotonic ไม่ extrapolate
    """

    def fit(self, p_raw: np.ndarray, y: np.ndarray) -> "IsotonicCalibrator":
        self._iso = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0)
        self._iso.fit(np.asarray(p_raw, dtype=float), np.asarray(y, dtype=float))
        return self

    def transform(self, p_raw: np.ndarray) -> np.ndarray:
        return self._iso.predict(np.asarray(p_raw, dtype=float))


# ----------------------------------------------------- 4a: decomposition

def compute_decomposition(pred: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for t, l, m, g in io.iter_combos(pred):
        y = g["y"].to_numpy(float)
        p = g["p"].to_numpy(float)
        dec = brier_decomposition(y, p)
        rows.append({
            "target": t, "lead": l, "model": m, "n": len(y),
            "REL": dec["REL"], "RES": dec["RES"], "UNC": dec["UNC"], "WBV": dec["WBV"],
            "brier": dec["brier"], "ece": ece(y, p), "mce": mce(y, p),
        })
    return pd.DataFrame(rows).sort_values(["target", "lead", "model"]).reset_index(drop=True)


# ------------------------------------------- 4b: Platt vs Isotonic (lgbm)

def _variants_for_fold(X_tr, y_tr, X_te):
    """train lgbm บน core แล้วคืน prob ของ test 3 แบบ: raw / platt / isotonic.

    ใช้ core/calib split แยกตามเวลา (เว้น GAP) แบบเดียวกับ train.fit_predict_calibrated
    คืน None ถ้า train สั้นไป หรือ calib block มีคลาสเดียว
    """
    n = len(y_tr)
    n_calib = int(n * CALIB_FRAC)
    n_core = n - n_calib - GAP
    if n_core < 300 or n_calib < 100:
        return None
    y_cal = y_tr[n - n_calib:]
    if len(np.unique(y_cal)) < 2:
        return None
    model = make_estimator("lgbm")
    model.fit(X_tr[:n_core], y_tr[:n_core])
    p_cal_raw = model.predict_proba(X_tr[n - n_calib:])[:, 1]
    p_te_raw = model.predict_proba(X_te)[:, 1]
    return {
        "raw": p_te_raw,
        "platt": PlattCalibrator().fit(p_cal_raw, y_cal).transform(p_te_raw),
        "isotonic": IsotonicCalibrator().fit(p_cal_raw, y_cal).transform(p_te_raw),
    }


def compute_calibration_methods(targets, leads) -> pd.DataFrame:
    df = pd.read_csv(DATASET, parse_dates=["date"], index_col="date")
    recipes = ("raw", "platt", "isotonic")
    rows = []
    for target in targets:
        for lead in leads:
            col = f"{target}_l{lead}"
            sub = df[FEATURES + ["doy", col]].dropna().sort_index()
            X = sub[FEATURES].to_numpy(float)
            y = sub[col].to_numpy(float)
            doy = sub["doy"].to_numpy(int)
            cv = RollingOriginCV(n_splits=N_SPLITS, test_size=TEST_SIZE, gap=GAP, expanding=True)
            pool = {r: {"y": [], "p": [], "pc": []} for r in recipes}
            for tr, te in cv.split(len(sub)):
                if len(np.unique(y[tr])) < 2:
                    continue
                var = _variants_for_fold(X[tr], y[tr], X[te])
                if var is None:
                    continue
                pc = predict_seasonal_climatology(doy[tr], y[tr], doy[te])
                for r in recipes:
                    pool[r]["y"].append(y[te])
                    pool[r]["p"].append(var[r])
                    pool[r]["pc"].append(pc)
            for r in recipes:
                if not pool[r]["y"]:
                    continue
                yy = np.concatenate(pool[r]["y"])
                pp = np.concatenate(pool[r]["p"])
                pcc = np.concatenate(pool[r]["pc"])
                rows.append({
                    "target": target, "lead": lead, "recipe": r, "n": int(yy.size),
                    "brier": brier(yy, pp),
                    "bss": brier_skill_score(yy, pp, baseline_prob=pcc),
                    "ece": ece(yy, pp), "mce": mce(yy, pp),
                })
    return pd.DataFrame(rows)


def _print_summary(decomp, methods) -> None:
    t = io.PRIMARY_TARGET
    print(f"\n=== Brier decomposition — {t}, lead 2 (REL ต่ำ=calibrate ดี, RES สูง=แยกแยะดี) ===")
    d = decomp[(decomp.target == t) & (decomp.lead == 2)].set_index("model")
    print(d[["REL", "RES", "UNC", "brier", "ece"]].round(4).to_string())
    print("\n  -> เทียบ 'lgbm' (ดิบ) กับ 'lgbm_cal': REL ควรลดลงมากหลัง recalibrate")

    if not methods.empty:
        print(f"\n=== วิธี calibrate lgbm — {t} (BSS สูง=ดี, ECE ต่ำ=ดี) ===")
        mt = methods[methods.target == t].pivot_table(index="recipe", columns="lead",
                                                       values="bss")
        print("BSS:")
        print(mt.round(3).to_string())


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="calibration diagnostics (4a decomp + 4b methods)")
    ap.add_argument("--targets", nargs="*", default=[io.PRIMARY_TARGET])
    ap.add_argument("--leads", nargs="*", type=int, default=io.LEADS)
    ap.add_argument("--skip-methods", action="store_true", help="ทำเฉพาะ 4a (อ่านอย่างเดียว)")
    args = ap.parse_args(argv)

    io.ensure_dirs()
    pred = io.load_predictions()
    print(f"=== calibration: predictions {len(pred)} แถว ===")

    decomp = compute_decomposition(pred)
    decomp.to_csv(io.ANALYSIS_DIR / "calibration_decomp.csv", index=False)
    print(f"[OK] calibration_decomp.csv : {len(decomp)} แถว")

    methods = pd.DataFrame()
    if not args.skip_methods:
        print(f"[..] 4b: เทรน lgbm ใหม่ + เทียบ raw/platt/isotonic "
              f"(targets={args.targets}, leads={args.leads}) ...", flush=True)
        methods = compute_calibration_methods(args.targets, args.leads)
        methods.to_csv(io.ANALYSIS_DIR / "calibration_methods.csv", index=False)
        print(f"[OK] calibration_methods.csv : {len(methods)} แถว")

    _print_summary(decomp, methods)
    print(f"\n[OK] ผลอยู่ที่ {io.ANALYSIS_DIR}")
    return 0


# ---------------------------------------------------------------- self-test

def _selftest() -> None:
    rng = np.random.default_rng(2)

    # 1) IsotonicCalibrator: monotonic, อยู่ใน [0,1], clip นอกช่วง
    praw = rng.random(500)
    yy = (rng.random(500) < praw).astype(float)   # y สัมพันธ์กับ praw
    cal = IsotonicCalibrator().fit(praw, yy)
    grid = np.linspace(0, 1, 50)
    out = cal.transform(grid)
    assert np.all(np.diff(out) >= -1e-9), "isotonic ต้อง monotonic ไม่ลด"
    assert out.min() >= 0 and out.max() <= 1
    assert np.isfinite(cal.transform(np.array([5.0, -3.0]))).all(), "นอกช่วงต้องถูก clip ไม่ใช่ NaN"
    print("[OK] IsotonicCalibrator: monotonic, [0,1], clip นอกช่วง")

    # 2) decomposition table: คอลัมน์ครบ + REL ของ forecast เฟ้อ > REL ของ calibrate ดี
    n = 3000
    y = (rng.random(n) < 0.2).astype(float)
    p_good = np.clip(0.2 + 0.02 * rng.standard_normal(n), 0.01, 0.99)
    p_inflated = np.clip(0.6 + 0.02 * rng.standard_normal(n), 0.01, 0.99)   # เฟ้อสูง
    pred = pd.concat([
        pd.DataFrame({"target": "y_rm", "lead": 2, "fold": 1, "model": "good",
                      "date": pd.date_range("2015-01-01", periods=n, freq="h"),
                      "y": y, "p": p_good, "p_clim": 0.2}),
        pd.DataFrame({"target": "y_rm", "lead": 2, "fold": 1, "model": "inflated",
                      "date": pd.date_range("2015-01-01", periods=n, freq="h"),
                      "y": y, "p": p_inflated, "p_clim": 0.2}),
    ], ignore_index=True)
    dec = compute_decomposition(pred).set_index("model")
    assert {"REL", "RES", "UNC", "WBV", "ece"} <= set(dec.columns)
    assert dec.loc["inflated", "REL"] > dec.loc["good", "REL"], dec[["REL"]]
    assert dec.loc["inflated", "ece"] > dec.loc["good", "ece"]

    # 3) 4b machinery: _variants_for_fold คืน 3 แบบ prob อยู่ใน [0,1]
    n_tr, n_te, d = 700, 120, len(FEATURES)
    Xtr = rng.standard_normal((n_tr, d))
    ytr = (rng.random(n_tr) < 1 / (1 + np.exp(-Xtr[:, 0]))).astype(float)
    Xte = rng.standard_normal((n_te, d))
    var = _variants_for_fold(Xtr, ytr, Xte)
    assert var is not None and set(var) == {"raw", "platt", "isotonic"}
    for r, pp in var.items():
        assert pp.shape == (n_te,) and pp.min() >= 0 and pp.max() <= 1, r
    print(f"[OK] decomposition: REL(เฟ้อ)={dec.loc['inflated','REL']:.3f} > "
          f"REL(ดี)={dec.loc['good','REL']:.3f} ; 4b คืน raw/platt/isotonic ใน [0,1]")
    print("[OK] calibration self-test ผ่าน")


if __name__ == "__main__":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass
    if len(sys.argv) > 1 and sys.argv[1] == "test":
        _selftest()
    else:
        sys.exit(main())
