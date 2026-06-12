"""
เฟส 2 — แยกสกิลตาม fold และตาม ENSO regime (อ่าน predictions.csv + nino34.csv อย่างเดียว)

ตอบ: "โมเดลทำงานดี/แย่เมื่อไร" และอธิบายว่าทำไม fold/ปี El Niño ถึงดูอ่อน

ทำอะไร:
  1) per-fold : เมตริก (Brier/BSS/AUC เทียบ p_clim) ราย fold + ช่วงวันที่ + บริบท ENSO ของ fold
  2) per-ENSO regime : แบ่งวันออกพยากรณ์เป็น elnino/lanina/neutral แล้ววัดสกิลในแต่ละกลุ่ม
     - stratify ด้วย nino34_lag1m เป็นหลัก (ค่าที่โมเดล "เห็นจริง") + concurrent เป็น robustness
     - baseline ของ BSS = seasonal climatology (p_clim) "ภายในกลุ่มนั้น" (ยุติธรรม)
     - guard: กลุ่มที่ n < 50 ทำเครื่องหมาย reliable=False (ตัวเลขไม่เสถียร อย่าอ้างอิง)

ประเด็นสำคัญที่ต้องอภิปราย: heatwave "ถี่ขึ้น" ในปี El Niño -> base rate สูงขึ้น ->
climatology แข็งขึ้น -> BSS อาจตก แม้ Brier ใกล้เดิม (เทียบ brier กับ brier_clim ในตารางได้)

outputs/analysis/regime_by_fold.csv , regime_by_enso.csv

ใช้งาน:  python regime_strat.py        |  python regime_strat.py test
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from evaluate import evaluate_probabilistic, brier  # noqa: E402

import io_utils as io  # noqa: E402

MIN_STRATUM = 50   # ขั้นต่ำของจำนวนตัวอย่างต่อกลุ่มก่อนจะเชื่อ BSS/AUC


def per_fold_table(pred: pd.DataFrame) -> pd.DataFrame:
    """เมตริกราย (target, lead, model, fold) + ช่วงวันที่ + บริบท ENSO ของ fold."""
    p = io.attach_regime(pred, use="lag1m")
    rows = []
    keys = ["target", "lead", "model", "fold"]
    for (t, l, m, fold), g in p.groupby(keys):
        y = g["y"].to_numpy(float)
        met = evaluate_probabilistic(y, g["p"].to_numpy(float),
                                     baseline_prob=g["p_clim"].to_numpy(float))
        reg = g["regime"].value_counts(normalize=True)
        rows.append({
            "target": t, "lead": int(l), "model": m, "fold": int(fold),
            "date_min": g["date"].min().date(), "date_max": g["date"].max().date(),
            "n": met["n"], "base_rate": met["base_rate"],
            "brier": met["brier"], "bss": met["bss"], "auc": met["auc"],
            "mean_enso_anom": float(g["enso_anom"].mean()),
            "frac_elnino": float(reg.get("elnino", 0.0)),
            "frac_lanina": float(reg.get("lanina", 0.0)),
            "dominant_regime": reg.idxmax(),
        })
    return pd.DataFrame(rows).sort_values(["target", "lead", "model", "fold"]).reset_index(drop=True)


def per_regime_table(pred: pd.DataFrame, use: str = "lag1m") -> pd.DataFrame:
    """เมตริกราย (target, lead, model, regime) — baseline = p_clim ภายในกลุ่ม."""
    p = io.attach_regime(pred, use=use)
    rows = []
    for (t, l, m, reg), g in p.groupby(["target", "lead", "model", "regime"]):
        y = g["y"].to_numpy(float)
        pc = g["p_clim"].to_numpy(float)
        met = evaluate_probabilistic(y, g["p"].to_numpy(float), baseline_prob=pc)
        n = met["n"]
        reliable = (n >= MIN_STRATUM) and (len(np.unique(y)) >= 2)
        rows.append({
            "target": t, "lead": int(l), "model": m, "regime": reg, "use": use,
            "n": n, "base_rate": met["base_rate"],
            "brier": met["brier"], "brier_clim": float(brier(y, pc)),
            "bss": met["bss"] if reliable else float("nan"),
            "auc": met["auc"] if reliable else float("nan"),
            "reliable": reliable,
        })
    return pd.DataFrame(rows)


def _print_summary(fold_df, regime_df) -> None:
    t = io.PRIMARY_TARGET
    print(f"\n=== base rate ของ {t} แยกตาม ENSO regime (lag1m) — เห็น base-rate shift ===")
    br = (regime_df[(regime_df.target == t) & (regime_df.use == "lag1m")
                    & (regime_df.model == "climatology")]
          .pivot_table(index="regime", columns="lead", values="base_rate"))
    print(br.round(3).to_string())

    print(f"\n=== BSS ของ {t} แยกตาม regime (lag1m, เทียบ seasonal clim ในกลุ่ม) ===")
    for m in io.REPORT_MODELS:
        sub = regime_df[(regime_df.target == t) & (regime_df.use == "lag1m")
                        & (regime_df.model == m)]
        if sub.empty:
            continue
        piv = sub.pivot_table(index="regime", columns="lead", values="bss")
        print(f"\n[{m}]")
        print(piv.round(3).to_string())
    print("\n(NaN = กลุ่มที่ n < 50 หรือไม่มีทั้งสองคลาส -> ไม่อ้างอิง)")


def main() -> int:
    io.ensure_dirs()
    pred = io.load_predictions()
    print(f"=== regime_strat: predictions {len(pred)} แถว ===")

    fold_df = per_fold_table(pred)
    fold_df.to_csv(io.ANALYSIS_DIR / "regime_by_fold.csv", index=False)
    print(f"[OK] regime_by_fold.csv : {len(fold_df)} แถว")

    reg_lag = per_regime_table(pred, use="lag1m")
    reg_con = per_regime_table(pred, use="concurrent")
    regime_df = pd.concat([reg_lag, reg_con], ignore_index=True).sort_values(
        ["use", "target", "lead", "model", "regime"]).reset_index(drop=True)
    regime_df.to_csv(io.ANALYSIS_DIR / "regime_by_enso.csv", index=False)
    print(f"[OK] regime_by_enso.csv : {len(regime_df)} แถว")

    _print_summary(fold_df, regime_df)
    print(f"\n[OK] ผลอยู่ที่ {io.ANALYSIS_DIR}")
    return 0


# ---------------------------------------------------------------- self-test

def _selftest() -> None:
    # สร้าง predictions สังเคราะห์คร่อมช่วง El Niño (2015-2016) และ La Niña (2010-2011)
    rng = np.random.default_rng(1)
    frames = []
    spans = {1: "2010-09-01", 2: "2015-09-01"}  # fold1 ~ลานีญา, fold2 ~เอลนีโญ
    for fold, start in spans.items():
        dates = pd.date_range(start, periods=120, freq="D")
        y = (rng.random(120) < 0.25).astype(float)
        p_clim = np.full(120, 0.2)
        p = np.clip(0.2 + 0.1 * (y - 0.25) + 0.05 * rng.standard_normal(120), 0.01, 0.99)
        for model, pp in (("climatology", p_clim), ("logistic", p)):
            frames.append(pd.DataFrame({"target": "y_rm", "lead": 2, "fold": fold,
                                        "model": model, "date": dates, "y": y, "p": pp,
                                        "p_clim": p_clim}))
    pred = pd.concat(frames, ignore_index=True)

    fold_df = per_fold_table(pred)
    regime_df = per_regime_table(pred, use="lag1m")

    # 1) per-fold: คอลัมน์ครบ + bss ของ climatology = 0
    assert {"date_min", "date_max", "dominant_regime", "mean_enso_anom"} <= set(fold_df.columns)
    cb = fold_df[(fold_df.model == "climatology")]
    assert (cb["bss"].abs() < 1e-9).all(), "climatology เทียบตัวเอง BSS ต้อง 0"

    # 2) regime ถูกจำแนก: fold2 (ก.ย.2015) ต้องโดน elnino, fold1 (ก.ย.2010) ต้อง lanina
    dom = dict(zip(fold_df[fold_df.model == "climatology"]["fold"],
                   fold_df[fold_df.model == "climatology"]["dominant_regime"]))
    assert dom[2] == "elnino", dom
    assert dom[1] == "lanina", dom

    # 3) per-regime: มีคอลัมน์ reliable + brier_clim ; กลุ่มเล็ก -> bss = nan
    assert {"reliable", "brier_clim", "use"} <= set(regime_df.columns)
    assert regime_df.loc[~regime_df["reliable"], "bss"].isna().all()
    print(f"[OK] selftest: fold2->{dom[2]}, fold1->{dom[1]} ; climatology BSS=0 ; "
          f"กลุ่ม n<{MIN_STRATUM} -> bss=NaN")
    print("[OK] regime_strat self-test ผ่าน")


if __name__ == "__main__":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass
    if len(sys.argv) > 1 and sys.argv[1] == "test":
        _selftest()
    else:
        sys.exit(main())
