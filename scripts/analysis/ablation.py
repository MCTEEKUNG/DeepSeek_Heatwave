"""
เฟส 5 — Feature-group ablation (วัดส่วนร่วมของ "กลุ่มฟีเจอร์" ต่อ BSS)

reuse train.run_target_lead ซ้ำ (ไม่ก๊อป CV loop) ผ่านพารามิเตอร์ใหม่:
  features=<subset>  เลือกคอลัมน์ที่ป้อนโมเดล (dropna บน FEATURES ครบ -> แถวเดียวกันทุก run)
  models=<subset>    เทรนเฉพาะโมเดลที่ต้องรายงาน (เร่งความเร็ว)

การทดลอง:
  full                         ฟีเจอร์ครบ (ต้อง reproduce metrics_pooled.csv)
  drop_ENSO / SOIL / MJO / THERMAL   leave-one-group-out -> ส่วนร่วม "ส่วนเพิ่ม" ของกลุ่ม
  soil_only / enso_only / soil_enso  additive -> กลุ่มเดียว/สองกลุ่มไปได้ไกลแค่ไหน
  delta_bss_vs_full = bss(set) - bss(full) ; ค่าติดลบของ drop_X = กลุ่ม X สำคัญ

outputs/analysis/ablation.csv

ใช้งาน:
  python ablation.py --leads 2 --models logistic     # ทดสอบเร็ว (วินาที)
  python ablation.py                                  # เต็ม: y_rm ทุก lead, logistic+lgbm_cal
  python ablation.py test
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from train import FEATURES, run_target_lead, pooled_metrics, DATASET  # noqa: E402

import io_utils as io  # noqa: E402

# กลุ่มฟีเจอร์ (รวมกันต้องเท่ากับ FEATURES ครบ 20 ตัว)
GROUPS = {
    "ENSO": ["nino34_lag1m"],
    "SOIL": ["sm1", "sm1_mean7", "sm1_mean30", "sm1_trend",
             "sm3", "sm3_mean7", "sm3_mean30", "sm3_trend"],
    "MJO": ["mjo_rmm1", "mjo_rmm2", "mjo_amp", "mjo_sin", "mjo_cos"],
    "THERMAL": ["tmax_rm", "tmax_mean7", "in_hw_today", "hot_frac7"],
    "SEASONAL": ["doy_sin", "doy_cos"],
}
LOO_GROUPS = ["ENSO", "SOIL", "MJO", "THERMAL"]   # leave-one-out (คง SEASONAL เป็น control)


def feature_sets() -> dict[str, list[str]]:
    assert set().union(*GROUPS.values()) == set(FEATURES), "กลุ่มฟีเจอร์ไม่ครอบคลุม FEATURES"
    sets = {"full": list(FEATURES)}
    for g in LOO_GROUPS:
        drop = set(GROUPS[g])
        sets[f"drop_{g}"] = [f for f in FEATURES if f not in drop]
    sets["soil_only"] = list(GROUPS["SOIL"])
    sets["enso_only"] = list(GROUPS["ENSO"])
    sets["soil_enso"] = GROUPS["SOIL"] + GROUPS["ENSO"]
    return sets


def run_ablation(df, targets, leads, models, sets) -> pd.DataFrame:
    rows = []
    for set_name, feats in sets.items():
        for target in targets:
            for lead in leads:
                _, pred = run_target_lead(df, target, lead, verbose=False,
                                          features=feats, models=models)
                pooled = pooled_metrics(pred)
                for m in models:
                    r = pooled[(pooled.target == target) & (pooled.lead == lead)
                               & (pooled.model == m)]
                    if r.empty:
                        continue
                    row = r.iloc[0]
                    rows.append({
                        "target": target, "lead": lead, "model": m, "feature_set": set_name,
                        "n_features": len(feats), "n": int(row["n"]),
                        "base_rate": row["base_rate"], "brier": row["brier"],
                        "bss": row["bss"], "auc": row["auc"],
                    })
    out = pd.DataFrame(rows)
    full = (out[out.feature_set == "full"][["target", "lead", "model", "bss"]]
            .rename(columns={"bss": "bss_full"}))
    out = out.merge(full, on=["target", "lead", "model"], how="left")
    out["delta_bss_vs_full"] = out["bss"] - out["bss_full"]
    return out.drop(columns=["bss_full"]).sort_values(
        ["target", "lead", "model", "feature_set"]).reset_index(drop=True)


def _verify_full_matches_pooled(out) -> None:
    """ตรวจว่า feature_set=full reproduce ตัวเลขใน metrics_pooled.csv เดิม (กันเพี้ยน)."""
    if not io.POOLED_FILE.exists():
        print("[skip] ไม่พบ metrics_pooled.csv — ข้ามการตรวจ full")
        return
    pooled = io.load_pooled()
    full = out[out.feature_set == "full"]
    n_ok = 0
    for _, r in full.iterrows():
        ref = pooled[(pooled.target == r.target) & (pooled.lead == r.lead)
                     & (pooled.model == r.model)]
        if ref.empty:
            continue
        assert abs(float(ref.iloc[0]["bss"]) - r.bss) < 1e-6, (r.target, r.lead, r.model,
                                                               ref.iloc[0]["bss"], r.bss)
        n_ok += 1
    print(f"[OK] full reproduce metrics_pooled.csv ครบ {n_ok} ช่อง (ตรงเป๊ะ)")


def _print_summary(out) -> None:
    t = io.PRIMARY_TARGET
    print(f"\n=== delta BSS เทียบ full — {t} (ค่าติดลบของ drop_X = กลุ่ม X สำคัญ) ===")
    for m in sorted(out["model"].unique()):
        sub = out[(out.target == t) & (out.model == m)
                  & (out.feature_set.str.startswith("drop_"))]
        if sub.empty:
            continue
        piv = sub.pivot_table(index="feature_set", columns="lead", values="delta_bss_vs_full")
        print(f"\n[{m}] leave-one-group-out:")
        print(piv.round(3).to_string())


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="feature-group ablation (reuse train.run_target_lead)")
    ap.add_argument("--targets", nargs="*", default=[io.PRIMARY_TARGET])
    ap.add_argument("--leads", nargs="*", type=int, default=io.LEADS)
    ap.add_argument("--models", nargs="*", default=["logistic", "lgbm_cal"])
    ap.add_argument("--sets", nargs="*", default=None, help="จำกัดชุดฟีเจอร์ (ค่าเริ่มต้น=ทั้งหมด)")
    args = ap.parse_args(argv)

    io.ensure_dirs()
    df = pd.read_csv(DATASET, parse_dates=["date"], index_col="date")
    sets = feature_sets()
    if args.sets:
        sets = {k: v for k, v in sets.items() if k in args.sets}
        if "full" not in sets:                      # ต้องมี full ไว้คำนวณ delta
            sets = {"full": list(FEATURES), **sets}
    print(f"=== ablation: dataset {len(df)} วัน | sets={list(sets)} | "
          f"targets={args.targets} leads={args.leads} models={args.models} ===", flush=True)

    out = run_ablation(df, args.targets, args.leads, args.models, sets)
    out.to_csv(io.ANALYSIS_DIR / "ablation.csv", index=False)
    print(f"[OK] ablation.csv : {len(out)} แถว")

    _verify_full_matches_pooled(out)
    _print_summary(out)
    print(f"\n[OK] ผลอยู่ที่ {io.ANALYSIS_DIR}")
    return 0


# ---------------------------------------------------------------- self-test

def _selftest() -> None:
    # 1) feature_sets ถูกต้อง: full = 20, drop_ENSO ตัด nino34_lag1m
    sets = feature_sets()
    assert len(sets["full"]) == 20
    assert "nino34_lag1m" not in sets["drop_ENSO"] and len(sets["drop_ENSO"]) == 19
    assert sets["enso_only"] == ["nino34_lag1m"]

    # 2) ablation จริง: y ขึ้นกับ nino34_lag1m -> drop_ENSO ต้องทำ BSS ตก (delta < 0)
    rng = np.random.default_rng(4)
    n = 1650  # พอสำหรับ CV จริง (5 splits x 300 + gap)
    idx = pd.date_range("2008-01-01", periods=n, freq="D")
    df = pd.DataFrame(index=idx)
    for c in FEATURES:
        df[c] = rng.standard_normal(n)
    df["doy"] = idx.dayofyear
    logit = 2.0 * df["nino34_lag1m"].to_numpy() + 1.0 * df["sm1_mean30"].to_numpy()
    p_true = 1 / (1 + np.exp(-logit))
    for L in io.LEADS:
        df[f"y_rm_l{L}"] = (rng.random(n) < p_true).astype(float)
    df.index.name = "date"

    out = run_ablation(df, ["y_rm"], [2], ["logistic"], {k: sets[k] for k in ("full", "drop_ENSO")})
    d = out[(out.feature_set == "drop_ENSO") & (out.model == "logistic")].iloc[0]
    assert d["delta_bss_vs_full"] < 0, f"ตัด ENSO ควรทำ BSS ตก ได้ {d['delta_bss_vs_full']}"
    assert "delta_bss_vs_full" in out.columns
    print(f"[OK] selftest: drop_ENSO -> delta BSS = {d['delta_bss_vs_full']:+.3f} (<0 ตามคาด)")
    print("[OK] ablation self-test ผ่าน")


if __name__ == "__main__":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass
    if len(sys.argv) > 1 and sys.argv[1] == "test":
        _selftest()
    else:
        sys.exit(main())
