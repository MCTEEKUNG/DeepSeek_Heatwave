"""
เฟส 4 — Permutation importance (วัดความสำคัญฟีเจอร์แบบ "ตรงกับสกิลที่วัดผล")

ทำไมแทน split-gain ของ lgbm (outputs/figures/feature_importance.csv):
  - split-gain เอนเอียงเข้าหาฟีเจอร์ที่ค่าต่อเนื่อง/cardinality สูง และวัดบน "train"
  - permutation วัดบน "test" และให้คะแนนด้วย BSS โดยตรง (ภาษาเดียวกับการประเมินผล)
    -> สลับคอลัมน์ฟีเจอร์ใน test แล้วดู BSS ตกเท่าไร = ความสำคัญต่อสกิลจริง

วิธี: ต่อ fold (RollingOriginCV) -> fit บน train, วัด BSS ฐานบน test (เทียบ seasonal clim ของ fold)
      -> สลับทีละคอลัมน์ใน test (n_repeats ครั้ง) วัด BSS ใหม่ -> drop = bss_base - bss_perm
      -> เฉลี่ยข้าม fold. drop สูง = ฟีเจอร์สำคัญ

ข้อควรระวัง: ฟีเจอร์ correlated (เช่น sm1_mean7 vs sm1_mean30) จะ "แชร์" ความสำคัญกัน
-> permute ทีละตัวจะ "ประเมินต่ำ" ทั้งคู่ ; ใช้ ablation ราย "กลุ่ม" (เฟส 5) เสริมการระบุกลุ่ม

outputs/analysis/permutation_importance.csv

ใช้งาน:
  python permutation_importance.py                       # lgbm+logistic, y_rm, lead 2
  python permutation_importance.py --models lgbm --leads 2 3
  python permutation_importance.py test
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from evaluate import brier_skill_score, predict_seasonal_climatology  # noqa: E402
from cv import RollingOriginCV  # noqa: E402
from train import FEATURES, GAP, N_SPLITS, TEST_SIZE, make_estimator, DATASET  # noqa: E402

import io_utils as io  # noqa: E402
from stats import SEED  # noqa: E402


def permutation_importance(df, target, lead, model_name, n_repeats=5, seed=SEED,
                           n_splits=N_SPLITS, test_size=TEST_SIZE, gap=GAP,
                           features=FEATURES) -> pd.DataFrame:
    """drop ของ BSS เมื่อสลับทีละฟีเจอร์ใน test (เฉลี่ยข้าม fold)."""
    col = f"{target}_l{lead}"
    sub = df[list(features) + ["doy", col]].dropna().sort_index()
    X = sub[features].to_numpy(float)
    y = sub[col].to_numpy(float)
    doy = sub["doy"].to_numpy(int)
    rng = np.random.default_rng(seed)
    cv = RollingOriginCV(n_splits=n_splits, test_size=test_size, gap=gap, expanding=True)

    fold_drops = []
    for tr, te in cv.split(len(sub)):
        if len(np.unique(y[tr])) < 2:
            continue
        est = make_estimator(model_name)
        est.fit(X[tr], y[tr])
        pc = predict_seasonal_climatology(doy[tr], y[tr], doy[te])
        bss_base = brier_skill_score(y[te], est.predict_proba(X[te])[:, 1], baseline_prob=pc)
        Xte = X[te]
        drops = np.zeros(len(features))
        for j in range(len(features)):
            acc = 0.0
            for _ in range(n_repeats):
                Xp = Xte.copy()
                Xp[:, j] = rng.permutation(Xp[:, j])
                acc += brier_skill_score(y[te], est.predict_proba(Xp)[:, 1], baseline_prob=pc)
            drops[j] = bss_base - acc / n_repeats
        fold_drops.append(drops)

    M = np.array(fold_drops)
    out = pd.DataFrame({
        "feature": list(features), "target": target, "lead": lead, "model": model_name,
        "mean_drop_bss": M.mean(axis=0), "std_drop_bss": M.std(axis=0), "n_folds": len(M),
    }).sort_values("mean_drop_bss", ascending=False).reset_index(drop=True)
    out["rank"] = np.arange(1, len(out) + 1)
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="permutation importance (BSS-based, on test)")
    ap.add_argument("--models", nargs="*", default=["lgbm", "logistic"])
    ap.add_argument("--targets", nargs="*", default=[io.PRIMARY_TARGET])
    ap.add_argument("--leads", nargs="*", type=int, default=[2])
    ap.add_argument("--n-repeats", type=int, default=5)
    args = ap.parse_args(argv)

    io.ensure_dirs()
    df = pd.read_csv(DATASET, parse_dates=["date"], index_col="date")
    print(f"=== permutation_importance: dataset {len(df)} วัน | models={args.models} "
          f"targets={args.targets} leads={args.leads} ===", flush=True)

    tables = []
    for target in args.targets:
        for lead in args.leads:
            for m in args.models:
                print(f"[..] {m} {target} lead {lead} ...", flush=True)
                tables.append(permutation_importance(df, target, lead, m, n_repeats=args.n_repeats))
    out = pd.concat(tables, ignore_index=True)
    out.to_csv(io.ANALYSIS_DIR / "permutation_importance.csv", index=False)
    print(f"[OK] permutation_importance.csv : {len(out)} แถว")

    for (t, l, m), g in out.groupby(["target", "lead", "model"]):
        print(f"\n=== top features: {m}, {t} lead {l} (BSS drop เมื่อสลับ) ===")
        print(g.head(6)[["rank", "feature", "mean_drop_bss", "std_drop_bss"]]
              .to_string(index=False))
    print(f"\n[OK] ผลอยู่ที่ {io.ANALYSIS_DIR}")
    return 0


# ---------------------------------------------------------------- self-test

def _selftest() -> None:
    rng = np.random.default_rng(3)
    n = 400
    idx = pd.date_range("2015-01-01", periods=n, freq="D")
    df = pd.DataFrame(index=idx)
    for c in FEATURES:
        df[c] = rng.standard_normal(n)
    df["doy"] = idx.dayofyear
    # y ขึ้นกับ sm1_mean30 เป็นหลัก -> ฟีเจอร์นี้ต้องสำคัญสุด
    driver = "sm1_mean30"
    logit = 2.5 * df[driver].to_numpy()
    df["y_rm_l2"] = (rng.random(n) < 1 / (1 + np.exp(-logit))).astype(float)

    imp = permutation_importance(df, "y_rm", 2, "logistic", n_repeats=3,
                                 n_splits=2, test_size=80, gap=10)
    assert imp.iloc[0]["feature"] == driver, imp.head()
    assert imp.iloc[0]["mean_drop_bss"] > 0, imp.head()
    print(f"[OK] selftest: ฟีเจอร์ที่ขับเคลื่อน '{driver}' มาอันดับ 1 "
          f"(drop={imp.iloc[0]['mean_drop_bss']:.3f})")
    print("[OK] permutation_importance self-test ผ่าน")


if __name__ == "__main__":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass
    if len(sys.argv) > 1 and sys.argv[1] == "test":
        _selftest()
    else:
        sys.exit(main())
