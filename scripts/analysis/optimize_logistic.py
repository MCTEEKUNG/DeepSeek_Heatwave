"""
ดันสกิลต่อจาก feature_selection — หาคอนฟิก logistic ที่ดี/แกร่งสุด

ตอบ 2 คำถาม:
  (ก) L2 regularization แรง ๆ บน "full" (20 ฟีเจอร์) แก้ overfitting ได้เท่าการ "ตัดฟีเจอร์" ไหม?
      (ถ้าได้ใกล้เคียง = ปัญหาคือ regularization ไม่พอ ; การตัดฟีเจอร์เป็นวิธี regularize แบบหนึ่ง)
  (ข) เอา core/lean ไป recalibrate (class_weight balanced + Platt) แล้วดีขึ้นอีกไหม?

คุมเอง 3 ปัจจัย: (ชุดฟีเจอร์ × ค่า C ของ L2 × ทำ calibration ไหม) บน CV เดิม (rolling-origin gap 49)
reuse: cv.RollingOriginCV, evaluate (climatology/persistence/BSS), train.PlattCalibrator + ค่าคงที่,
       stats (block-bootstrap CI + paired test + BH-FDR), feature_selection.FEATURE_SETS

outputs/analysis/optimize_logistic.csv , figures/optimize_logistic_bss.png

ใช้งาน:  python optimize_logistic.py --leads 2   |   python optimize_logistic.py   |   ... test
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from evaluate import (brier_skill_score, predict_seasonal_climatology,  # noqa: E402
                      persistence_probs)
from cv import RollingOriginCV  # noqa: E402
from train import FEATURES, N_SPLITS, TEST_SIZE, GAP, CALIB_FRAC, PlattCalibrator, DATASET  # noqa: E402

import io_utils as io  # noqa: E402
from stats import bootstrap_ci, paired_block_test, bh_fdr, DEFAULT_BLOCK, DEFAULT_B, SEED  # noqa: E402
from feature_selection import FEATURE_SETS  # noqa: E402

# คอนฟิกที่เทียบ: (ชื่อ, ชุดฟีเจอร์, C ของ L2 (เล็ก=แรง), ทำ calibration ไหม)
CONFIGS = [
    {"name": "full",      "features": FEATURE_SETS["full"], "C": 1.0,  "cal": False},
    {"name": "full_L2",   "features": FEATURE_SETS["full"], "C": 0.05, "cal": False},
    {"name": "core",      "features": FEATURE_SETS["core"], "C": 1.0,  "cal": False},
    {"name": "core_cal",  "features": FEATURE_SETS["core"], "C": 1.0,  "cal": True},
    {"name": "lean_cal",  "features": FEATURE_SETS["lean"], "C": 1.0,  "cal": True},
]


def _bss(y, p, pc):
    return brier_skill_score(y, p, baseline_prob=pc)


def fit_predict(X_tr, y_tr, X_te, C, cal):
    """logistic (มี L2 ผ่าน C) ; ถ้า cal=True ใช้ balanced + Platt บน block แยกเวลา (กัน leak)."""
    if not cal:
        m = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000, C=C))
        m.fit(X_tr, y_tr)
        return m.predict_proba(X_te)[:, 1]
    n = len(y_tr)
    n_calib = int(n * CALIB_FRAC)
    n_core = n - n_calib - GAP
    if n_core < 300 or n_calib < 100:
        return None
    y_cal = y_tr[n - n_calib:]
    if len(np.unique(y_cal)) < 2:
        return None
    m = make_pipeline(StandardScaler(),
                      LogisticRegression(max_iter=1000, C=C, class_weight="balanced"))
    m.fit(X_tr[:n_core], y_tr[:n_core])
    p_cal = m.predict_proba(X_tr[n - n_calib:])[:, 1]
    p_te = m.predict_proba(X_te)[:, 1]
    return PlattCalibrator().fit(p_cal, y_cal).transform(p_te)


def run_config(df, lead, cfg):
    """รัน CV หนึ่งคอนฟิก -> pooled (y, p, p_clim, p_pers) ; None ถ้ารันไม่ได้."""
    col = f"{io.PRIMARY_TARGET}_l{lead}"
    sub = df[FEATURES + ["doy", col]].dropna().sort_index()
    Xc = sub[cfg["features"]].to_numpy(float)
    y = sub[col].to_numpy(float)
    doy = sub["doy"].to_numpy(int)
    state = sub["in_hw_today"].to_numpy(int)
    cv = RollingOriginCV(n_splits=N_SPLITS, test_size=TEST_SIZE, gap=GAP, expanding=True)
    Y, P, PC, PR = [], [], [], []
    for tr, te in cv.split(len(sub)):
        if len(np.unique(y[tr])) < 2:
            continue
        p = fit_predict(Xc[tr], y[tr], Xc[te], cfg["C"], cfg["cal"])
        if p is None:
            continue
        Y.append(y[te]); P.append(p)
        PC.append(predict_seasonal_climatology(doy[tr], y[tr], doy[te]))
        PR.append(persistence_probs(state[tr], y[tr], state[te]))
    if not Y:
        return None
    return tuple(np.concatenate(a) for a in (Y, P, PC, PR))


def run_experiment(df, leads, block=DEFAULT_BLOCK, B=DEFAULT_B) -> pd.DataFrame:
    rows = []
    for cfg in CONFIGS:
        for lead in leads:
            res = run_config(df, lead, cfg)
            if res is None:
                continue
            y, p, pc, pr = res
            ci = bootstrap_ci(_bss, (y, p, pc), L=block, B=B, seed=SEED)
            d = (pr - y) ** 2 - (p - y) ** 2          # persistence_brier - model_brier
            pt = paired_block_test(d, L=block, B=B, seed=SEED)
            rows.append({
                "config": cfg["name"], "n_features": len(cfg["features"]),
                "C": cfg["C"], "calibrated": cfg["cal"], "lead": lead, "n": len(y),
                "bss": ci["point"], "bss_lo95": ci["lo"], "bss_hi95": ci["hi"],
                "p_vs_persist": pt["p_boot"],
            })
    out = pd.DataFrame(rows)
    full = out[out.config == "full"][["lead", "bss"]].rename(columns={"bss": "bss_full"})
    out = out.merge(full, on="lead", how="left")
    out["delta_vs_full"] = out["bss"] - out["bss_full"]
    out = out.drop(columns=["bss_full"])
    out["q_vs_persist"] = bh_fdr(out["p_vs_persist"].to_numpy())   # family = ทุก config x lead
    out["beats_clim"] = out["bss_lo95"] > 0
    out["beats_persist"] = out["q_vs_persist"] < 0.05
    return out.sort_values(["config", "lead"]).reset_index(drop=True)


def fig_compare(out, path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    colors = {"full": "#7f7f7f", "full_L2": "#9467bd", "core": "#ff7f0e",
              "core_cal": "#2ca02c", "lean_cal": "#1f77b4"}
    fig, ax = plt.subplots(figsize=(8.5, 5))
    for c in [cfg["name"] for cfg in CONFIGS]:
        g = out[out.config == c].sort_values("lead")
        if g.empty:
            continue
        yerr = np.vstack([g.bss - g.bss_lo95, g.bss_hi95 - g.bss])
        ax.errorbar(g.lead, g.bss, yerr=yerr, marker="o", capsize=3, lw=1.8,
                    label=c, color=colors.get(c))
    ax.axhline(0, color="black", lw=1)
    ax.set_xlabel("Lead time (weeks)")
    ax.set_ylabel("BSS (vs climatology)")
    ax.set_title("Logistic configs: regularization vs feature selection vs calibration (y_rm)")
    ax.set_xticks(sorted(out.lead.unique()))
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def _print_summary(out) -> None:
    print("\n=== BSS (pooled) — แต่ละ config x lead ===")
    piv = out.pivot_table(index="config", columns="lead", values="bss") \
        .reindex([c["name"] for c in CONFIGS])
    print(piv.round(3).to_string())
    print("\n=== delta BSS เทียบ full (บวก=ดีขึ้น) ===")
    dp = out.pivot_table(index="config", columns="lead", values="delta_vs_full") \
        .reindex([c["name"] for c in CONFIGS])
    print(dp.round(3).to_string())
    print("\n=== จำนวน lead ที่ชนะ climatology (CI>0) / ชนะ persistence (q<0.05) ===")
    for c in [cfg["name"] for cfg in CONFIGS]:
        g = out[out.config == c]
        print(f"  {c:10s} ชนะ clim {int(g.beats_clim.sum())}/{len(g)} | "
              f"ชนะ persist {int(g.beats_persist.sum())}/{len(g)}")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="optimize logistic: reg + feature selection + calibration")
    ap.add_argument("--leads", nargs="*", type=int, default=io.LEADS)
    ap.add_argument("--B", type=int, default=DEFAULT_B)
    args = ap.parse_args(argv)

    io.ensure_dirs()
    df = pd.read_csv(DATASET, parse_dates=["date"], index_col="date")
    print(f"=== optimize_logistic: dataset {len(df)} วัน | configs={[c['name'] for c in CONFIGS]} "
          f"| leads={args.leads} ===", flush=True)

    out = run_experiment(df, args.leads, B=args.B)
    out.to_csv(io.ANALYSIS_DIR / "optimize_logistic.csv", index=False)
    print(f"[OK] optimize_logistic.csv : {len(out)} แถว")
    fig_compare(out, io.FIG_DIR / "optimize_logistic_bss.png")
    print("[OK] optimize_logistic_bss.png")
    _print_summary(out)
    print(f"\n[OK] ผลอยู่ที่ {io.ANALYSIS_DIR}")
    return 0


# ---------------------------------------------------------------- self-test

def _selftest() -> None:
    rng = np.random.default_rng(6)
    n = 1650
    idx = pd.date_range("2008-01-01", periods=n, freq="D")
    df = pd.DataFrame(index=idx)
    for c in FEATURES:
        df[c] = rng.standard_normal(n)
    df["doy"] = idx.dayofyear
    logit = 1.8 * df["nino34_lag1m"].to_numpy() + 1.5 * df["sm1_mean30"].to_numpy()
    df["y_rm_l2"] = (rng.random(n) < 1 / (1 + np.exp(-logit))).astype(float)
    df.index.name = "date"

    out = run_experiment(df, [2], B=250)
    assert set(out.config) == {c["name"] for c in CONFIGS}
    assert {"delta_vs_full", "q_vs_persist", "beats_clim"} <= set(out.columns)
    # มีสัญญาณจริง -> ทุก config ควร BSS > 0 ; full_L2/core ไม่ควรแย่กว่า full อย่างมาก
    b = out.set_index("config")["bss"]
    assert (b > 0).all(), b
    assert b["full_L2"] >= b["full"] - 0.03 and b["core"] >= b["full"] - 0.03, b
    print(f"[OK] selftest: full={b['full']:+.3f} full_L2={b['full_L2']:+.3f} "
          f"core={b['core']:+.3f} core_cal={b['core_cal']:+.3f} lean_cal={b['lean_cal']:+.3f}")
    print("[OK] optimize_logistic self-test ผ่าน")


if __name__ == "__main__":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass
    if len(sys.argv) > 1 and sys.argv[1] == "test":
        _selftest()
    else:
        sys.exit(main())
