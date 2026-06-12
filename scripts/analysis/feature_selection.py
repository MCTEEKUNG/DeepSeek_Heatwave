"""
ต่อยอดจากข้อค้นพบ ablation — ทดสอบว่า "logistic ที่ feature น้อยลง" ดีขึ้น/แกร่งขึ้นจริงไหม

ที่มา: ablation (เฟส 5) พบว่า logistic (linear) overfit ฟีเจอร์ soil/thermal ที่ซ้ำซ้อน
(ตัดออกแล้ว BSS กลับดีขึ้น) ส่วน ENSO + ฤดูกาล + soil_mean30 สำคัญจริง (permutation/ablation)
สมมติฐาน: โมเดลที่คัดฟีเจอร์ให้เหลือเฉพาะตัวสำคัญ จะ generalize ดีขึ้น และอาจชนะ persistence ได้มากขึ้น

วิธี: reuse train.run_target_lead(features=<subset>) บน CV เดิม -> วัด pooled BSS +
block-bootstrap 95% CI + paired test เทียบ persistence (BH-FDR) ; เทียบ delta จาก full

feature sets ที่ทดสอบ (target หลัก y_rm):
  full        20 ตัว (ครบ — baseline)
  no_thermal  16 ตัว (ตัดกลุ่ม thermal/state ที่ ablation ชี้ว่าเป็นภาระ)
  lean        10 ตัว (ฤดูกาล + ENSO + soil_mean30 ทั้ง 2 ชั้น + MJO)
  core         5 ตัว (ฤดูกาล + ENSO + soil_mean30 — แก่นล้วน)

outputs/analysis/feature_selection.csv , figures/feature_selection_bss.png

ใช้งาน:
  python feature_selection.py --leads 2          # ทดสอบเร็ว
  python feature_selection.py                    # เต็ม (ทุก lead, logistic + logistic_balanced_cal)
  python feature_selection.py test
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from evaluate import brier_skill_score  # noqa: E402
from train import FEATURES, run_target_lead, DATASET  # noqa: E402

import io_utils as io  # noqa: E402
from stats import bootstrap_ci, paired_block_test, bh_fdr, DEFAULT_BLOCK, DEFAULT_B, SEED  # noqa: E402

THERMAL = ["tmax_rm", "tmax_mean7", "in_hw_today", "hot_frac7"]
SEASONAL = ["doy_sin", "doy_cos"]
ENSO = ["nino34_lag1m"]
MJO = ["mjo_rmm1", "mjo_rmm2", "mjo_amp", "mjo_sin", "mjo_cos"]
SOIL_KEY = ["sm1_mean30", "sm3_mean30"]   # soil ที่ permutation/ablation ชี้ว่าสำคัญสุด

FEATURE_SETS = {
    "full": list(FEATURES),
    "no_thermal": [f for f in FEATURES if f not in THERMAL],
    "lean": SEASONAL + ENSO + SOIL_KEY + MJO,
    "core": SEASONAL + ENSO + SOIL_KEY,
}
MODELS = ["logistic", "logistic_balanced_cal"]


def _bss(y, p, pc):
    return brier_skill_score(y, p, baseline_prob=pc)


def run_experiment(df, leads, models, sets=FEATURE_SETS,
                   block=DEFAULT_BLOCK, B=DEFAULT_B) -> pd.DataFrame:
    rows = []
    for set_name, feats in sets.items():
        for lead in leads:
            _, pred = run_target_lead(df, io.PRIMARY_TARGET, lead, verbose=False,
                                      features=feats, models=models)
            pers = pred[pred.model == "persistence"].sort_values(["fold", "date"])
            for m in models:
                g = pred[pred.model == m].sort_values(["fold", "date"])
                if g.empty:
                    continue
                y = g["y"].to_numpy(float)
                p = g["p"].to_numpy(float)
                pc = g["p_clim"].to_numpy(float)
                ci = bootstrap_ci(_bss, (y, p, pc), L=block, B=B, seed=SEED)
                # paired vs persistence (align fold+date)
                mg = g.merge(pers[["fold", "date", "p"]], on=["fold", "date"],
                             suffixes=("", "_pers")).sort_values(["fold", "date"])
                yy = mg["y"].to_numpy(float)
                d = (mg["p_pers"].to_numpy(float) - yy) ** 2 - (mg["p"].to_numpy(float) - yy) ** 2
                pt = paired_block_test(d, L=block, B=B, seed=SEED)
                rows.append({
                    "feature_set": set_name, "n_features": len(feats), "lead": lead,
                    "model": m, "n": len(y), "bss": ci["point"],
                    "bss_lo95": ci["lo"], "bss_hi95": ci["hi"],
                    "p_vs_persist": pt["p_boot"],
                })
    out = pd.DataFrame(rows)
    # delta เทียบ full
    full = (out[out.feature_set == "full"][["lead", "model", "bss"]]
            .rename(columns={"bss": "bss_full"}))
    out = out.merge(full, on=["lead", "model"], how="left")
    out["delta_vs_full"] = out["bss"] - out["bss_full"]
    out = out.drop(columns=["bss_full"])
    # BH-FDR ต่อ model (family = sets x leads) vs persistence
    out["q_vs_persist"] = np.nan
    for _, idx in out.groupby("model").groups.items():
        out.loc[idx, "q_vs_persist"] = bh_fdr(out.loc[idx, "p_vs_persist"].to_numpy())
    out["beats_clim"] = out["bss_lo95"] > 0
    out["beats_persist"] = out["q_vs_persist"] < 0.05
    return out.sort_values(["model", "feature_set", "lead"]).reset_index(drop=True)


def fig_compare(out, path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    colors = {"full": "#7f7f7f", "no_thermal": "#1f77b4", "lean": "#2ca02c", "core": "#ff7f0e"}
    sub = out[out.model == "logistic"]
    fig, ax = plt.subplots(figsize=(8, 5))
    for s in FEATURE_SETS:
        g = sub[sub.feature_set == s].sort_values("lead")
        if g.empty:
            continue
        yerr = np.vstack([g.bss - g.bss_lo95, g.bss_hi95 - g.bss])
        ax.errorbar(g.lead, g.bss, yerr=yerr, marker="o", capsize=3, lw=1.8,
                    label=f"{s} ({g.n_features.iloc[0]} feat)", color=colors.get(s))
    ax.axhline(0, color="black", lw=1)
    ax.set_xlabel("Lead time (weeks)")
    ax.set_ylabel("BSS (vs climatology)")
    ax.set_title("Logistic: full vs reduced feature sets (y_rm)")
    ax.set_xticks(sorted(sub.lead.unique()))
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def _print_summary(out) -> None:
    print("\n=== BSS (pooled) — logistic, แต่ละ feature set x lead ===")
    piv = out[out.model == "logistic"].pivot_table(index="feature_set", columns="lead",
                                                    values="bss")
    piv = piv.reindex(list(FEATURE_SETS))
    print(piv.round(3).to_string())

    print("\n=== จำนวน lead ที่ 'ชนะ persistence' (q<0.05) ต่อ feature set ===")
    for m in out.model.unique():
        cnt = (out[out.model == m].groupby("feature_set")["beats_persist"].sum()
               .reindex(list(FEATURE_SETS)).astype(int))
        print(f"[{m}] " + " | ".join(f"{s}:{int(cnt[s])}" for s in FEATURE_SETS))

    print("\n=== delta BSS เทียบ full (บวก = ดีขึ้น) — logistic ===")
    dp = out[out.model == "logistic"].pivot_table(index="feature_set", columns="lead",
                                                  values="delta_vs_full")
    print(dp.reindex(list(FEATURE_SETS)).round(3).to_string())


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="reduced-feature logistic experiment")
    ap.add_argument("--leads", nargs="*", type=int, default=io.LEADS)
    ap.add_argument("--models", nargs="*", default=MODELS)
    ap.add_argument("--B", type=int, default=DEFAULT_B)
    args = ap.parse_args(argv)

    io.ensure_dirs()
    df = pd.read_csv(DATASET, parse_dates=["date"], index_col="date")
    print(f"=== feature_selection: dataset {len(df)} วัน | sets={list(FEATURE_SETS)} | "
          f"leads={args.leads} models={args.models} ===", flush=True)

    out = run_experiment(df, args.leads, args.models, B=args.B)
    out.to_csv(io.ANALYSIS_DIR / "feature_selection.csv", index=False)
    print(f"[OK] feature_selection.csv : {len(out)} แถว")
    fig_compare(out, io.FIG_DIR / "feature_selection_bss.png")
    print("[OK] feature_selection_bss.png")
    _print_summary(out)
    print(f"\n[OK] ผลอยู่ที่ {io.ANALYSIS_DIR}")
    return 0


# ---------------------------------------------------------------- self-test

def _selftest() -> None:
    # y ขึ้นกับ 2 ฟีเจอร์ (nino34_lag1m, sm1_mean30) ; ที่เหลือเป็น noise
    # -> 'core' (เฉพาะสัญญาณ) ควร BSS >= 'full' (มี noise) สำหรับ logistic
    rng = np.random.default_rng(5)
    n = 1650
    idx = pd.date_range("2008-01-01", periods=n, freq="D")
    df = pd.DataFrame(index=idx)
    for c in FEATURES:
        df[c] = rng.standard_normal(n)
    df["doy"] = idx.dayofyear
    logit = 1.8 * df["nino34_lag1m"].to_numpy() + 1.5 * df["sm1_mean30"].to_numpy()
    p_true = 1 / (1 + np.exp(-logit))
    df["y_rm_l2"] = (rng.random(n) < p_true).astype(float)
    df.index.name = "date"

    out = run_experiment(df, [2], ["logistic"], B=300)
    bss = out.set_index("feature_set")["bss"]
    assert {"full", "core", "lean", "no_thermal"} <= set(bss.index)
    assert "delta_vs_full" in out.columns and "beats_persist" in out.columns
    assert bss["core"] >= bss["full"] - 0.02, f"core ควรไม่แย่กว่า full: core={bss['core']:.3f} full={bss['full']:.3f}"
    print(f"[OK] selftest: core BSS={bss['core']:+.3f} ~>= full BSS={bss['full']:+.3f} "
          f"(noise ไม่ช่วย linear model)")
    print("[OK] feature_selection self-test ผ่าน")


if __name__ == "__main__":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass
    if len(sys.argv) > 1 and sys.argv[1] == "test":
        _selftest()
    else:
        sys.exit(main())
