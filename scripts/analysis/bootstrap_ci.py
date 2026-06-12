"""
เฟส 1 — ความไม่แน่นอน & นัยสำคัญของสกิล (อ่าน outputs/predictions.csv อย่างเดียว)

ตอบคำถามหลักของวิทยานิพนธ์: "สกิล BSS +0.168 ที่ lead 2 จริง หรือบังเอิญ?"

ทำอะไร:
  1) CI 95% (moving-block bootstrap) ของ BSS / Brier / AUC ต่อ (target, lead, model)
     - block bootstrap คุม autocorrelation ของอนุกรมพยากรณ์ (ดู stats.py)
     - resample ดัชนีชุดเดียวใช้กับ y/p/p_clim พร้อมกัน (BSS ไม่เพี้ยน)
  2) paired test (block bootstrap บนผลต่าง Brier ราย "วัน"):
        model vs climatology (ใช้คอลัมน์ p_clim ของแถวนั้น)
        model vs persistence (join แถว persistence ด้วย fold+date)
  3) คุม multiple comparisons ด้วย Benjamini–Hochberg FDR (q-value) ต่อ (target, reference)

เกณฑ์ตัดสิน "สกิลจริง": CI ของ BSS ไม่คร่อม 0  และ  paired-vs-persistence q < 0.05
(persistence เป็น baseline ที่แข็งกว่า climatology จึงเป็นด่านชี้ขาดที่ตึงกว่า)

outputs/analysis/bootstrap_ci.csv , paired_tests.csv

ใช้งาน:
  python bootstrap_ci.py                      # รันเต็ม (ทุก target/lead/model)
  python bootstrap_ci.py --targets y_rm --leads 2 --B 500   # ทดสอบเร็ว
  python bootstrap_ci.py test                 # self-test ข้อมูลสังเคราะห์
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from evaluate import brier, brier_skill_score, auc  # noqa: E402

import io_utils as io  # noqa: E402
from stats import bootstrap_ci, paired_block_test, bh_fdr, DEFAULT_BLOCK, DEFAULT_B, SEED  # noqa: E402


# --- stat functions (resample ดัชนีเดียวกันให้ทุก array) ---
def _bss(y, p, pc):
    return brier_skill_score(y, p, baseline_prob=pc)


def _brier(y, p):
    return brier(y, p)


def _auc(y, p):
    return auc(y, p)


# models ที่นำมาทดสอบ paired (ไม่รวม climatology เพราะมันคือ reference ฐาน)
def _models_to_test(present: list[str]) -> list[str]:
    return [m for m in present if m != "climatology"]


def compute_ci_table(pred, targets, leads, models, block=DEFAULT_BLOCK, B=DEFAULT_B) -> pd.DataFrame:
    rows = []
    for t, l, m, g in io.iter_combos(pred, targets=targets, leads=leads, models=models):
        y = g["y"].to_numpy(float)
        p = g["p"].to_numpy(float)
        pc = g["p_clim"].to_numpy(float)
        specs = (("bss", _bss, (y, p, pc)), ("brier", _brier, (y, p)), ("auc", _auc, (y, p)))
        for metric, fn, arrs in specs:
            ci = bootstrap_ci(fn, arrs, L=block, B=B, seed=SEED)
            rows.append({
                "target": t, "lead": l, "model": m, "metric": metric,
                "point": ci["point"], "lo95": ci["lo"], "hi95": ci["hi"],
                "n": len(y), "n_dropped": ci["n_dropped"],
            })
    return pd.DataFrame(rows)


def compute_paired_table(pred, targets, leads, block=DEFAULT_BLOCK, B=DEFAULT_B) -> pd.DataFrame:
    rows = []
    pairs = pred[["target", "lead"]].drop_duplicates()
    if targets is not None:
        pairs = pairs[pairs["target"].isin(targets)]
    if leads is not None:
        pairs = pairs[pairs["lead"].isin(leads)]

    for t, l in pairs.itertuples(index=False):
        present = pred[(pred.target == t) & (pred.lead == l)]["model"].unique().tolist()
        pers = io.model_series(pred, t, l, "persistence") if "persistence" in present else None
        for m in _models_to_test(present):
            g = io.model_series(pred, t, l, m)
            y = g["y"].to_numpy(float)
            bs_model = (g["p"].to_numpy(float) - y) ** 2

            # --- vs climatology (p_clim อยู่ในแถวเดียวกัน ไม่ต้อง join) ---
            d_clim = (g["p_clim"].to_numpy(float) - y) ** 2 - bs_model
            r = paired_block_test(d_clim, L=block, B=B, seed=SEED)
            rows.append({"target": t, "lead": l, "model": m, "reference": "climatology", **r})

            # --- vs persistence (join ด้วย fold+date) ---
            if pers is not None and m != "persistence":
                mg = g.merge(pers[["fold", "date", "p"]], on=["fold", "date"],
                             suffixes=("", "_ref")).sort_values(["fold", "date"])
                yy = mg["y"].to_numpy(float)
                d_pers = (mg["p_ref"].to_numpy(float) - yy) ** 2 - (mg["p"].to_numpy(float) - yy) ** 2
                r2 = paired_block_test(d_pers, L=block, B=B, seed=SEED)
                rows.append({"target": t, "lead": l, "model": m, "reference": "persistence", **r2})

    df = pd.DataFrame(rows).rename(columns={"mean_d": "mean_brier_diff"})
    # BH-FDR ต่อ family (target, reference) คุม false discovery ข้าม (lead x model)
    df["q_boot"] = np.nan
    for _, idx in df.groupby(["target", "reference"]).groups.items():
        df.loc[idx, "q_boot"] = bh_fdr(df.loc[idx, "p_boot"].to_numpy())
    cols = ["target", "lead", "model", "reference", "mean_brier_diff", "ci_lo", "ci_hi",
            "se", "p_boot", "p_normal", "q_boot", "n"]
    return df[cols].sort_values(["target", "reference", "lead", "model"]).reset_index(drop=True)


def _print_summary(ci_df, paired_df) -> None:
    t = io.PRIMARY_TARGET
    print(f"\n=== สรุป: BSS 95% CI (block bootstrap) — {t}, โมเดลที่ใช้รายงาน ===")
    bss = ci_df[(ci_df.target == t) & (ci_df.metric == "bss")]
    pp = paired_df[(paired_df.target == t) & (paired_df.reference == "persistence")]
    for m in io.REPORT_MODELS:
        for l in io.LEADS:
            row = bss[(bss.model == m) & (bss.lead == l)]
            if row.empty:
                continue
            r = row.iloc[0]
            sig = "✓" if (r.lo95 > 0) else " "
            pr = pp[(pp.model == m) & (pp.lead == l)]
            extra = ""
            if not pr.empty:
                q = pr.iloc[0].q_boot
                extra = f" | vs persist q={q:.3f}{' *' if q < 0.05 else ''}"
            print(f"  {m:22s} lead {l}: BSS={r.point:+.3f} [{r.lo95:+.3f}, {r.hi95:+.3f}] {sig}{extra}")
    print("  (✓ = CI ของ BSS ไม่คร่อม 0 ; * = ชนะ persistence อย่างมีนัยสำคัญหลังคุม FDR)")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="block-bootstrap CI + paired significance")
    ap.add_argument("--targets", nargs="*", default=None)
    ap.add_argument("--leads", nargs="*", type=int, default=None)
    ap.add_argument("--models", nargs="*", default=None)
    ap.add_argument("--B", type=int, default=DEFAULT_B)
    ap.add_argument("--block", type=int, default=DEFAULT_BLOCK)
    args = ap.parse_args(argv)

    io.ensure_dirs()
    pred = io.load_predictions()
    print(f"=== bootstrap_ci: predictions {len(pred)} แถว | B={args.B}, block={args.block} วัน ===")

    ci_df = compute_ci_table(pred, args.targets, args.leads, args.models, args.block, args.B)
    ci_df.to_csv(io.ANALYSIS_DIR / "bootstrap_ci.csv", index=False)
    print(f"[OK] bootstrap_ci.csv : {len(ci_df)} แถว")

    paired_df = compute_paired_table(pred, args.targets, args.leads, args.block, args.B)
    paired_df.to_csv(io.ANALYSIS_DIR / "paired_tests.csv", index=False)
    print(f"[OK] paired_tests.csv : {len(paired_df)} แถว")

    _print_summary(ci_df, paired_df)
    print(f"\n[OK] ผลอยู่ที่ {io.ANALYSIS_DIR}")
    return 0


# ---------------------------------------------------------------- self-test

def _make_synth_pred(seed=0) -> pd.DataFrame:
    """predictions สังเคราะห์: โมเดล 'skilled' ดีกว่า climatology จริง ; 'noise' ไม่ดีกว่า."""
    rng = np.random.default_rng(seed)
    rows = []
    base = 0.2
    for fold in range(1, 4):
        dates = pd.date_range(f"20{14 + fold}-01-01", periods=200, freq="D")
        # ความเสี่ยงจริงแกว่งตามฤดู -> climatology จับได้บางส่วน, skilled จับได้มากกว่า
        season = 0.5 + 0.5 * np.sin(2 * np.pi * np.arange(200) / 180)
        p_true = np.clip(base * (0.3 + 1.4 * season), 0.02, 0.95)
        y = (rng.random(200) < p_true).astype(float)
        p_clim = np.full(200, base)                       # climatology = base rate คงที่
        p_skill = np.clip(p_true + 0.05 * rng.standard_normal(200), 0.01, 0.99)
        p_noise = np.clip(base + 0.05 * rng.standard_normal(200), 0.01, 0.99)
        p_pers = np.clip(base + 0.1 * (season - 0.5), 0.01, 0.99)
        for model, p in (("climatology", p_clim), ("persistence", p_pers),
                         ("skilled", p_skill), ("noise", p_noise)):
            rows.append(pd.DataFrame({"target": "y_rm", "lead": 2, "fold": fold,
                                      "model": model, "date": dates, "y": y, "p": p,
                                      "p_clim": p_clim}))
    return pd.concat(rows, ignore_index=True)


def _selftest() -> None:
    pred = _make_synth_pred()
    ci = compute_ci_table(pred, None, None, None, block=20, B=300)
    paired = compute_paired_table(pred, None, None, block=20, B=300)

    # 1) ตารางมีครบทุกเมตริก/โมเดล
    assert set(ci["metric"]) == {"bss", "brier", "auc"}
    assert {"skilled", "noise", "climatology", "persistence"} <= set(ci["model"])

    # 2) climatology เทียบตัวเอง: BSS = 0 เป๊ะ (p == p_clim)
    cb = ci[(ci.model == "climatology") & (ci.metric == "bss")].iloc[0]
    assert abs(cb.point) < 1e-9, cb.point

    # 3) skilled ต้องมี BSS > 0 และชนะ climatology อย่างมีนัยสำคัญ ; noise ต้องไม่ชนะ
    sb = ci[(ci.model == "skilled") & (ci.metric == "bss")].iloc[0]
    assert sb.point > 0 and sb.lo95 > 0, sb
    ps = paired[(paired.model == "skilled") & (paired.reference == "climatology")].iloc[0]
    pn = paired[(paired.model == "noise") & (paired.reference == "climatology")].iloc[0]
    assert ps.p_boot < 0.05 and ps.mean_brier_diff > 0, ps
    assert pn.p_boot > 0.05, pn

    # 4) มีคอลัมน์ q_boot และอยู่ใน [0,1]
    assert paired["q_boot"].between(0, 1).all()
    print(f"[OK] selftest: skilled BSS={sb.point:+.3f} [{sb.lo95:+.3f},{sb.hi95:+.3f}] "
          f"ชนะ clim (p={ps.p_boot:.3f}) ; noise ไม่ชนะ (p={pn.p_boot:.2f})")
    print("[OK] bootstrap_ci self-test ผ่าน")


if __name__ == "__main__":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass
    if len(sys.argv) > 1 and sys.argv[1] == "test":
        _selftest()
    else:
        sys.exit(main())
