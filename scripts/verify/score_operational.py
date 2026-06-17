"""Score operational/backtest forecast pairs and emit scorecard + reliability + summary."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent.parent   # scripts/verify/ -> root
sys.path.insert(0, str(ROOT / "scripts"))

from evaluate import brier, brier_skill_score, auc, reliability_curve
from analysis.stats import bootstrap_ci, ece, DEFAULT_BLOCK, DEFAULT_B, SEED

OUT_DIR = ROOT / "outputs" / "operational" / "verification"


def _compute_metrics(df: pd.DataFrame, n_prov: int) -> dict:
    """Compute metrics + bootstrap CIs for a subset of pairs.

    df must be sorted by (issue_date, province_id) before calling.

    Returns dict: metric -> {value, ci_low, ci_high, n}

    Note on block size: L = n_prov * DEFAULT_BLOCK gives an exact 28-date block
    for per-lead strata (each issue_date contributes n_prov rows).  For the "all"
    stratum each date has n_prov * n_leads rows, so the same L spans ~5-6 dates
    instead of 28 — CIs are mildly under-blocked.  The spec prescribes this formula
    and per-lead skill is the primary reporting metric, so it is implemented as-is.
    """
    y = df["y_obs"].to_numpy(float)
    p = df["probability"].to_numpy(float)
    br = df["base_rate"].to_numpy(float)
    n = len(y)
    L = n_prov * DEFAULT_BLOCK    # block size respects date-level autocorrelation

    results = {}

    # Brier
    ci = bootstrap_ci(lambda _y, _p: brier(_y, _p), (y, p), L=L, B=DEFAULT_B, seed=SEED)
    results["brier"] = {"value": ci["point"], "ci_low": ci["lo"], "ci_high": ci["hi"], "n": n}

    # BSS (with per-row base_rate as baseline — resample y/p/br together, never separately)
    ci = bootstrap_ci(lambda _y, _p, _br: brier_skill_score(_y, _p, _br),
                      (y, p, br), L=L, B=DEFAULT_B, seed=SEED)
    results["bss"] = {"value": ci["point"], "ci_low": ci["lo"], "ci_high": ci["hi"], "n": n}

    # AUC
    ci = bootstrap_ci(lambda _y, _p: auc(_y, _p), (y, p), L=L, B=DEFAULT_B, seed=SEED)
    results["auc"] = {"value": ci["point"], "ci_low": ci["lo"], "ci_high": ci["hi"], "n": n}

    # ECE (no bootstrap — just point estimate)
    ece_val = ece(y, p)
    results["ece"] = {"value": ece_val, "ci_low": float("nan"), "ci_high": float("nan"), "n": n}

    return results


def score(pairs_path: Path, out_dir: Path = OUT_DIR) -> None:
    pairs = pd.read_csv(pairs_path, parse_dates=["issue_date"])
    pairs = pairs.sort_values(["issue_date", "province_id", "lead"]).reset_index(drop=True)
    n_prov = pairs["province_id"].nunique()

    scorecard_rows = []
    reliability_rows = []

    def _add_reliability(df, stratum, lead):
        y = df["y_obs"].to_numpy(float)
        p = df["probability"].to_numpy(float)
        mp, of, ct = reliability_curve(y, p, n_bins=10)
        for b_idx, (mp_b, of_b, ct_b) in enumerate(zip(mp, of, ct)):
            reliability_rows.append({
                "stratum": stratum, "lead": lead,
                "bin": b_idx, "mean_pred": mp_b, "obs_freq": of_b, "count": int(ct_b),
            })

    def _add_scorecard(metrics, stratum, lead):
        for metric, r in metrics.items():
            scorecard_rows.append({
                "stratum": stratum, "lead": lead,
                "metric": metric, "value": r["value"],
                "ci_low": r["ci_low"], "ci_high": r["ci_high"], "n": r["n"],
            })

    # Overall (all leads)
    print("[score] Overall metrics ...")
    all_sorted = pairs.sort_values(["issue_date", "province_id"])
    metrics_all = _compute_metrics(all_sorted, n_prov)
    _add_scorecard(metrics_all, "all", None)
    _add_reliability(all_sorted, "all", None)

    # Per lead
    for L in sorted(pairs["lead"].unique()):
        print(f"[score] Lead {L} metrics ...")
        sub = pairs[pairs["lead"] == L].sort_values(["issue_date", "province_id"])
        metrics_L = _compute_metrics(sub, n_prov)
        _add_scorecard(metrics_L, f"lead_{L}", int(L))
        _add_reliability(sub, f"lead_{L}", int(L))

    # Write outputs
    out_dir.mkdir(parents=True, exist_ok=True)

    scorecard = pd.DataFrame(scorecard_rows)
    scorecard.to_csv(out_dir / "scorecard.csv", index=False)
    print(f"[score] scorecard.csv : {len(scorecard)} rows")

    rel = pd.DataFrame(reliability_rows)
    rel.to_csv(out_dir / "reliability.csv", index=False)
    print(f"[score] reliability.csv : {len(rel)} rows")

    _write_summary(pairs, scorecard, out_dir, n_prov)
    print(f"[score] summary.md written")


def _write_summary(pairs: pd.DataFrame, scorecard: pd.DataFrame,
                   out_dir: Path, n_prov: int) -> None:
    """Write human-readable summary.md."""
    date_min = pairs["issue_date"].min().date()
    date_max = pairs["issue_date"].max().date()
    n_total = len(pairs)
    n_dates = pairs["issue_date"].nunique()
    leads = sorted(pairs["lead"].unique())

    bss_all = scorecard[(scorecard["stratum"] == "all") & (scorecard["metric"] == "bss")].iloc[0]

    lines = [
        "# Operational Verification Scorecard",
        "",
        "## Dataset",
        f"- Pairs: {n_total:,}",
        f"- Issue dates: {n_dates} ({date_min} to {date_max})",
        f"- Provinces: {n_prov}",
        f"- Leads: {leads}",
        "",
        "## BSS by Lead (vs per-province climatology base_rate)",
        "",
        "| Lead | BSS | 95% CI (lo) | 95% CI (hi) | n |",
        "|------|-----|-------------|-------------|---|",
    ]
    for L in leads:
        r = scorecard[(scorecard["stratum"] == f"lead_{int(L)}") & (scorecard["metric"] == "bss")]
        if r.empty:
            continue
        r = r.iloc[0]
        lines.append(f"| {int(L)} | {r['value']:+.3f} | {r['ci_low']:+.3f} | {r['ci_high']:+.3f} | {int(r['n'])} |")

    lines += [
        "",
        f"**Overall BSS: {bss_all['value']:+.3f} [{bss_all['ci_low']:+.3f}, {bss_all['ci_high']:+.3f}]**",
        "",
        "## Caveats",
        f"- CIs use moving-block bootstrap (B=2000, block={DEFAULT_BLOCK} issue-dates ≈ {DEFAULT_BLOCK}×77 rows).",
        "- Data covers January–July only (Thai hot season + buffer). Skill estimates apply to this period.",
        "- Model trained through 2023; these scores are on 2024–2025 (genuinely out-of-sample).",
        "- BSS compared to per-province per-lead climatology base_rate (frozen 1994-2023 baseline).",
        "- Positive BSS = model beats climatology; CI not crossing 0 = statistically significant.",
    ]

    (out_dir / "summary.md").write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--pairs", required=True, type=Path,
                    help="Path to backtest_pairs.csv or operational_pairs.csv")
    ap.add_argument("--out-dir", type=Path, default=OUT_DIR)
    args = ap.parse_args()
    score(args.pairs, args.out_dir)
