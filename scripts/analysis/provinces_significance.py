"""Significance ของโมเดล pooled per-province (ตาม spec §8 — rigor เดียวกับ regional).

bootstrap CI ของ BSS (vs climatology) + paired block test vs climatology AND persistence + BH-FDR.
reuse stats.py (moving-block bootstrap). block = ~วัน: pooled มี ~77 แถว/วัน ->
L_rows = DEFAULT_BLOCK(วัน) * แถวต่อวัน บนชุดที่เรียงตาม (date, province_id).
"""
from __future__ import annotations

import sys
from pathlib import Path

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "scripts" / "analysis"))
from stats import bootstrap_ci, paired_block_test, bh_fdr, DEFAULT_BLOCK, DEFAULT_B
from evaluate import brier_skill_score

ANALYSIS_DIR = ROOT / "outputs" / "analysis"
PRED = ANALYSIS_DIR / "provinces_predictions.csv"


def _bss(y, p, pc):
    return brier_skill_score(y, p, baseline_prob=pc)


def main() -> int:
    d = pd.read_csv(PRED, parse_dates=["date"])
    rows = []
    for lead, g in d.groupby("lead"):
        g = g.sort_values(["date", "province_id"])
        y = g["y"].to_numpy(float); p = g["p"].to_numpy(float)
        pc = g["p_clim"].to_numpy(float); pp = g["p_pers"].to_numpy(float)
        per_day = len(g) / g["date"].nunique()
        L = max(1, int(round(DEFAULT_BLOCK * per_day)))
        ci = bootstrap_ci(_bss, (y, p, pc), L=L, B=DEFAULT_B)
        t_clim = paired_block_test((pc - y) ** 2 - (p - y) ** 2, L=L, B=DEFAULT_B)
        t_pers = paired_block_test((pp - y) ** 2 - (p - y) ** 2, L=L, B=DEFAULT_B)
        rows.append({"lead": int(lead), "block_rows": L, "bss": ci["point"],
                     "bss_lo95": ci["lo"], "bss_hi95": ci["hi"],
                     "p_vs_clim": t_clim["p_boot"], "p_vs_persist": t_pers["p_boot"]})
    res = pd.DataFrame(rows).sort_values("lead")
    res["q_vs_clim"] = bh_fdr(res["p_vs_clim"].to_numpy())
    res["q_vs_persist"] = bh_fdr(res["p_vs_persist"].to_numpy())
    res["beats_clim"] = res["bss_lo95"] > 0
    res["beats_persist"] = res["q_vs_persist"] < 0.05
    res.to_csv(ANALYSIS_DIR / "provinces_significance.csv", index=False)
    print(res.round(4).to_string(index=False))
    nclim = int(res["beats_clim"].sum()); npers = int(res["beats_persist"].sum())
    print(f"\nbeats climatology (BSS 95% CI > 0): {nclim}/{len(res)} leads")
    print(f"beats persistence (q<0.05 BH-FDR):  {npers}/{len(res)} leads")
    hw = ((res["bss_hi95"] - res["bss_lo95"]) / 2).round(4).tolist()
    print(f"BSS CI half-widths per lead: {hw}  (R1 ΔBSS=-0.0055 should be << these)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
