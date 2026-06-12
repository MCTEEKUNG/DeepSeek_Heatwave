"""
เฟส 6 (ส่วนรูป) — สร้างรูปคุณภาพตีพิมพ์จาก outputs/analysis/*.csv

รูปที่สร้าง (ข้ามอัตโนมัติถ้า input ยังไม่มี):
  bss_vs_lead_ci.png          BSS vs lead + แท่ง error 95% CI (รูปเด็ด) + เส้น persistence/climatology
  skill_by_enso.png           BSS แยกตาม ENSO regime + annotate base rate (เล่าเรื่อง base-rate shift)
  permutation_importance.png  ความสำคัญฟีเจอร์ (BSS drop) ของ lgbm/logistic
  calibration_methods_lgbm.png  เทียบ raw/platt/isotonic ของ lgbm

หมายเหตุ: ใช้ "ป้ายอังกฤษ" ในรูป เพื่อเลี่ยงปัญหา font ไทยใน matplotlib (และเป็นมาตรฐาน thesis)
ใช้งาน:  python figures.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import io_utils as io  # noqa: E402

DPI = 130
COLORS = {"logistic": "#1f77b4", "lgbm_cal": "#d62728", "logistic_balanced_cal": "#2ca02c"}


def _read(name: str):
    p = io.ANALYSIS_DIR / name
    if not p.exists():
        print(f"[skip] ไม่พบ {name} (ยังไม่ได้รันเฟสที่เกี่ยวข้อง)")
        return None
    return pd.read_csv(p)


def fig_bss_vs_lead(target: str = io.PRIMARY_TARGET) -> None:
    ci = _read("bootstrap_ci.csv")
    if ci is None:
        return
    bss = ci[(ci.target == target) & (ci.metric == "bss")]
    fig, ax = plt.subplots(figsize=(8, 5))
    for m in io.REPORT_MODELS:
        s = bss[bss.model == m].sort_values("lead")
        if s.empty:
            continue
        yerr = np.vstack([s.point - s.lo95, s.hi95 - s.point])
        ax.errorbar(s.lead, s.point, yerr=yerr, marker="o", capsize=4, lw=2,
                    label=io.MODEL_LABELS.get(m, m), color=COLORS.get(m))
    pers = bss[bss.model == "persistence"].sort_values("lead")
    if not pers.empty:
        ax.plot(pers.lead, pers.point, "--", color="gray", lw=1.5, label="Persistence")
    ax.axhline(0, color="black", lw=1)
    ax.text(0.99, 0.02, "Climatology (BSS = 0)", transform=ax.transAxes,
            ha="right", va="bottom", fontsize=8)
    ax.set_xlabel("Lead time (weeks)")
    ax.set_ylabel("Brier Skill Score (vs climatology)")
    ax.set_title(f"Sub-seasonal heatwave skill with 95% CI — {target}")
    ax.set_xticks(sorted(bss.lead.unique()))
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    out = io.FIG_DIR / "bss_vs_lead_ci.png"
    fig.savefig(out, dpi=DPI)
    plt.close(fig)
    print(f"[OK] {out.name}")


def fig_skill_by_enso(target: str = io.PRIMARY_TARGET, lead: int = 2) -> None:
    reg = _read("regime_by_enso.csv")
    if reg is None:
        return
    reg = reg[(reg.target == target) & (reg.lead == lead) & (reg.use == "lag1m")]
    order = ["lanina", "neutral", "elnino"]
    x = np.arange(len(order))
    w = 0.8 / len(io.REPORT_MODELS)
    fig, ax = plt.subplots(figsize=(8, 5))
    for i, m in enumerate(io.REPORT_MODELS):
        vals = [reg[(reg.model == m) & (reg.regime == r)]["bss"].mean() for r in order]
        ax.bar(x + i * w, vals, w, label=io.MODEL_LABELS.get(m, m), color=COLORS.get(m))
    ymax = ax.get_ylim()[1]
    for j, r in enumerate(order):
        br = reg[(reg.model == "climatology") & (reg.regime == r)]["base_rate"].mean()
        if np.isfinite(br):
            ax.text(x[j] + 0.3, ymax * 0.9, f"base rate\n{br:.2f}", ha="center",
                    fontsize=8, color="dimgray")
    ax.axhline(0, color="black", lw=1)
    ax.set_xticks(x + 0.3)
    ax.set_xticklabels([r.capitalize() for r in order])
    ax.set_ylabel("BSS (vs seasonal climatology within regime)")
    ax.set_title(f"Skill by ENSO regime — {target}, lead {lead} wk")
    ax.legend()
    ax.grid(alpha=0.3, axis="y")
    fig.tight_layout()
    out = io.FIG_DIR / "skill_by_enso.png"
    fig.savefig(out, dpi=DPI)
    plt.close(fig)
    print(f"[OK] {out.name}")


def fig_permutation(target: str = io.PRIMARY_TARGET, lead: int = 2) -> None:
    imp = _read("permutation_importance.csv")
    if imp is None:
        return
    models = [m for m in ("lgbm", "logistic") if m in imp["model"].unique()]
    if not models:
        return
    fig, axes = plt.subplots(1, len(models), figsize=(6 * len(models), 6), squeeze=False)
    for ax, m in zip(axes[0], models):
        s = imp[(imp.target == target) & (imp.lead == lead) & (imp.model == m)] \
            .sort_values("mean_drop_bss")
        ax.barh(s["feature"], s["mean_drop_bss"], xerr=s["std_drop_bss"],
                color="#d62728", capsize=2)
        ax.set_title(io.MODEL_LABELS.get(m, m))
        ax.set_xlabel("BSS drop when feature permuted")
        ax.grid(alpha=0.3, axis="x")
    fig.suptitle(f"Permutation importance — {target}, lead {lead} wk")
    fig.tight_layout()
    out = io.FIG_DIR / "permutation_importance.png"
    fig.savefig(out, dpi=DPI)
    plt.close(fig)
    print(f"[OK] {out.name}")


def fig_calibration_methods(target: str = io.PRIMARY_TARGET) -> None:
    cm = _read("calibration_methods.csv")
    if cm is None:
        return
    cm = cm[cm.target == target]
    if cm.empty:
        return
    fig, ax = plt.subplots(figsize=(8, 5))
    for recipe, color in (("raw", "#7f7f7f"), ("platt", "#1f77b4"), ("isotonic", "#ff7f0e")):
        s = cm[cm.recipe == recipe].sort_values("lead")
        if s.empty:
            continue
        ax.plot(s.lead, s.bss, marker="o", lw=2, label=recipe, color=color)
    ax.axhline(0, color="black", lw=1)
    ax.set_xlabel("Lead time (weeks)")
    ax.set_ylabel("BSS (vs climatology)")
    ax.set_title(f"LightGBM calibration: raw vs Platt vs Isotonic — {target}")
    ax.set_xticks(sorted(cm.lead.unique()))
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    out = io.FIG_DIR / "calibration_methods_lgbm.png"
    fig.savefig(out, dpi=DPI)
    plt.close(fig)
    print(f"[OK] {out.name}")


def main() -> int:
    io.ensure_dirs()
    print(f"=== figures: เขียนรูปลง {io.FIG_DIR} ===")
    fig_bss_vs_lead()
    fig_skill_by_enso()
    fig_permutation()
    fig_calibration_methods()
    print("[OK] เสร็จ")
    return 0


if __name__ == "__main__":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass
    sys.exit(main())
