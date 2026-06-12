"""
เฟส 6 (ส่วนตาราง) — รวมผลวิเคราะห์เป็น "ตารางสรุปหลัก" สำหรับลงบท Results

รวม: metrics_pooled.csv + bootstrap_ci.csv (BSS CI) + paired_tests.csv (q-value)
  -> results_master.csv (long-form ครบทุก target/lead/model)
  -> results_master.md  (โฟกัส y_rm + โมเดลที่ใช้รายงาน + สรุป ablation/calibration)

ตัดสิน "สกิลจริง":
  beats_clim    = BSS 95% CI ไม่คร่อม 0
  beats_persist = q (paired vs persistence, หลัง BH-FDR) < 0.05

ใช้งาน:  python make_report.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import io_utils as io  # noqa: E402

KEYS = ["target", "lead", "model"]


def _read(name: str):
    p = io.ANALYSIS_DIR / name
    return pd.read_csv(p) if p.exists() else None


def build_master() -> pd.DataFrame:
    pooled = io.load_pooled()                       # target,lead,model,n,base_rate,brier,bss,auc
    master = pooled.copy()

    ci = _read("bootstrap_ci.csv")
    if ci is not None:
        bss_ci = ci[ci.metric == "bss"][["target", "lead", "model", "lo95", "hi95"]]
        master = master.merge(bss_ci.rename(columns={"lo95": "bss_lo95", "hi95": "bss_hi95"}),
                              on=KEYS, how="left")

    paired = _read("paired_tests.csv")
    if paired is not None:
        for ref, suf in (("climatology", "clim"), ("persistence", "persist")):
            sub = (paired[paired.reference == ref][["target", "lead", "model", "p_boot", "q_boot"]]
                   .rename(columns={"p_boot": f"p_vs_{suf}", "q_boot": f"q_vs_{suf}"}))
            master = master.merge(sub, on=KEYS, how="left")

    if "bss_lo95" in master:
        master["beats_clim"] = master["bss_lo95"] > 0
    if "q_vs_persist" in master:
        master["beats_persist"] = master["q_vs_persist"] < 0.05
    return master.sort_values(KEYS).reset_index(drop=True)


def df_to_md(df: pd.DataFrame, floatfmt: int = 3) -> str:
    """ตาราง markdown โดยไม่พึ่ง tabulate (ไม่อยู่ใน requirements)."""
    def fmt(v):
        if isinstance(v, (float, np.floating)):
            return "" if pd.isna(v) else f"{v:.{floatfmt}f}"
        if isinstance(v, (bool, np.bool_)):
            return "✓" if v else "·"
        return str(v)
    cols = list(df.columns)
    head = "| " + " | ".join(str(c) for c in cols) + " |"
    sep = "| " + " | ".join("---" for _ in cols) + " |"
    body = ["| " + " | ".join(fmt(r[c]) for c in cols) + " |" for _, r in df.iterrows()]
    return "\n".join([head, sep, *body])


def _bss_ci_str(r) -> str:
    if "bss_lo95" in r and pd.notna(r.get("bss_lo95")):
        return f"{r['bss']:+.3f} [{r['bss_lo95']:+.3f}, {r['bss_hi95']:+.3f}]"
    return f"{r['bss']:+.3f}"


def build_md(master: pd.DataFrame) -> str:
    t = io.PRIMARY_TARGET
    out = ["# ผลการวิเคราะห์ robustness — สรุปหลัก",
           "",
           f"target หลัก = `{t}` (regional-mean p90, heatwave >=3 วันติด) ; "
           "BSS เทียบ seasonal climatology ; CI = moving-block bootstrap 95% ; "
           "q = paired test หลัง BH-FDR",
           ""]

    # --- ตารางหลัก: y_rm, โมเดลรายงาน + baseline ---
    show_models = io.REPORT_MODELS + ["lgbm", "persistence"]
    sub = master[(master.target == t) & (master.model.isin(show_models))].copy()
    rows = []
    for _, r in sub.iterrows():
        rows.append({
            "lead": int(r["lead"]), "model": io.MODEL_LABELS.get(r["model"], r["model"]),
            "BSS [95% CI]": _bss_ci_str(r), "Brier": r["brier"], "AUC": r["auc"],
            "q vs clim": r.get("q_vs_clim", np.nan), "q vs persist": r.get("q_vs_persist", np.nan),
            "beats clim": bool(r.get("beats_clim", False)),
            "beats persist": bool(r.get("beats_persist", False)),
        })
    tbl = pd.DataFrame(rows).sort_values(["lead", "model"])
    out += ["## ตารางหลัก (BSS + ความไม่แน่นอน + นัยสำคัญ)", "", df_to_md(tbl), ""]

    # --- ablation ---
    abl = _read("ablation.csv")
    if abl is not None:
        a = abl[(abl.target == t) & (abl.feature_set.str.startswith("drop_"))]
        piv = a.pivot_table(index="feature_set", columns="lead",
                            values="delta_bss_vs_full", aggfunc="mean").reset_index()
        piv.columns = [c if isinstance(c, str) else f"lead{c}" for c in piv.columns]
        out += ["## Feature-group ablation — delta BSS เทียบ full (ติดลบ = กลุ่มสำคัญ)",
                "", "_leave-one-group-out (เฉลี่ยข้ามโมเดลที่รายงาน)_", "",
                df_to_md(piv), ""]

    # --- calibration decomposition ---
    dec = _read("calibration_decomp.csv")
    if dec is not None:
        d = dec[(dec.target == t) & (dec.lead == 2)][["model", "REL", "RES", "UNC", "brier", "ece"]]
        d = d.copy()
        d["model"] = d["model"].map(lambda m: io.MODEL_LABELS.get(m, m))
        out += ["## Brier decomposition — y_rm lead 2 (REL ต่ำ=calibrate ดี, RES สูง=แยกแยะดี)",
                "", df_to_md(d.sort_values("brier")), ""]

    # --- regime ---
    reg = _read("regime_by_enso.csv")
    if reg is not None:
        rr = reg[(reg.target == t) & (reg.use == "lag1m") & (reg.model == "climatology")]
        piv = rr.pivot_table(index="regime", columns="lead", values="base_rate").reset_index()
        piv.columns = [c if isinstance(c, str) else f"lead{c}" for c in piv.columns]
        out += ["## Base rate ของ heatwave แยกตาม ENSO regime (เห็น base-rate shift)",
                "", df_to_md(piv), ""]
    return "\n".join(out)


def main() -> int:
    io.ensure_dirs()
    master = build_master()
    master.to_csv(io.ANALYSIS_DIR / "results_master.csv", index=False)
    print(f"[OK] results_master.csv : {len(master)} แถว")

    md = build_md(master)
    (io.ANALYSIS_DIR / "results_master.md").write_text(md, encoding="utf-8")
    print(f"[OK] results_master.md")

    # echo ตารางหลักลงคอนโซล
    t = io.PRIMARY_TARGET
    cols = [c for c in ("lead", "model", "bss", "bss_lo95", "bss_hi95",
                        "q_vs_clim", "q_vs_persist", "beats_persist") if c in master]
    view = master[(master.target == t) & (master.model.isin(io.REPORT_MODELS))][cols]
    print(f"\n=== สรุป {t} (โมเดลรายงาน) ===")
    print(view.to_string(index=False))
    print(f"\n[OK] ผลอยู่ที่ {io.ANALYSIS_DIR}")
    return 0


if __name__ == "__main__":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass
    sys.exit(main())
