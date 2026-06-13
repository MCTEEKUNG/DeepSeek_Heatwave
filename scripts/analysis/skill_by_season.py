"""
Skill by season — โมเดล "ชนะ climatology" ในฤดูไหนบ้าง (คำถามชี้ขาดของการขยายข้อมูลทั้งปี)

คำถาม: หลังขยายข้อมูลเป็นทั้งปี โมเดลมี skill (BSS > 0 เทียบ seasonal climatology)
ตลอดทั้งปีจริง หรือ skill กระจุกเฉพาะฤดูร้อน/El Niño ส่วนมรสุม (เหตุการณ์น้อย) degenerate?
  -> ตัดสิน "เรื่องเล่าของแอป": พยากรณ์ทั้งปีได้จริง vs ครอบคลุมทั้งปีแต่ skill กระจุก

stratify ตามเดือนของ "หน้าต่างเป้าหมาย" (valid month = issue date + lead สัปดาห์)
  เพราะคำถามคือ "ทำนายเหตุการณ์ที่ 'เกิด' ในเดือนนั้นได้แค่ไหน" ไม่ใช่เดือนที่ออกพยากรณ์

อ่าน outputs/predictions.csv อย่างเดียว (ไม่เทรนใหม่) ; reuse brier_skill_score (เทียบ p_clim)
numpy ล้วน. รายงานพร้อม base_rate + n ต่อกลุ่ม -> เห็นกลุ่ม degenerate (base rate ~0) ชัด

ใช้งาน:  python skill_by_season.py        # -> outputs/analysis/skill_by_season/*
         python skill_by_season.py test   # self-test ข้อมูลสังเคราะห์
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

sys.path.insert(0, str(Path(__file__).resolve().parent))
from io_utils import (ANALYSIS_DIR, LEADS, MODEL_LABELS, PRIMARY_TARGET,  # noqa: E402
                      REPORT_MODELS, load_predictions)
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from evaluate import brier_skill_score  # noqa: E402

OUT_DIR = ANALYSIS_DIR / "skill_by_season"

# ฤดูของไทย (ตามเดือนของหน้าต่างเป้าหมาย)
SEASON_OF_MONTH = {2: "hot", 3: "hot", 4: "hot", 5: "hot",
                   6: "rainy", 7: "rainy", 8: "rainy", 9: "rainy", 10: "rainy",
                   11: "cool", 12: "cool", 1: "cool"}
SEASON_TH = {"hot": "ร้อน (ก.พ.-พ.ค.)", "rainy": "ฝน/มรสุม (มิ.ย.-ต.ค.)", "cool": "หนาว (พ.ย.-ม.ค.)"}
SEASON_ORDER = ["hot", "rainy", "cool"]
MONTH_EN = {1: "Jan", 2: "Feb", 3: "Mar", 4: "Apr", 5: "May", 6: "Jun",
            7: "Jul", 8: "Aug", 9: "Sep", 10: "Oct", 11: "Nov", 12: "Dec"}


def add_valid_month(pred: pd.DataFrame) -> pd.DataFrame:
    """เพิ่ม valid_month = เดือนของกลางหน้าต่างเป้าหมาย (issue date + lead สัปดาห์)."""
    out = pred.copy()
    valid_date = out["date"] + pd.to_timedelta(out["lead"] * 7, unit="D")
    out["valid_month"] = valid_date.dt.month
    out["season"] = out["valid_month"].map(SEASON_OF_MONTH)
    return out


def _bss_group(g: pd.DataFrame) -> dict:
    """BSS เทียบ seasonal climatology (p_clim) + base rate + n ของกลุ่มหนึ่ง."""
    y = g["y"].to_numpy(dtype=float)
    p = g["p"].to_numpy(dtype=float)
    pc = g["p_clim"].to_numpy(dtype=float)
    return {"n": int(y.size), "base_rate": round(float(y.mean()), 4),
            "bss": round(brier_skill_score(y, p, baseline_prob=pc), 4)}


def skill_table(pred: pd.DataFrame, by: str, target: str = PRIMARY_TARGET,
                models=None) -> pd.DataFrame:
    """BSS ต่อ (model, <by>) — by = 'season' หรือ 'valid_month'. pool ทุก lead."""
    models = models or REPORT_MODELS
    sub = pred[(pred["target"] == target) & (pred["model"].isin(models))]
    rows = []
    for (model, key), g in sub.groupby(["model", by]):
        rows.append({"model": model, by: key, **_bss_group(g)})
    out = pd.DataFrame(rows)
    if by == "season":
        out["_ord"] = out["season"].map({s: i for i, s in enumerate(SEASON_ORDER)})
        out = out.sort_values(["model", "_ord"]).drop(columns="_ord")
    else:
        out = out.sort_values(["model", by])
    return out.reset_index(drop=True)


def skill_by_season_lead(pred: pd.DataFrame, target: str = PRIMARY_TARGET,
                         models=None) -> pd.DataFrame:
    """BSS ต่อ (model, season, lead) — เผื่อดูว่า skill ฤดู/lead ไหนหาย."""
    models = models or REPORT_MODELS
    sub = pred[(pred["target"] == target) & (pred["model"].isin(models))]
    rows = []
    for (model, season, lead), g in sub.groupby(["model", "season", "lead"]):
        rows.append({"model": model, "season": season, "lead": int(lead), **_bss_group(g)})
    return pd.DataFrame(rows)


def _save_fig(season_tbl: pd.DataFrame, path: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    models = list(season_tbl["model"].unique())
    x = np.arange(len(SEASON_ORDER))
    w = 0.8 / max(len(models), 1)
    fig, ax = plt.subplots(figsize=(8, 4.8))
    for i, m in enumerate(models):
        g = season_tbl[season_tbl["model"] == m].set_index("season").reindex(SEASON_ORDER)
        ax.bar(x + i * w, g["bss"].to_numpy(dtype=float), w, label=MODEL_LABELS.get(m, m))
    ax.axhline(0, color="k", lw=0.8)
    ax.set_xticks(x + w * (len(models) - 1) / 2)
    ax.set_xticklabels(["Hot (Feb-May)", "Rainy (Jun-Oct)", "Cool (Nov-Jan)"])
    ax.set_ylabel("BSS vs seasonal climatology")
    ax.set_title("Forecast skill by season (y_rm, pooled leads) — >0 beats climatology")
    ax.legend(fontsize=8)
    fig.tight_layout(); fig.savefig(path, dpi=130); plt.close(fig)


def run(verbose: bool = True) -> Path:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / "figures").mkdir(exist_ok=True)
    pred = add_valid_month(load_predictions())

    season_tbl = skill_table(pred, by="season")
    month_tbl = skill_table(pred, by="valid_month")
    sl_tbl = skill_by_season_lead(pred)

    season_tbl.to_csv(OUT_DIR / "bss_by_season.csv", index=False)
    month_tbl.to_csv(OUT_DIR / "bss_by_month.csv", index=False)
    sl_tbl.to_csv(OUT_DIR / "bss_by_season_lead.csv", index=False)
    _save_fig(season_tbl, OUT_DIR / "figures" / "bss_by_season.png")

    if verbose:
        print(f"=== Skill by season (y_rm, เทียบ seasonal climatology) ===")
        for m in REPORT_MODELS:
            g = season_tbl[season_tbl["model"] == m]
            if g.empty:
                continue
            print(f"\n{MODEL_LABELS.get(m, m)}:")
            for r in g.itertuples():
                flag = "ชนะ" if r.bss > 0 else "แพ้/เสมอ"
                print(f"  {SEASON_TH[r.season]:24s} BSS={r.bss:+.3f} "
                      f"(base={r.base_rate:.3f}, n={r.n:5d})  {flag}")
        print(f"\n[OK] ผลที่ {OUT_DIR}")
    return OUT_DIR


# ---------------------------------------------------------------- self-test

def _selftest() -> None:
    # 1) add_valid_month: issue 2020-01-01 + lead2 (14 วัน) -> 2020-01-15 = เดือน 1 = cool
    p = pd.DataFrame({"date": pd.to_datetime(["2020-01-01", "2020-03-20"]), "lead": [2, 4],
                      "target": "y_rm", "model": "logistic", "y": 0.0, "p": 0.1, "p_clim": 0.1})
    pv = add_valid_month(p)
    assert pv["valid_month"].tolist() == [1, 4], pv["valid_month"].tolist()
    assert pv["season"].tolist() == ["cool", "hot"], pv["season"].tolist()
    print("[OK] valid_month: issue+lead -> เดือนหน้าต่างเป้าหมาย + ฤดูถูกต้อง")

    # 2) skill_table: โมเดลที่ทำนายดีในฤดูร้อนต้อง BSS>0, ฤดูที่ทำนายมั่วต้อง BSS<0
    rng = np.random.default_rng(0)
    n = 4000
    dates = pd.to_datetime("2015-01-01") + pd.to_timedelta(rng.integers(0, 365 * 5, n), unit="D")
    valid = dates + pd.to_timedelta(14, unit="D")
    season = valid.month.map(SEASON_OF_MONTH)
    y = (rng.random(n) < 0.2).astype(float)
    p_clim = np.full(n, 0.2)
    # ฤดูร้อน: ทำนายตรง y (BSS สูง) ; ฤดูอื่น: ทำนายมั่ว (BSS ติดลบ)
    p = np.where(season.to_numpy() == "hot",
                 np.where(y == 1, 0.9, 0.1), np.where(y == 1, 0.1, 0.9))
    df = pd.DataFrame({"date": dates, "lead": 2, "target": "y_rm", "model": "logistic",
                       "y": y, "p": p, "p_clim": p_clim})
    tbl = skill_table(add_valid_month(df), by="season", models=["logistic"])
    bss_hot = float(tbl.loc[tbl.season == "hot", "bss"].iloc[0])
    bss_rainy = float(tbl.loc[tbl.season == "rainy", "bss"].iloc[0])
    assert bss_hot > 0.5, bss_hot
    assert bss_rainy < 0, bss_rainy
    print(f"[OK] skill_table: ร้อน BSS={bss_hot:+.2f}(>0 ชนะ) ; มรสุม BSS={bss_rainy:+.2f}(<0 แพ้)")

    # 3) base rate ~0 -> เห็นเป็นกลุ่ม degenerate (ไม่ crash)
    g0 = pd.DataFrame({"y": np.zeros(50), "p": np.full(50, 0.1), "p_clim": np.full(50, 0.05)})
    r = _bss_group(g0)
    assert r["base_rate"] == 0.0 and r["n"] == 50
    print("[OK] กลุ่ม base_rate=0 รายงานได้ ไม่ crash (เห็น degenerate)")

    print("[OK] self-test ผ่านทั้งหมด")


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "test":
        _selftest()
    else:
        run()
