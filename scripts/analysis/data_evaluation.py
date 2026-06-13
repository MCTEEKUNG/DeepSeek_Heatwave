"""
Data Evaluation — ประเมิน "ข้อมูลนำเข้า" (data/processed/dataset.csv) ก่อน/ระหว่างสร้างโมเดล

ต่างจากโมดูลอื่นใน analysis/ ที่ประเมิน "ผลโมเดล" (predictions.csv) — ไฟล์นี้ประเมิน
"ตัวข้อมูลเอง": ความครบถ้วน, ช่วงค่า, ฤดูกาลของเหตุการณ์, การกระจายตาม ENSO,
สัญญาณ feature->target, และ multicollinearity (ป้อนการตัดสินใจ feature selection)

หัวใจ (centerpiece): heatwave-day rate ราย "เดือน" + ราย "ENSO regime"
  -> ตอบเชิงประจักษ์ว่า "มีสัญญาณคลื่นความร้อนนอกฤดูร้อน (ก.พ.-พ.ค.) ไหม"
     ซึ่งคือคำถามเบื้องหลังการตัดสินใจขยายข้อมูลเป็นทั้งปี

สำคัญ: ออกแบบให้ rerun ได้ — ตรวจ coverage ของข้อมูลแล้วเลือกโฟลเดอร์ผลลัพธ์เอง
  ข้อมูล ม.ค.-ก.ค.  -> outputs/analysis/data_eval_janjul/   (baseline ชั่วคราว)
  ข้อมูลทั้งปี      -> outputs/analysis/data_eval_fullyear/  (ตัวจริงหลังขยายข้อมูล)
  -> ผลสองชุดไม่ทับกัน + Streamlit (อ่าน analysis/*.csv ราก) ไม่หยิบไปใช้โดยพลาด

numpy ล้วน (ไม่พึ่ง scipy/statsmodels) ; reuse stats.bootstrap_ci, io_utils ENSO

ใช้งาน:  python data_evaluation.py        # อ่าน dataset.csv -> รายงาน + ตาราง + รูป
         python data_evaluation.py test   # self-test กับข้อมูลสังเคราะห์ (ไม่แตะข้อมูลจริง)
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
from io_utils import ROOT, classify_enso, enso_value_for_dates  # noqa: E402
from stats import bootstrap_ci  # noqa: E402

DATASET = ROOT / "data" / "processed" / "dataset.csv"

# 20 feature ที่ป้อนโมเดล (ตรงกับ train.FEATURES) + กลุ่มสำหรับตีความ multicollinearity
FEATURES = [
    "sm1", "sm1_mean7", "sm1_mean30", "sm1_trend",
    "sm3", "sm3_mean7", "sm3_mean30", "sm3_trend",
    "tmax_rm", "tmax_mean7", "in_hw_today", "hot_frac7",
    "mjo_rmm1", "mjo_rmm2", "mjo_amp", "mjo_sin", "mjo_cos",
    "nino34_lag1m", "doy_sin", "doy_cos",
]
FEATURE_GROUP = {
    "sm1": "Soil", "sm1_mean7": "Soil", "sm1_mean30": "Soil", "sm1_trend": "Soil",
    "sm3": "Soil", "sm3_mean7": "Soil", "sm3_mean30": "Soil", "sm3_trend": "Soil",
    "tmax_rm": "Thermal", "tmax_mean7": "Thermal", "in_hw_today": "Thermal",
    "hot_frac7": "Thermal", "mjo_rmm1": "MJO", "mjo_rmm2": "MJO", "mjo_amp": "MJO",
    "mjo_sin": "MJO", "mjo_cos": "MJO", "nino34_lag1m": "ENSO",
    "doy_sin": "Seasonal", "doy_cos": "Seasonal",
}
# NaN ที่ "เป็นโครงสร้าง" (rolling/lookback ข้ามช่องว่างปีไม่ได้) — ไม่ใช่ data defect
STRUCTURAL_NAN_FEATURES = {
    "sm1_mean7", "sm1_mean30", "sm1_trend", "sm3_mean7", "sm3_mean30", "sm3_trend",
    "tmax_mean7", "hot_frac7",
}
LEADS = [2, 3, 4, 5, 6]
PRIMARY_TARGET = "y_rm"
MONTH_TH = {1: "ม.ค.", 2: "ก.พ.", 3: "มี.ค.", 4: "เม.ย.", 5: "พ.ค.", 6: "มิ.ย.",
            7: "ก.ค.", 8: "ส.ค.", 9: "ก.ย.", 10: "ต.ค.", 11: "พ.ย.", 12: "ธ.ค."}
# label อังกฤษสำหรับรูป (ตรง convention figures เดิม + กันปัญหาฟอนต์ไทยใน matplotlib)
MONTH_EN = {1: "Jan", 2: "Feb", 3: "Mar", 4: "Apr", 5: "May", 6: "Jun",
            7: "Jul", 8: "Aug", 9: "Sep", 10: "Oct", 11: "Nov", 12: "Dec"}


# ----------------------------------------------------------------- helpers

def coverage_label(df: pd.DataFrame) -> str:
    """เลือก label/โฟลเดอร์ผลลัพธ์ตาม coverage เดือนของข้อมูล (กัน rerun ทับกัน)."""
    months = set(pd.DatetimeIndex(df.index).month)
    return "fullyear" if months >= set(range(1, 13)) else "janjul"


def out_dir_for(df: pd.DataFrame) -> Path:
    d = ROOT / "outputs" / "analysis" / f"data_eval_{coverage_label(df)}"
    (d / "figures").mkdir(parents=True, exist_ok=True)
    return d


def _mean(a: np.ndarray) -> float:
    return float(np.mean(a))


def pearson(x: np.ndarray, y: np.ndarray) -> float:
    """สหสัมพันธ์ Pearson (ใช้กับ feature vs target binary = point-biserial)."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    m = ~(np.isnan(x) | np.isnan(y))
    x, y = x[m], y[m]
    if x.size < 3 or x.std() == 0 or y.std() == 0:
        return float("nan")
    return float(np.corrcoef(x, y)[0, 1])


# ----------------------------------------------------------- 1) coverage

def coverage_table(df: pd.DataFrame) -> pd.DataFrame:
    """ต่อปี: จำนวนวัน, เดือนที่มี, วันแรก/สุดท้าย — เปิดให้เห็นข้อจำกัด ม.ค.-ก.ค. ชัดเจน."""
    idx = pd.DatetimeIndex(df.index)
    rows = []
    for yr, g in df.groupby(idx.year):
        gi = pd.DatetimeIndex(g.index)
        rows.append({
            "year": int(yr), "n_days": len(g),
            "months": ",".join(str(m) for m in sorted(set(gi.month))),
            "first": str(gi.min().date()), "last": str(gi.max().date()),
        })
    return pd.DataFrame(rows)


# --------------------------------------------------- 2) missingness

def missingness_table(df: pd.DataFrame) -> pd.DataFrame:
    """ต่อ feature: จำนวน/สัดส่วน NaN + ป้ายว่าเป็น 'structural' (lookback) หรือไม่.

    NaN เชิงโครงสร้างเกิดจาก rolling window เอื้อมข้ามช่องว่างระหว่างปี (ส.ค.-ธ.ค. หาย)
    ไม่ได้ -> ไม่ใช่ข้อมูลเสีย และจะ "หายเอง" เมื่อข้อมูลต่อเนื่องทั้งปี (เลขที่ไวต่อ before/after สุด).
    """
    n = len(df)
    rows = []
    for c in FEATURES:
        nmiss = int(df[c].isna().sum())
        rows.append({
            "feature": c, "n_missing": nmiss, "pct_missing": round(100 * nmiss / n, 2),
            "kind": "structural" if c in STRUCTURAL_NAN_FEATURES else "data",
        })
    return pd.DataFrame(rows).sort_values("n_missing", ascending=False).reset_index(drop=True)


# ----------------------------------------------- 3) feature ranges

def feature_ranges(df: pd.DataFrame) -> pd.DataFrame:
    """ช่วงค่าต่อ feature (ยืนยัน physical plausibility — หน่วยผ่านด่าน units_utils มาแล้ว)."""
    rows = []
    for c in FEATURES:
        s = df[c].dropna()
        rows.append({
            "feature": c, "n": int(s.size), "min": float(s.min()), "p01": float(s.quantile(0.01)),
            "median": float(s.median()), "mean": float(s.mean()),
            "p99": float(s.quantile(0.99)), "max": float(s.max()), "std": float(s.std()),
        })
    return pd.DataFrame(rows)


# ---------------------------- 4) heatwave-day rate by month (CENTERPIECE)

def heatwave_rate_by_month(df: pd.DataFrame) -> pd.DataFrame:
    """สัดส่วนวันที่เป็น 'วันคลื่นความร้อน' (in_hw_today) ต่อเดือนปฏิทิน + CI (block bootstrap).

    in_hw_today = วันนั้นอยู่ในช่วง heatwave จริง (regional-mean p90, ติดต่อ >=3 วัน)
    -> ฤดูกาลของ "ตัวปรากฏการณ์" โดยตรง ; ถ้าข้อมูลทั้งปี จะเห็นทันทีว่ามีเหตุการณ์
       นอกฤดูร้อนหรือไม่ (เหตุผลของการขยายข้อมูล)
    """
    idx = pd.DatetimeIndex(df.index)
    rows = []
    for m in range(1, 13):
        y = df.loc[idx.month == m, "in_hw_today"].dropna().to_numpy(dtype=float)
        if y.size == 0:
            continue
        ci = bootstrap_ci(_mean, (y,), L=28, B=1000)
        rows.append({
            "month": m, "month_th": MONTH_TH[m], "n_days": int(y.size),
            "hw_day_rate": round(float(y.mean()), 4),
            "ci_lo": round(ci["lo"], 4), "ci_hi": round(ci["hi"], 4),
        })
    return pd.DataFrame(rows)


# ------------------------ 5) target base rate by lead x ENSO regime

def base_rate_by_regime(df: pd.DataFrame, target: str = PRIMARY_TARGET) -> pd.DataFrame:
    """base rate ของ target (ราย lead) แยกตาม ENSO regime ของ 'วันออกพยากรณ์'.

    ระดับข้อมูล (ไม่ใช่ระดับโมเดล) — บอกว่าความถี่เหตุการณ์เลื่อนตาม El Niño/La Niña แค่ไหน
    (สอดคล้องข้อค้นพบเดิม: El Niño เสี่ยงสูง, La Niña เสี่ยงต่ำ).
    """
    regime = classify_enso(enso_value_for_dates(df.index, use="lag1m"))
    rows = []
    for L in LEADS:
        col = f"{target}_l{L}"
        sub = df[[col]].copy()
        sub["regime"] = regime
        sub = sub.dropna(subset=[col])
        for reg in ("elnino", "neutral", "lanina"):
            yv = sub.loc[sub["regime"] == reg, col].to_numpy(dtype=float)
            rows.append({
                "lead": L, "regime": reg, "n": int(yv.size),
                "base_rate": round(float(yv.mean()), 4) if yv.size else float("nan"),
            })
    return pd.DataFrame(rows)


def base_rate_by_lead(df: pd.DataFrame, target: str = PRIMARY_TARGET) -> pd.DataFrame:
    """base rate รวมต่อ lead (ภาพ class imbalance) — รับทุก target ที่มี."""
    rows = []
    for tgt in ("y_rm", "y_rm95", "y_af"):
        for L in LEADS:
            col = f"{tgt}_l{L}"
            if col not in df.columns:
                continue
            y = df[col].dropna().to_numpy(dtype=float)
            rows.append({"target": tgt, "lead": L, "n": int(y.size),
                         "n_events": int(y.sum()), "base_rate": round(float(y.mean()), 4)})
    return pd.DataFrame(rows)


# --------------------------- 6) feature -> target signal (point-biserial)

def feature_target_corr(df: pd.DataFrame, target: str = PRIMARY_TARGET,
                        lead: int = 2) -> pd.DataFrame:
    """point-biserial (Pearson) ของแต่ละ feature กับ target — สัญญาณเชิงเส้นอย่างหยาบ."""
    col = f"{target}_l{lead}"
    y = df[col].to_numpy(dtype=float)
    rows = []
    for c in FEATURES:
        r = pearson(df[c].to_numpy(dtype=float), y)
        rows.append({"feature": c, "group": FEATURE_GROUP[c], "corr": round(r, 4),
                     "abs_corr": round(abs(r), 4)})
    return pd.DataFrame(rows).sort_values("abs_corr", ascending=False).reset_index(drop=True)


# ------------------------------------ 7) multicollinearity (VIF + corr)

def vif_table(df: pd.DataFrame) -> pd.DataFrame:
    """Variance Inflation Factor ต่อ feature = diag(inv(corr)) — สูง = ซ้ำซ้อนกับตัวอื่น.

    ใช้ pseudo-inverse กันเมทริกซ์ near-singular (soil moisture สหสัมพันธ์สูงมาก).
    VIF>10 = multicollinearity รุนแรง (ป้อนการตัดสินใจ feature selection: logistic
    เคย overfit soil/thermal ที่ซ้ำซ้อน -> ตัดทิ้งแล้วดีขึ้น).
    """
    X = df[FEATURES].dropna().to_numpy(dtype=float)
    C = np.corrcoef(X, rowvar=False)
    inv = np.linalg.pinv(C)
    vif = np.diag(inv)
    out = pd.DataFrame({"feature": FEATURES, "group": [FEATURE_GROUP[f] for f in FEATURES],
                        "vif": np.round(vif, 2)})
    return out.sort_values("vif", ascending=False).reset_index(drop=True)


def corr_matrix(df: pd.DataFrame) -> pd.DataFrame:
    X = df[FEATURES].dropna()
    return X.corr()


# ------------------------------------------------------------- figures

def _save_month_fig(monthly: pd.DataFrame, path: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(8, 4.5))
    x = monthly["month"]
    yerr = [monthly["hw_day_rate"] - monthly["ci_lo"], monthly["ci_hi"] - monthly["hw_day_rate"]]
    ax.bar(x, monthly["hw_day_rate"], color="#d62728", yerr=yerr, capsize=3)
    ax.set_xticks(list(monthly["month"]))
    ax.set_xticklabels([MONTH_EN[m] for m in monthly["month"]])
    ax.set_ylabel("heatwave-day fraction (in_hw_today)")
    ax.set_title("Heatwave-day rate by month — 95% CI (block bootstrap)")
    fig.tight_layout(); fig.savefig(path, dpi=130); plt.close(fig)


def _save_regime_fig(reg: pd.DataFrame, path: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(8, 4.5))
    colors = {"elnino": "#d62728", "neutral": "#7f7f7f", "lanina": "#1f77b4"}
    for reg_name in ("elnino", "neutral", "lanina"):
        g = reg[reg["regime"] == reg_name]
        ax.plot(g["lead"], g["base_rate"], "o-", color=colors[reg_name], label=reg_name)
    ax.set_xlabel("lead (weeks)"); ax.set_ylabel("base rate (y_rm)")
    ax.set_title("Event base rate by lead, stratified by ENSO regime")
    ax.legend(); fig.tight_layout(); fig.savefig(path, dpi=130); plt.close(fig)


def _save_corr_fig(ftc: pd.DataFrame, path: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    g = ftc.sort_values("corr")
    colors = {"Soil": "#2ca02c", "ENSO": "#1f77b4", "MJO": "#9467bd",
              "Thermal": "#d62728", "Seasonal": "#ff7f0e"}
    fig, ax = plt.subplots(figsize=(7, 6))
    ax.barh(g["feature"], g["corr"], color=[colors.get(x, "#777") for x in g["group"]])
    ax.set_xlabel("point-biserial corr with y_rm lead 2"); ax.axvline(0, color="k", lw=0.6)
    ax.set_title("Linear feature -> target signal")
    fig.tight_layout(); fig.savefig(path, dpi=130); plt.close(fig)


# -------------------------------------------------------------- report

def _md_table(df: pd.DataFrame) -> str:
    cols = list(df.columns)
    head = "| " + " | ".join(cols) + " |"
    sep = "| " + " | ".join("---" for _ in cols) + " |"
    body = "\n".join("| " + " | ".join(str(v) for v in row) + " |"
                     for row in df.itertuples(index=False))
    return "\n".join([head, sep, body])


def run(verbose: bool = True) -> Path:
    df = pd.read_csv(DATASET, parse_dates=["date"], index_col="date").sort_index()
    label = coverage_label(df)
    out = out_dir_for(df)
    fig = out / "figures"

    cov = coverage_table(df)
    miss = missingness_table(df)
    rng = feature_ranges(df)
    monthly = heatwave_rate_by_month(df)
    reg = base_rate_by_regime(df)
    blead = base_rate_by_lead(df)
    ftc = feature_target_corr(df)
    vif = vif_table(df)
    cmat = corr_matrix(df)

    # ตาราง CSV
    cov.to_csv(out / "coverage.csv", index=False)
    miss.to_csv(out / "missingness.csv", index=False)
    rng.to_csv(out / "feature_ranges.csv", index=False)
    monthly.to_csv(out / "hw_rate_by_month.csv", index=False)
    reg.to_csv(out / "base_rate_by_regime.csv", index=False)
    blead.to_csv(out / "base_rate_by_lead.csv", index=False)
    ftc.to_csv(out / "feature_target_corr.csv", index=False)
    vif.to_csv(out / "vif.csv", index=False)
    cmat.round(3).to_csv(out / "corr_matrix.csv")

    # รูป
    _save_month_fig(monthly, fig / "hw_rate_by_month.png")
    _save_regime_fig(reg, fig / "base_rate_by_regime.png")
    _save_corr_fig(ftc, fig / "feature_target_corr.png")

    # รายงาน markdown
    provisional = (label == "janjul")
    banner = ("> ⚠️ **ชั่วคราว (ม.ค.-ก.ค. เท่านั้น)** — ตัวเลขในไฟล์นี้เป็น baseline ก่อนขยาย"
              "ข้อมูลเป็นทั้งปี ใช้ตรวจโค้ด/เทียบ before-after เท่านั้น **อย่าเพิ่งสรุปผลเชิงวิจัย**\n"
              if provisional else
              "> ✅ ข้อมูลทั้งปี (ม.ค.-ธ.ค.) — ชุดประเมินตัวจริง\n")
    enso_known = reg[reg["regime"] != "unknown"]
    top_feats = ", ".join(f"{r.feature} ({r.corr:+.2f})" for r in ftc.head(5).itertuples())
    high_vif = vif[vif["vif"] > 10]
    md = f"""# Data Evaluation — {label}

{banner}
ข้อมูล: `{DATASET.name}` | {len(df):,} วัน | {pd.DatetimeIndex(df.index).year.nunique()} ปี | \
ช่วง {df.index.min().date()} ถึง {df.index.max().date()}

## 1. ความครบถ้วน (coverage)
เดือนที่มีข้อมูล: {sorted(set(pd.DatetimeIndex(df.index).month))}
{_md_table(cov.head(5))}
... (ดู `coverage.csv` ครบทุกปี)

## 2. ค่าหาย (missingness)
NaN เชิงโครงสร้าง (lookback ข้ามช่องว่างปีไม่ได้) — ไม่ใช่ข้อมูลเสีย, หายเองเมื่อข้อมูลต่อเนื่อง:
{_md_table(miss.head(8))}

## 3. ฤดูกาลของคลื่นความร้อน (centerpiece)
สัดส่วนวันคลื่นความร้อน (`in_hw_today`) ต่อเดือน — ตอบว่ามีสัญญาณนอกฤดูร้อนไหม:
{_md_table(monthly)}

![hw rate by month](figures/hw_rate_by_month.png)

## 4. base rate ต่อ ENSO regime (ระดับข้อมูล)
{_md_table(enso_known)}

![base rate by regime](figures/base_rate_by_regime.png)

## 5. class imbalance (base rate ต่อ lead)
{_md_table(blead[blead.target == PRIMARY_TARGET])}

## 6. สัญญาณ feature -> target (point-biserial vs y_rm lead 2)
Top 5: {top_feats}
{_md_table(ftc.head(8))}

![feature-target corr](figures/feature_target_corr.png)

## 7. Multicollinearity (VIF)
feature ที่ VIF > 10 (ซ้ำซ้อนรุนแรง — ป้อนการตัดสินใจ feature selection):
{_md_table(high_vif) if len(high_vif) else '(ไม่มี VIF > 10)'}

ดูตารางสหสัมพันธ์เต็มที่ `corr_matrix.csv`
"""
    (out / "report.md").write_text(md, encoding="utf-8")

    if verbose:
        print(f"=== Data Evaluation [{label}] : {len(df):,} วัน ===")
        print(f"เดือนที่มี: {sorted(set(pd.DatetimeIndex(df.index).month))}")
        print("\nฤดูกาลคลื่นความร้อน (in_hw_today rate ต่อเดือน):")
        for r in monthly.itertuples():
            bar = "█" * int(r.hw_day_rate * 100)
            print(f"  {r.month_th:>5s} {r.hw_day_rate:.3f} {bar}")
        print("\nbase rate y_rm ต่อ ENSO regime (lead 2):")
        for r in reg[(reg.lead == 2) & (reg.regime != "unknown")].itertuples():
            print(f"  {r.regime:8s} n={r.n:5d}  base_rate={r.base_rate}")
        print(f"\nTop feature->target: {top_feats}")
        print(f"VIF > 10: {list(high_vif['feature']) if len(high_vif) else 'ไม่มี'}")
        print(f"\n[OK] ผลทั้งหมดที่ {out}")
    return out


# ---------------------------------------------------------------- self-test

def _selftest() -> None:
    rng = np.random.default_rng(11)
    dates = []
    for yr in range(2005, 2020):
        dates.extend(pd.date_range(f"{yr}-01-01", f"{yr}-07-31", freq="D"))
    idx = pd.DatetimeIndex(dates)
    n = len(idx)
    doy = idx.dayofyear.to_numpy()
    df = pd.DataFrame(index=idx)
    for c in FEATURES:
        df[c] = rng.normal(0, 1, n)
    df.index.name = "date"
    # in_hw_today ขึ้นกับฤดู (พีค เม.ย.) -> heatwave_rate_by_month ต้องเห็นพีคนั้น
    season = np.exp(-((doy - 105) ** 2) / (2 * 25 ** 2))
    df["in_hw_today"] = (rng.random(n) < 0.5 * season).astype(float)
    # nino34 ให้เป็นสัญญาณจริงของ target -> feature_target_corr ต้องจับได้
    df["nino34_lag1m"] = rng.normal(0, 1, n)
    for L in LEADS:
        df[f"y_rm_l{L}"] = (rng.random(n) < 1 / (1 + np.exp(-(df["nino34_lag1m"] - 1)))).astype(float)
        df[f"y_rm95_l{L}"] = df[f"y_rm_l{L}"]
        df[f"y_af_l{L}"] = df[f"y_rm_l{L}"]
    # ทำ sm1_mean30 ซ้ำกับ sm1 เกือบเป๊ะ -> VIF ต้องสูง
    df["sm1_mean30"] = df["sm1"] + rng.normal(0, 0.01, n)

    # 1) coverage label
    assert coverage_label(df) == "janjul"
    full = df.copy()
    full2 = pd.concat([full, full.assign(**{}).set_index(
        pd.DatetimeIndex(full.index) + pd.Timedelta(days=200))])
    assert coverage_label(full2) == "fullyear"
    print("[OK] coverage_label: ม.ค.-ก.ค.->janjul, ครบ 12 เดือน->fullyear")

    # 2) heatwave_rate_by_month: เห็นพีค เม.ย. (เดือน 4) สูงกว่า ม.ค. (เดือน 1)
    monthly = heatwave_rate_by_month(df)
    r4 = float(monthly.loc[monthly.month == 4, "hw_day_rate"].iloc[0])
    r1 = float(monthly.loc[monthly.month == 1, "hw_day_rate"].iloc[0])
    assert r4 > r1, (r4, r1)
    assert (monthly["ci_lo"] <= monthly["hw_day_rate"]).all()
    assert (monthly["hw_day_rate"] <= monthly["ci_hi"]).all()
    print(f"[OK] heatwave_rate_by_month: พีค เม.ย.={r4:.3f} > ม.ค.={r1:.3f} + CI ครอบ point")

    # 3) feature_target_corr: nino34_lag1m ต้องติด top (เป็นสัญญาณจริง)
    ftc = feature_target_corr(df)
    top3 = set(ftc.head(3)["feature"])
    assert "nino34_lag1m" in top3, ftc.head(5)
    print(f"[OK] feature_target_corr: nino34_lag1m ติด top3 ({sorted(top3)})")

    # 4) VIF: sm1/sm1_mean30 ที่ซ้ำกันต้อง VIF สูง
    vif = vif_table(df)
    sm1_vif = float(vif.loc[vif.feature == "sm1", "vif"].iloc[0])
    assert sm1_vif > 10, sm1_vif
    print(f"[OK] VIF จับ multicollinearity: sm1 VIF={sm1_vif:.1f} (>10)")

    # 5) missingness: structural ติดป้ายถูก
    miss = missingness_table(df)
    assert miss.loc[miss.feature == "sm1_mean30", "kind"].iloc[0] == "structural"
    assert miss.loc[miss.feature == "nino34_lag1m", "kind"].iloc[0] == "data"
    print("[OK] missingness: ป้าย structural/data ถูกต้อง")

    print("[OK] self-test ผ่านทั้งหมด")


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "test":
        _selftest()
    else:
        run()
