"""
เทรน + ประเมินผลโมเดลพยากรณ์ความน่าจะเป็นของ heatwave ราย lead time (2-6 สัปดาห์)

ระเบียบวิธี (ผูกกับ design spec):
  - CV: RollingOriginCV (expanding) gap = 49 วัน >= lead สูงสุด (6 สัปดาห์) + หน้าต่างเป้าหมาย
  - baseline: seasonal climatology (= อ้างอิงของ BSS เสมอ) + conditional persistence
  - โมเดลหลัก (ไม่ถ่วงน้ำหนัก): logistic (ครอบ StandardScaler), lgbm
  - ablation (ถ่วงน้ำหนัก): logistic_balanced, lgbm_balanced, balanced_rf
    -> ต้อง recalibrate (Platt) บน validation block ที่ "แยกตามเวลา" จากท้ายชุด train
       (เว้น gap เท่ากันกับ CV) เพราะ CalibratedClassifierCV ใช้ KFold ภายในซึ่ง leak
  - รายงาน: BSS (เทียบ seasonal climatology จาก train), AUC, Brier, reliability diagram

outputs/
  metrics_folds.csv    เมตริกราย fold
  metrics_pooled.csv   เมตริกรวมทุก fold (pooled predictions)
  predictions.csv      คำทำนายรายวัน (target, lead, model, date, y, p)
  figures/reliability_lead{L}.png , feature_importance.png

ใช้งาน:  python train.py          # รันเต็ม (ทุก target, ทุก lead)
         python train.py test     # self-test กับข้อมูลสังเคราะห์
"""
from __future__ import annotations

import sys
from pathlib import Path

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

import warnings

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

# cosmetic: sklearn เตือนเรื่อง feature names เมื่อปนการ fit/predict ด้วย numpy array
warnings.filterwarnings("ignore", message="X does not have valid feature names")
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parent))
from cv import RollingOriginCV
from evaluate import (
    evaluate_probabilistic,
    predict_seasonal_climatology,
    persistence_probs,
    reliability_curve,
)
from models import get_model

DATASET = Path(__file__).resolve().parent.parent / "data" / "processed" / "dataset.csv"
OUT_DIR = Path(__file__).resolve().parent.parent / "outputs"
FIG_DIR = OUT_DIR / "figures"

FEATURES = [
    "sm1", "sm1_mean7", "sm1_mean30", "sm1_trend",
    "sm3", "sm3_mean7", "sm3_mean30", "sm3_trend",
    "tmax_rm", "tmax_mean7", "in_hw_today", "hot_frac7",
    "mjo_rmm1", "mjo_rmm2", "mjo_amp", "mjo_sin", "mjo_cos",
    "nino34_lag1m", "doy_sin", "doy_cos",
]
LEADS = [2, 3, 4, 5, 6]
TARGETS = ["y_rm", "y_rm95", "y_af"]   # หลัก, ablation p95, ablation area-fraction
PRIMARY_TARGET = "y_rm"
MAIN_MODELS = ["logistic", "lgbm"]
ABLATION_MODELS = ["logistic_balanced", "lgbm_balanced", "balanced_rf"]
# โมเดลหลักที่ probability ดิบมัก miscalibrate บนเหตุการณ์หายาก (พบจริง: lgbm ดิบ
# AUC ดีแต่ BSS ติดลบ) -> เพิ่มเวอร์ชัน recalibrate ด้วยกลไกเดียวกับ ablation
CALIBRATED_MAIN = ["lgbm"]
GAP = 49            # วัน (lead 6 สัปดาห์ = 42 + หน้าต่าง 7)
N_SPLITS = 5
TEST_SIZE = 300     # ~2 ปีของวันออกพยากรณ์ที่ใช้ได้
CALIB_FRAC = 0.2    # สัดส่วนท้ายชุด train ที่กันไว้ recalibrate (ablation)


def make_estimator(name: str):
    """โมเดลจากทะเบียน ; logistic ครอบ StandardScaler (สเกล fit จาก train เท่านั้น)."""
    model = get_model(name)
    if name.startswith("logistic"):
        return make_pipeline(StandardScaler(), model)
    return model


class PlattCalibrator:
    """Platt scaling: logistic regression บน log-odds ของความน่าจะเป็นดิบ.

    ใช้กับโมเดลถ่วงน้ำหนักซึ่ง probability เฟ้อ — fit บน validation block
    ที่แยกตามเวลา (ไม่ใช่ KFold) เพื่อไม่ leak ข้าม time series
    """

    def fit(self, p_raw: np.ndarray, y: np.ndarray) -> "PlattCalibrator":
        z = self._logit(p_raw).reshape(-1, 1)
        self._lr = LogisticRegression(max_iter=1000).fit(z, y)
        return self

    def transform(self, p_raw: np.ndarray) -> np.ndarray:
        z = self._logit(p_raw).reshape(-1, 1)
        return self._lr.predict_proba(z)[:, 1]

    @staticmethod
    def _logit(p: np.ndarray) -> np.ndarray:
        p = np.clip(np.asarray(p, dtype=float), 1e-6, 1 - 1e-6)
        return np.log(p / (1 - p))


def fit_predict_calibrated(name: str, X_tr, y_tr, X_te) -> np.ndarray | None:
    """เทรนโมเดล ablation บน train ส่วนต้น + recalibrate บน block ท้าย (เว้น gap).

    คืน None ถ้า train สั้นเกินไป หรือ calibration block มีคลาสเดียว (calibrate ไม่ได้)
    """
    n = len(y_tr)
    n_calib = int(n * CALIB_FRAC)
    n_core = n - n_calib - GAP
    if n_core < 300 or n_calib < 100:
        return None
    y_cal = y_tr[n - n_calib:]
    if len(np.unique(y_cal)) < 2:
        return None
    model = make_estimator(name)
    model.fit(X_tr[:n_core], y_tr[:n_core])
    cal = PlattCalibrator().fit(model.predict_proba(X_tr[n - n_calib:])[:, 1], y_cal)
    return cal.transform(model.predict_proba(X_te)[:, 1])


def run_target_lead(df: pd.DataFrame, target: str, lead: int,
                    verbose: bool = True, features: list[str] | None = None,
                    models: list[str] | None = None) -> tuple[pd.DataFrame, pd.DataFrame]:
    """รัน CV ครบทุกโมเดล/บาสไลน์ของ (target, lead) หนึ่งคู่.

    คืน (per-fold metrics, per-day predictions)

    features: ชุดฟีเจอร์ที่ป้อนโมเดล (None = FEATURES ครบ) — ใช้ทำ ablation รายกลุ่ม.
              เลือกแถวด้วย dropna บน FEATURES "ครบ" เสมอ -> ทุก ablation ใช้แถวเดียวกัน
              (เทียบยุติธรรม ต่างกันแค่คอลัมน์ที่ป้อนโมเดล).
    models:   จำกัดรายชื่อโมเดลที่จะเทรน (None = ครบ) — เร่ง ablation.
              baseline climatology/persistence ผลิตเสมอ (reference ของ BSS และราคาถูก).
    """
    feats = list(features) if features is not None else FEATURES
    want = lambda name: models is None or name in models  # noqa: E731
    col = f"{target}_l{lead}"
    cols = FEATURES + ["doy", col]
    sub = df[cols].dropna().sort_index()
    X = sub[feats].to_numpy(dtype=float)
    y = sub[col].to_numpy(dtype=float)
    doy = sub["doy"].to_numpy(dtype=int)
    state = sub["in_hw_today"].to_numpy(dtype=int)
    dates = sub.index

    rows, preds = [], []
    cv = RollingOriginCV(n_splits=N_SPLITS, test_size=TEST_SIZE, gap=GAP, expanding=True)
    for fold, (tr, te) in enumerate(cv.split(len(sub)), 1):
        y_tr, y_te = y[tr], y[te]
        if len(np.unique(y_tr)) < 2:
            continue  # train ไม่มีเหตุการณ์เลย — เทรนไม่ได้ (ไม่น่าเกิดกับ 30 ปี)

        # --- baselines (เรียนจาก train เท่านั้น) ---
        p_clim = predict_seasonal_climatology(doy[tr], y_tr, doy[te])
        p_pers = persistence_probs(state[tr], y_tr, state[te])
        forecasts: dict[str, np.ndarray] = {
            "climatology": p_clim,
            "persistence": p_pers,
        }

        # --- โมเดลหลัก ---
        for name in MAIN_MODELS:
            if not want(name):
                continue
            est = make_estimator(name)
            est.fit(X[tr], y_tr)
            forecasts[name] = est.predict_proba(X[te])[:, 1]

        # --- ablation + recalibration (รวม lgbm หลักเวอร์ชัน calibrate) ---
        for name in ABLATION_MODELS + CALIBRATED_MAIN:
            if not want(f"{name}_cal"):
                continue
            p_cal = fit_predict_calibrated(name, X[tr], y_tr, X[te])
            if p_cal is not None:
                forecasts[f"{name}_cal"] = p_cal

        for name, p in forecasts.items():
            m = evaluate_probabilistic(y_te, p, baseline_prob=p_clim)
            rows.append({"target": target, "lead": lead, "fold": fold,
                         "model": name, **m})
            preds.append(pd.DataFrame({
                "target": target, "lead": lead, "fold": fold, "model": name,
                "date": dates[te], "y": y_te, "p": p, "p_clim": p_clim,
            }))

    folds = pd.DataFrame(rows)
    pred = pd.concat(preds, ignore_index=True) if preds else pd.DataFrame()
    if verbose and not folds.empty:
        mean_bss = folds[folds.model == "lgbm"]["bss"].mean()
        print(f"  {target} lead {lead}: n={len(sub)}, base_rate={y.mean():.3f}, "
              f"lgbm BSS เฉลี่ย={mean_bss:+.3f}", flush=True)
    return folds, pred


def pooled_metrics(pred: pd.DataFrame) -> pd.DataFrame:
    """เมตริกจาก predictions รวมทุก fold (test ไม่ทับกัน จึง pool ได้ตรงๆ)."""
    rows = []
    for (target, lead, model), g in pred.groupby(["target", "lead", "model"]):
        m = evaluate_probabilistic(g["y"].to_numpy(), g["p"].to_numpy(),
                                   baseline_prob=g["p_clim"].to_numpy())
        rows.append({"target": target, "lead": lead, "model": model, **m})
    return pd.DataFrame(rows).sort_values(["target", "lead", "model"])


def plot_reliability(pred: pd.DataFrame, target: str, lead: int, path: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.plot([0, 1], [0, 1], "k--", lw=1, label="perfect")
    for model, color in (("climatology", "#888888"), ("logistic", "#1f77b4"),
                         ("lgbm", "#d62728"), ("lgbm_cal", "#2ca02c")):
        g = pred[(pred.target == target) & (pred.lead == lead) & (pred.model == model)]
        if g.empty:
            continue
        mp, of, ct = reliability_curve(g["y"].to_numpy(), g["p"].to_numpy(), n_bins=10)
        ok = ct > 0
        ax.plot(mp[ok], of[ok], "o-", color=color, label=f"{model}")
    ax.set_xlabel("forecast probability")
    ax.set_ylabel("observed frequency")
    ax.set_title(f"Reliability — {target}, lead {lead} wk")
    ax.legend()
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def plot_feature_importance(df: pd.DataFrame, path: Path) -> None:
    """feature importance ของ lgbm (เฉลี่ยข้าม fold) บน target หลัก lead 2."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    col = f"{PRIMARY_TARGET}_l2"
    sub = df[FEATURES + [col]].dropna().sort_index()
    X = sub[FEATURES].to_numpy(dtype=float)
    y = sub[col].to_numpy(dtype=float)
    cv = RollingOriginCV(n_splits=N_SPLITS, test_size=TEST_SIZE, gap=GAP)
    imps = []
    for tr, _te in cv.split(len(sub)):
        m = get_model("lgbm")
        m.fit(X[tr], y[tr])
        imps.append(m.feature_importances_)
    imp = pd.Series(np.mean(imps, axis=0), index=FEATURES).sort_values()
    fig, ax = plt.subplots(figsize=(7, 6))
    imp.plot.barh(ax=ax, color="#d62728")
    ax.set_title(f"LightGBM feature importance (split) — {PRIMARY_TARGET} lead 2 wk")
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)
    imp.sort_values(ascending=False).to_csv(path.with_suffix(".csv"),
                                            header=["importance"])


def main() -> int:
    df = pd.read_csv(DATASET, parse_dates=["date"], index_col="date")
    OUT_DIR.mkdir(exist_ok=True)
    FIG_DIR.mkdir(exist_ok=True)
    print(f"=== train: dataset {len(df)} วัน | targets={TARGETS} | leads={LEADS} ===")

    all_folds, all_preds = [], []
    for target in TARGETS:
        for lead in LEADS:
            folds, pred = run_target_lead(df, target, lead)
            all_folds.append(folds)
            all_preds.append(pred)

    folds = pd.concat(all_folds, ignore_index=True)
    pred = pd.concat(all_preds, ignore_index=True)
    pooled = pooled_metrics(pred)

    folds.to_csv(OUT_DIR / "metrics_folds.csv", index=False)
    pooled.to_csv(OUT_DIR / "metrics_pooled.csv", index=False)
    pred.to_csv(OUT_DIR / "predictions.csv", index=False)

    for lead in LEADS:
        plot_reliability(pred, PRIMARY_TARGET, lead,
                         FIG_DIR / f"reliability_lead{lead}.png")
    plot_feature_importance(df, FIG_DIR / "feature_importance.png")

    # --- สรุปผลหลักลงคอนโซล: BSS pooled ของ target หลัก ---
    print("\n=== BSS (pooled, เทียบ seasonal climatology) — target หลัก y_rm ===")
    view = pooled[pooled.target == PRIMARY_TARGET].pivot(
        index="model", columns="lead", values="bss")
    print(view.round(3).to_string())
    print("\n=== AUC (pooled) — target หลัก y_rm ===")
    view_auc = pooled[pooled.target == PRIMARY_TARGET].pivot(
        index="model", columns="lead", values="auc")
    print(view_auc.round(3).to_string())

    n_win = int((pooled[(pooled.target == PRIMARY_TARGET)
                        & (pooled.model.isin(MAIN_MODELS))]["bss"] > 0).sum())
    print(f"\nโมเดลหลักชนะ climatology (BSS>0): {n_win}/{len(MAIN_MODELS) * len(LEADS)} ช่อง")
    print(f"[OK] ผลทั้งหมดอยู่ที่ {OUT_DIR}")
    return 0


def _selftest() -> None:
    """ข้อมูลสังเคราะห์: เหตุการณ์ขึ้นกับ sm1 + ฤดูกาล -> โมเดลต้องชนะ climatology."""
    rng = np.random.default_rng(7)
    years = 12
    dates = []
    for yr in range(2000, 2000 + years):
        dates.extend(pd.date_range(f"{yr}-01-01", f"{yr}-07-31", freq="D"))
    idx = pd.DatetimeIndex(dates)
    n = len(idx)
    doy = idx.dayofyear.to_numpy()
    sm1 = 0.35 + 0.05 * np.sin(2 * np.pi * doy / 365) + rng.normal(0, 0.02, n)
    season = np.exp(-((doy - 105) ** 2) / (2 * 30 ** 2))   # พีคราว เม.ย.
    logit_p = -4.0 + 3.0 * season - 60.0 * (sm1 - 0.35)    # แล้งกว่า -> เสี่ยงกว่า
    p_true = 1 / (1 + np.exp(-logit_p))
    y_event = (rng.random(n) < p_true).astype(float)

    df = pd.DataFrame(index=idx)
    for c in FEATURES:
        df[c] = rng.normal(0, 1, n)          # ตัวแปรหลอก (noise)
    df["sm1"] = sm1                           # สัญญาณจริง
    df["doy_sin"] = np.sin(2 * np.pi * doy / 365.25)
    df["doy_cos"] = np.cos(2 * np.pi * doy / 365.25)
    df["in_hw_today"] = y_event               # ให้ persistence มีของจริงใช้
    df["doy"] = doy
    df[f"{PRIMARY_TARGET}_l2"] = y_event      # target ตรงกับสัญญาณ (ทดสอบ pipeline)
    df.index.name = "date"

    folds, pred = run_target_lead(df, PRIMARY_TARGET, 2, verbose=False)
    assert not folds.empty and not pred.empty
    pooled = pooled_metrics(pred)
    get = lambda m: float(pooled[pooled.model == m]["bss"].iloc[0])
    bss_log, bss_clim = get("logistic"), get("climatology")
    assert abs(bss_clim) < 0.15, f"climatology เทียบตัวเอง BSS ควร ~0 ได้ {bss_clim}"
    assert bss_log > 0.05, f"โมเดลต้องชนะ climatology บนข้อมูลที่มีสัญญาณ ได้ {bss_log}"
    # ablation ต้องมีผล (ข้อมูลยาวพอ) และ probability หลัง calibrate ต้องไม่เฟ้อ
    cal_models = [m for m in pooled.model.unique() if m.endswith("_cal")]
    assert cal_models, "ควรมีโมเดล ablation ที่ recalibrate แล้วอย่างน้อย 1 ตัว"
    for m in cal_models:
        g = pred[pred.model == m]
        infl = g["p"].mean() / max(g["y"].mean(), 1e-9)
        assert infl < 2.0, f"{m}: ความน่าจะเป็นเฉลี่ยเฟ้อ {infl:.2f} เท่าของ base rate"
    print(f"[OK] pipeline: logistic BSS={bss_log:+.3f} > 0, climatology~0, "
          f"calibrated ablation ไม่เฟ้อ ({', '.join(cal_models)})")
    print("[OK] self-test ผ่านทั้งหมด")


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "test":
        _selftest()
    else:
        main()
