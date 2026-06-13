"""
ทำนายความน่าจะเป็นของ heatwave ล่วงหน้า 2-6 สัปดาห์ สำหรับ "วันออกพยากรณ์ล่าสุด"
แล้วเขียนผลเป็น outputs/forecast.json ให้เว็บแอปอ่านไปแสดงผล

แนวคิด (สำคัญต่อความเร็ว/ความน่าเชื่อถือของระบบ):
  - ไม่ดึง ERA5 ตอน user กดดู — predict.py รันเป็น batch (เช่นรายสัปดาห์ผ่าน cron)
    แล้วเก็บผลเป็นไฟล์ JSON นิ่ง ๆ ; เว็บแอปแค่อ่านไฟล์ -> เร็วทุก device
  - feature ของวันทำนายสร้างจาก build_feature_table() "ตัวเดียวกับตอนเทรน"
    (reuse ไม่ reimplement) -> รับประกัน train/serve parity
  - โมเดลโหลดจาก models/*.pkl ที่ train_final.py เซฟไว้ (ไม่เทรนใหม่ตอนทำนาย)
  - ไม่ใส่ช่วงความเชื่อมั่นรายสัปดาห์ปลอม ๆ : โมเดล calibrated ให้ "เลขเดียว"
    เราแนบ base_rate (ฐานภูมิอากาศ) ให้ตีความว่า "สูง/ต่ำกว่าปกติ" อย่างซื่อสัตย์แทน

ใช้งาน:  python predict.py            # อ่าน raw ปัจจุบัน -> outputs/forecast.json
         python predict.py test       # self-test: feature row ตรงกับ dataset.csv เป๊ะ
"""
from __future__ import annotations

import json
import pickle
import sys
from datetime import datetime, timezone
from pathlib import Path

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from build_dataset import build_feature_table, load_climatology, CLIM_FILE
from train import FEATURES, LEADS, PRIMARY_TARGET
from train_final import MODEL_DIR, artifact_path

ROOT = Path(__file__).resolve().parent.parent
DATASET = ROOT / "data" / "processed" / "dataset.csv"
# forecast.json อยู่ใน docs/ (tracked + GitHub Pages เสิร์ฟให้เว็บแอป) — ไม่ใช่ outputs/ ที่ถูก gitignore
OUT_FILE = ROOT / "docs" / "forecast.json"
# โฟลเดอร์ข้อมูล "ล่าสุด" ที่ cron ดึงมา (~45-60 วัน) สำหรับ operational predict
RECENT_TMAX_DIR = ROOT / "data" / "raw_recent" / "tmax_thailand"
RECENT_SOIL_DIR = ROOT / "data" / "raw_recent" / "soil_moisture_thailand"
REGION = "Thailand"

# ระดับความเสี่ยง = อัตราส่วนความน่าจะเป็นต่อ "ฐานภูมิอากาศ" (base rate) ของ lead นั้น
# ตีความซื่อสัตย์กว่าเกณฑ์สัมบูรณ์: 0.30 อาจ "สูงผิดปกติ" ถ้าฐาน 0.05 แต่ "ปกติ" ถ้าฐาน 0.30
RISK_BANDS = [
    (0.75, "ต่ำ", "Low"),
    (1.5, "ปกติ", "Normal"),
    (2.5, "สูง", "Elevated"),
    (float("inf"), "สูงมาก", "High"),
]


def risk_level(prob: float, base_rate: float) -> tuple[str, str, float]:
    """คืน (ป้ายไทย, ป้ายอังกฤษ, อัตราส่วนต่อฐาน). prob ต่ำมากบังคับเป็น 'ต่ำ' เสมอ."""
    ratio = prob / base_rate if base_rate > 1e-6 else float("inf")
    if prob < 0.05:
        return "ต่ำ", "Low", ratio
    for hi, th, en in RISK_BANDS:
        if ratio < hi:
            return th, en, ratio
    return RISK_BANDS[-1][1], RISK_BANDS[-1][2], ratio


def load_artifact(lead: int) -> dict:
    # pickle ปลอดภัยที่นี่: .pkl สร้างเองด้วย train_final.py ในเครื่อง/CI เดียวกัน
    # (sklearn ใช้ pickle เป็นมาตรฐาน) — ไม่ได้โหลดจากแหล่งภายนอกที่ไม่น่าเชื่อถือ
    path = artifact_path(lead)
    if not path.exists():
        raise FileNotFoundError(
            f"ไม่พบโมเดล {path.name} — รัน `python train_final.py` ก่อน"
        )
    with open(path, "rb") as f:
        return pickle.load(f)


def latest_feature_row(feat: pd.DataFrame) -> pd.Series:
    """แถว feature ของ 'วันออกพยากรณ์ล่าสุด' ที่ feature ครบทุกตัว (มีข้อมูลย้อนหลังพอ)."""
    valid = feat[FEATURES].dropna()
    if valid.empty:
        raise RuntimeError("ไม่มีวันใดที่ feature ครบ — ข้อมูล raw ย้อนหลังไม่พอ (ต้อง >=30 วัน)")
    return valid.iloc[-1]


def build_forecast(issue_date: str | None = None, operational: bool = False) -> dict:
    """ออกพยากรณ์ ; issue_date=None -> วันล่าสุดที่ feature ครบ ;
    ระบุวันได้ (YYYY-MM-DD) สำหรับ backfill/ตรวจย้อนหลัง (ต้องมี feature ครบวันนั้น).

    operational=True: ใช้ climatology แช่แข็ง + ข้อมูลล่าสุดในโฟลเดอร์ raw_recent/
    (ไม่ต้องมี 30 ปี) — สำหรับ cron. False: ใช้ข้อมูลเทรนเต็ม (local/manual).
    """
    if operational:
        clim = load_climatology()
        tmax_dir, soil_dir = RECENT_TMAX_DIR, RECENT_SOIL_DIR
    else:
        clim = tmax_dir = soil_dir = None
    feat, _daily, _grid, _clim = build_feature_table(
        verbose=False, clim=clim, tmax_dir=tmax_dir, soil_dir=soil_dir)
    if issue_date is None:
        row = latest_feature_row(feat)
    else:
        ts = pd.Timestamp(issue_date)
        if ts not in feat.index:
            raise ValueError(f"ไม่มีวันที่ {ts.date()} ในตาราง feature")
        row = feat.loc[ts]
        if row[FEATURES].isna().any():
            raise ValueError(f"feature ของ {ts.date()} ไม่ครบ (ข้อมูลย้อนหลังไม่พอ)")
    issue_date_ts = pd.Timestamp(row.name)
    X = row[FEATURES].to_numpy(dtype=float).reshape(1, -1)

    issue_doy = int(issue_date_ts.dayofyear)
    forecasts = []
    warnings_out = []
    model_name = None
    for lead in LEADS:
        art = load_artifact(lead)
        model_name = art["model_name"]
        p = float(art["calibrator"].transform(art["estimator"].predict_proba(X)[:, 1])[0])
        th, en, ratio = risk_level(p, art["base_rate"])
        # หน้าต่างเป้าหมาย: สัปดาห์ที่ L หลังวันออกพยากรณ์ (7 วัน)
        valid_from = issue_date_ts + pd.Timedelta(days=7 * lead)
        valid_to = valid_from + pd.Timedelta(days=6)
        # in-domain: วันออกพยากรณ์ต้องอยู่ในช่วงฤดูที่โมเดลเคยเทรน (ข้อมูลรอบนี้ ม.ค.-ก.ค.)
        in_domain = art["train_issue_doy_min"] <= issue_doy <= art["train_issue_doy_max"]
        if not in_domain:
            warnings_out.append(
                f"lead {lead}: วันออกพยากรณ์ (doy {issue_doy}) อยู่นอกฤดูที่โมเดลเทรน "
                f"(doy {art['train_issue_doy_min']}-{art['train_issue_doy_max']}) "
                f"— ผลเป็นการ extrapolate เชื่อถือได้ต่ำ"
            )
        forecasts.append({
            "lead_weeks": lead,
            "valid_from": str(valid_from.date()),
            "valid_to": str(valid_to.date()),
            "probability": round(p, 4),
            "climatology_base_rate": round(art["base_rate"], 4),
            "ratio_vs_normal": round(ratio, 2),
            "risk_level_th": th,
            "risk_level_en": en,
            "in_training_domain": bool(in_domain),
        })

    return {
        "schema_version": 1,
        "region": REGION,
        "target": "heatwave (regional-mean Tmax > p90 ราย doy, ติดต่อกัน >=3 วัน)",
        "model": model_name,
        "issue_date": str(issue_date_ts.date()),
        "data_through": str(issue_date_ts.date()),
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "note": "ความน่าจะเป็น calibrated แล้ว ; เทียบ climatology_base_rate เพื่อดู 'สูง/ต่ำกว่าปกติ'",
        "warnings": warnings_out,
        "forecasts": forecasts,
    }


def predict(issue_date: str | None = None, operational: bool = False,
            verbose: bool = True) -> dict:
    fc = build_forecast(issue_date, operational=operational)
    OUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    OUT_FILE.write_text(json.dumps(fc, ensure_ascii=False, indent=2), encoding="utf-8")
    if verbose:
        print(f"=== predict: ออกพยากรณ์ ณ {fc['issue_date']} | โมเดล {fc['model']} ===")
        for f in fc["forecasts"]:
            print(f"  lead {f['lead_weeks']} สัปดาห์ ({f['valid_from']}..{f['valid_to']}): "
                  f"p={f['probability']:.3f} ({f['ratio_vs_normal']:.1f}x ปกติ) "
                  f"-> {f['risk_level_th']} / {f['risk_level_en']}"
                  f"{'' if f['in_training_domain'] else '  ⚠️ นอกโดเมน'}")
        for w in fc["warnings"]:
            print(f"  ⚠️  {w}")
        print(f"[OK] {OUT_FILE}")
    return fc


# ---------------------------------------------------------------- self-test

def _selftest() -> None:
    """พิสูจน์ train/serve parity: feature row ที่ predict สร้าง = แถวใน dataset.csv เป๊ะ.

    นี่คือด่านที่จับ bug ร้ายแรงที่สุดของ pipeline ทำนาย (feature ตอน serve != ตอน train).
    ต้องมี data/raw + dataset.csv ครบ ; ถ้าไม่มีจะข้าม (ใช้รันบนเครื่อง dev)
    """
    # 1) risk_level: ตรรกะอัตราส่วนต่อฐาน
    assert risk_level(0.30, 0.30)[1] == "Normal"
    assert risk_level(0.30, 0.05)[1] == "High"      # 6x ฐาน
    assert risk_level(0.02, 0.30)[1] == "Low"       # prob ต่ำมากบังคับ Low
    assert risk_level(0.10, 0.05)[1] == "Elevated"  # 2x ฐาน
    print("[OK] risk_level: อัตราส่วนต่อ base rate ถูกต้อง")

    if not DATASET.exists():
        print("[ข้าม] ไม่มี dataset.csv — ข้าม parity test (รันบนเครื่องที่มีข้อมูลจริง)")
        return
    raw_dir = ROOT / "data" / "raw" / "tmax_thailand"
    if not raw_dir.exists() or not list(raw_dir.glob("*.nc")):
        print("[ข้าม] ไม่มี data/raw — ข้าม parity test")
        return

    # 2) parity: feature ที่ build_feature_table สร้าง = ที่ build() เขียนลง dataset.csv
    print("[parity] โหลด raw + สร้าง feature table (อาจใช้เวลาสักครู่) ...")
    feat, _daily, _grid, _clim = build_feature_table(verbose=False)
    ds = pd.read_csv(DATASET, parse_dates=["date"], index_col="date")

    common = feat.index.intersection(ds.index)
    assert len(common) > 100, f"วันร่วมน้อยผิดปกติ: {len(common)}"
    a = feat.loc[common, FEATURES].to_numpy(dtype=float)
    b = ds.loc[common, FEATURES].to_numpy(dtype=float)
    both_nan = np.isnan(a) & np.isnan(b)
    close = np.isclose(a, b, rtol=1e-6, atol=1e-8) | both_nan
    if not close.all():
        bad = np.argwhere(~close)
        r, c = bad[0]
        raise AssertionError(
            f"feature ไม่ตรง dataset.csv ที่ {common[r].date()} คอลัมน์ {FEATURES[c]}: "
            f"predict={a[r, c]} vs dataset={b[r, c]} (รวม {len(bad)} จุด)"
        )
    print(f"[OK] train/serve parity: feature {len(FEATURES)} คอลัมน์ x {len(common)} วัน "
          f"ตรงกับ dataset.csv เป๊ะ")

    # 3) แถวล่าสุดที่ feature ครบต้องเลือกได้
    row = latest_feature_row(feat)
    assert not row[FEATURES].isna().any(), "แถวล่าสุดต้อง feature ครบ"
    print(f"[OK] วันออกพยากรณ์ล่าสุดที่ feature ครบ: {pd.Timestamp(row.name).date()}")

    # 4) operational parity: climatology แช่แข็ง + ข้อมูล "ชุดย่อย" -> feature row ต้องตรง dataset
    #    (พิสูจน์ว่า cron ที่มีแค่ข้อมูลล่าสุดให้ผลเท่าเส้นทางเทรนเต็ม)
    #    เทียบที่ "วันภายใน" ที่ไม่ใช่ขอบท้าย window -> มี forward context เท่า dataset.csv
    #    จึงแยกได้ว่า in_hw_today (มองหน้า-หลัง) ไม่ใช่ตัวที่ทำให้ไม่ตรง (advisor เตือน)
    if CLIM_FILE.exists():
        import tempfile, shutil
        clim = load_climatology()
        # เลือกหน้าต่าง ~150 วันของปีหนึ่ง จำลอง "ข้อมูลล่าสุดที่ cron ดึง"
        yrs = sorted({p.stem.split("_")[3] for p in raw_dir.glob("era5_tmax_thailand_*.nc")
                      if p.stem.split("_")[3].isdigit()})
        test_yr = yrs[len(yrs) // 2]  # ปีกลางๆ ที่มีข้อมูลครบ
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            tt, ts_ = tmp / "tmax", tmp / "soil"
            tt.mkdir(); ts_.mkdir()
            for p in raw_dir.glob(f"*_{test_yr}*.nc"):
                shutil.copy(p, tt / p.name)
            for p in (ROOT / "data" / "raw" / "soil_moisture_thailand").glob(f"*_{test_yr}*.nc"):
                shutil.copy(p, ts_ / p.name)
            feat_op, _d, _g, _c = build_feature_table(verbose=False, clim=clim,
                                                      tmax_dir=tt, soil_dir=ts_)
        # เทียบเฉพาะวัน "ภายใน" (ตัด 7 วันท้าย = ขอบ window ที่ in_hw_today ต่างได้)
        idx_in = feat_op.dropna(subset=FEATURES).index[:-7]
        common2 = idx_in.intersection(ds.index)
        assert len(common2) > 30, f"วันร่วม operational น้อย: {len(common2)}"
        ao = feat_op.loc[common2, FEATURES].to_numpy(dtype=float)
        bo = ds.loc[common2, FEATURES].to_numpy(dtype=float)
        close2 = np.isclose(ao, bo, rtol=1e-6, atol=1e-8) | (np.isnan(ao) & np.isnan(bo))
        if not close2.all():
            r, c = np.argwhere(~close2)[0]
            raise AssertionError(
                f"operational feature ไม่ตรง dataset ที่ {common2[r].date()} "
                f"{FEATURES[c]}: op={ao[r, c]} vs ds={bo[r, c]}")
        print(f"[OK] operational parity (clim แช่แข็ง + ข้อมูลปี {test_yr} เท่านั้น): "
              f"{len(common2)} วันภายในตรง dataset เป๊ะ")
    else:
        print(f"[ข้าม] ไม่มี {CLIM_FILE.name} — ข้าม operational parity (รัน build_dataset ก่อน)")
    print("[OK] self-test ผ่านทั้งหมด")


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "test":
        _selftest()
    elif len(sys.argv) > 1 and sys.argv[1] == "operational":
        predict(operational=True)         # cron: clim แช่แข็ง + data/raw_recent/ (ไม่ต้องมี 30 ปี)
    elif len(sys.argv) > 1:
        predict(issue_date=sys.argv[1])   # python predict.py 2010-02-01 (backfill/ตรวจย้อนหลัง)
    else:
        predict()
