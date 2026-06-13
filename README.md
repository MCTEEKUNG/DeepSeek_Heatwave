# DeepSeek_Heatwave — Sub-seasonal Heatwave Probability Prediction (Thailand)

งานวิทยานิพนธ์: ทำนาย**ความน่าจะเป็น**ของการเกิดคลื่นความร้อนในประเทศไทย
ล่วงหน้า **2–6 สัปดาห์ (sub-seasonal)** ด้วย classical ML บนข้อมูลตาราง
(soil moisture + climate indices) — รายละเอียดดีไซน์เต็มอยู่ที่
`docs/superpowers/specs/2026-06-08-heatwave-subseasonal-prediction-design.md`

## นิยามหลัก

- **Heatwave** = Tmax รายวัน > เปอร์เซ็นไทล์ 90 (เทียบ 95) ของภูมิอากาศฐาน
  (moving window ±15 วันรอบวันปฏิทิน) ติดต่อกัน ≥ 3 วัน
- **เกณฑ์ความสำเร็จ** = ชนะ baseline climatology + persistence
  (วัดด้วย Brier/BSS, ROC-AUC, reliability diagram)

## ข้อมูล

| ชนิด | แหล่ง | สคริปต์ |
|------|-------|---------|
| Tmax รายวัน (สร้าง target) | ERA5 hourly → daily max | `scripts/download_era5_hourly_aggregate.py` |
| Soil moisture ชั้น 1, 3 (feature หลัก) | ERA5 hourly → daily mean | `scripts/download_era5_hourly_aggregate.py` |
| MJO RMM, Niño3.4 (features) | BoM / NOAA PSL | `scripts/download_indices.py` |

ขอบเขต: กรอบไทย [21N, 97E, 5S(→5N), 106E], ปี 1994–2023, เดือน ม.ค.–ก.ค.
ต้องมี CDS API key ที่ `~/.cdsapirc` (`data/` ไม่อยู่ใน git — สร้างใหม่ด้วยสคริปต์ได้)

## โครงสร้าง

```
scripts/
  units_utils.py       ด่านหน่วย (Kelvin→°C, fail loudly)
  heatwave_target.py   นิยาม target (percentile + ≥3 วันติดกัน)
  cv.py                blocked/rolling-origin time-series CV (กัน leakage)
  evaluate.py          Brier/BSS/AUC/reliability + baselines (seasonal climatology, persistence)
  models.py            model registry (logistic / lgbm ไม่ถ่วงน้ำหนัก = หลัก ; *_balanced / balanced_rf = ablation)
  build_dataset.py     ประกอบตาราง feature/target ราย lead -> data/processed/dataset.csv
  train.py             CV + baselines + recalibration ablation -> outputs/ (metrics, figures)
  download_*.py        สคริปต์ดึงข้อมูล (resume + validate ได้)
data/     raw/ + processed/   (ไม่ commit)
outputs/  เมตริก + กราฟผลการทดลอง (ไม่ commit)
docs/     spec และเอกสาร
```

## นิยาม target เชิงพื้นที่ (ตัดสินใจ 2026-06-11)

- **หลัก — regional-mean**: เฉลี่ย Tmax ทั้งกรอบไทย (ถ่วง cos(lat)) ก่อน แล้วหาเกณฑ์
  p90 ราย doy + ≥3 วันติด — สัญญาณ sub-seasonal มาจาก driver ระดับใหญ่
  จึงสอดคล้องกับเหตุการณ์ระดับภูมิภาคที่สุด
- **รอง — area-fraction (ablation)**: heatwave รายเซลล์ แล้วนับวันที่สัดส่วนพื้นที่ ≥ 15%
- **เสริม**: regional-mean ที่ p95 (ตาม spec ให้เทียบ 90 vs 95)
- target รายสัปดาห์: y=1 ถ้ามีวัน heatwave ≥1 วันในหน้าต่าง 7 วันที่ lead 2-6 สัปดาห์

ทุกโมดูลมี self-test ในตัว: `python scripts/<module>.py`
(console ภาษาไทยบน Windows: ตั้ง `PYTHONIOENCODING=utf-8`)

## ติดตั้ง

```
pip install -r requirements.txt
```

## สถานะ (2026-06-11) — pipeline รอบแรกครบ ✅

- [x] โมดูลพื้นฐาน 5 ตัว + self-test ผ่าน
- [x] ดาวน์โหลด ERA5 ครบ 1994–2023 (ม.ค.–ก.ค.) + ดัชนี MJO/Niño3.4
- [x] ตัดสินนิยาม target เชิงพื้นที่ (regional-mean หลัก + area-fraction ablation)
- [x] BSS รับ baseline จากชุด train + baseline helpers (seasonal climatology, conditional persistence)
- [x] โมเดลไม่ถ่วงน้ำหนักเป็นค่าเริ่มต้น (`logistic`, `lgbm`) ; ตัวถ่วงน้ำหนักเป็น ablation
- [x] `build_dataset.py` — 6,367 วันออกพยากรณ์, 20 features, target 3 นิยาม × lead 2–6 สัปดาห์
- [x] `train.py` — rolling-origin CV (gap 49 วัน) + recalibration (Platt บน block แยกตามเวลา) + ประเมินผล

## สถานะ (2026-06-13) — leakage audit + serving hardening ✅

- [x] R2: `in_hw_today` เป็น trailing-only (กัน forward-looking feature + serve-consistent)
- [x] R1: วัด ΔBSS percentile-label leak (gate) — ดู `docs/INTEGRITY.md`
- [x] retrain final ครอบเต็มปี (issue doy เต็ม) + parity self-test ไม่ต้องตัด 7 วันท้าย
- [x] runbook ออก forecast on-demand: `docs/RUNBOOK.md`
- รายละเอียด: `docs/superpowers/specs/2026-06-13-rigor-serving-hardening-design.md`

## ผลหลักรอบแรก (2026-06-11)

BSS (pooled, เทียบ seasonal climatology) — target หลัก regional-mean p90:

| โมเดล | lead 2 | 3 | 4 | 5 | 6 |
|---|---|---|---|---|---|
| logistic | **+0.168** | +0.109 | +0.118 | +0.050 | −0.014 |
| lgbm_cal | +0.144 | **+0.156** | +0.059 | +0.024 | **+0.103** |
| logistic_balanced_cal | +0.194 | +0.138 | +0.164 | +0.081 | +0.071 |
| persistence (baseline) | +0.095 | +0.048 | +0.047 | +0.024 | +0.017 |

- **เกณฑ์ความสำเร็จผ่าน**: ชนะทั้ง climatology และ persistence — logistic ที่ lead 2–5, lgbm_cal ทุก lead
- บทเรียนสำคัญ: lgbm ดิบ AUC ดี (0.69–0.76) แต่ BSS ติดลบทุก lead → probability ต้อง recalibrate เสมอ
- Feature สำคัญสุด (lgbm gain): `nino34_lag1m` > `sm1_mean30` > `sm3_mean30` — สอดคล้องสมมติฐาน ENSO + soil-moisture memory
- รายละเอียดเต็ม: `outputs/metrics_pooled.csv`, กราฟ reliability + feature importance ที่ `outputs/figures/`

## งานถัดไป (ทางเลือก)

- [ ] วิเคราะห์ fold ที่อ่อน (ปี El Niño 2015/16, 2023) + sensitivity ของ AF_THRESHOLD
- [ ] ขยายข้อมูลเป็นทั้งปี / เพิ่ม feature (SST, geopotential)
- [ ] ablation NDVI ตาม spec (ท้ายเล่ม)
