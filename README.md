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
  download_*.py        สคริปต์ดึงข้อมูล (resume + validate ได้)
data/    raw/ + processed/   (ไม่ commit)
docs/    spec และเอกสาร
```

ทุกโมดูลมี self-test ในตัว: `python scripts/<module>.py`
(console ภาษาไทยบน Windows: ตั้ง `PYTHONIOENCODING=utf-8`)

## ติดตั้ง

```
pip install -r requirements.txt
```

## สถานะ (2026-06-10)

- [x] โมดูลพื้นฐาน 5 ตัว + self-test ผ่าน
- [ ] ดาวน์โหลด ERA5 ให้ครบ 1994–2023 (กำลังรัน)
- [ ] ตัดสินนิยาม target เชิงพื้นที่ (regional-mean vs area-fraction)
- [x] BSS รับ baseline จากชุด train + baseline helpers (seasonal climatology, conditional persistence)
- [x] โมเดลไม่ถ่วงน้ำหนักเป็นค่าเริ่มต้น (`logistic`, `lgbm`) ; ตัวถ่วงน้ำหนักเป็น ablation
- [ ] `build_dataset.py` (ประกอบ feature ตาม lead time, lag Niño3.4 1 เดือน)
- [ ] `train.py` + recalibration ของโมเดล ablation บน validation block แยกตามเวลา + ประเมินผล
