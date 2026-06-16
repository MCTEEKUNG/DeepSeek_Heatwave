# ดีไซน์: Operational Province Mode — issue_date รีเฟรชด้วยข้อมูล ERA5 ล่าสุด

วันที่: 2026-06-16
สถานะ: อนุมัติดีไซน์แล้ว (รอ review spec)
บริบท: Part 1 ของงานทำให้ forecast รายจังหวัด "ออกพยากรณ์ปัจจุบันจริง" ได้ — แก้ปัญหา issue_date ค้างที่ 2023-12-31

> **Part 2 (GitHub Actions cron — ย้าย loop ไปรันบน CI เอง) เป็น spec แยก หลัง Part 1 เสร็จ.** spec นี้ไม่ครอบ CI/deployment.

---

## 1. ปัญหา & เป้าหมาย

**ปัญหา (root cause ที่ตรวจพบ 2026-06-16):** pipeline serving รายจังหวัด (`predict_provinces.py` → `build_provinces_features`) อ่าน**เฉพาะ grid ประวัติศาสตร์เต็ม** (`data/raw/tmax_thailand`, `soil_moisture_thailand`) แล้วหยิบ "วันล่าสุดที่ feature ครบ" = 2023-12-31 เสมอ. **ไม่มี operational/recent mode** — ต่างจาก regional `predict.py` ที่มี `operational` mode (อ่าน `raw_recent/` + climatology แช่แข็ง). ผลคือ frontend แสดงพยากรณ์ของปี 2023 ตลอดไป ไม่ว่าจะรีเฟรชกี่ครั้ง.

**เป้าหมาย:** ให้ `predict_provinces.py operational` ออก forecast รายจังหวัดจากข้อมูล ERA5 ล่าสุด (พิสูจน์แล้วว่าดึงได้ถึง ~6 วันก่อนปัจจุบัน เช่น 2026-06-10) โดยคงมาตรฐาน train/serve parity + leakage-safe เดิม. ส่งมอบบน laptop ก่อน (รันมือ) — การย้ายไป CI เป็น Part 2.

## 2. Non-goals (กันงานบาน)

- **ไม่ทำ** GitHub Actions / cron / deployment (= Part 2)
- **ไม่ retrain** โมเดล — ใช้ `.pkl` รายจังหวัดเดิม (lead 2–6)
- **ไม่แตะ** pipeline regional หรือ pipeline train รายจังหวัด (build/train/validate ของเดิมต้องรันได้เหมือนเดิม)
- ไม่เพิ่ม data source ใหม่ (NDVI/SST) — future work
- ไม่ refactor `build_provinces_features` เป็น abstraction ใหญ่ (YAGNI — แค่เพิ่ม flag)

## 3. การตัดสินใจที่ล็อกแล้ว

1. **per-province base_rate** — freeze base_rate รายจังหวัด (ต่อ lead) แล้วให้ `ratio_vs_normal`/risk label คิดจาก base_rate ของจังหวัดนั้นเอง. แก้ quirk เดิมที่ทั้ง 77 จังหวัดใช้ pooled base_rate ค่าเดียว (0.1136 @lead2). **probability ดิบไม่เปลี่ยน** — เปลี่ยนแค่ตัวหารของ ratio และเกณฑ์ป้าย
2. **Part 1 ครอบ 2 repo** — DeepSeek (operational mode) + frontend (banner conditional)
3. **banner conditional** — `HistoricalRunBanner` โชว์เฉพาะเมื่อ issue_date เก่ากว่า **14 วัน** เทียบ generated_at
4. **MJO impute + warn** (มิเรอร์ regional) — ถ้า MJO ล่าสุดหาย impute ด้วย climatology mean + ใส่ warning ใน JSON (ไม่ fail). **แหล่ง mean = `mjo_means` ใน `models/climatology.pkl` ที่มีอยู่แล้ว** (regional) — reuse ไม่ต้อง duplicate ลง `climatology_provinces.pkl`
5. แนวทาง = เพิ่ม flag `operational=` ใน `build_provinces_features` (code path เดียวกับ train → parity)

## 4. Architecture & components

### 4.1 ใหม่ — artifact climatology แช่แข็งรายจังหวัด
`models/climatology_provinces.pkl` (สร้างครั้งเดียวจาก grid 30 ปี):
- `thr90_grid`: `doy_window_percentile(t_grid, q=90, window=15)` — เกณฑ์ p90 ราย doy ราย cell (366 × lat × lon). ใช้คำนวณ `hot_rm` → `in_hw_today`/`hot_frac7` ในโหมด operational (recompute จาก 70 วันไม่ได้ ต้องใช้ distribution เต็ม)
- `base_rate`: `{province_id: {lead: rate}}` — per-province per-lead จาก target columns ใน `dataset_provinces.parquet` (`y_rm_l{L}` ของแต่ละจังหวัด `.mean()`)
- metadata: `built_at`, `source_years`, `window`, `min_run`

### 4.2 ใหม่ — `scripts/freeze_provinces_climatology.py`
- โหลด t_grid เต็ม → คำนวณ `thr90_grid`
- โหลด parquet → คำนวณ per-province base_rate ต่อ lead
- เซฟ `climatology_provinces.pkl`
- self-test: thr90_grid มีมิติถูก, base_rate ทุกจังหวัด/lead ∈ [0,1]

### 4.3 แก้ — `scripts/build_provinces_dataset.py`
`build_provinces_features(verbose=False, operational=False)`:
- `operational=False` (เดิม): ไม่เปลี่ยนพฤติกรรม — อ่าน grid เต็ม, คำนวณ thr90 เอง
- `operational=True`:
  - อ่าน `data/raw_recent/{tmax,soil}` แทน grid เต็ม
  - โหลด `thr90_grid` จาก `climatology_provinces.pkl` (ไม่ recompute)
  - คำนวณ `hot_rm = hot_days(t_recent, thr90_grid)` แล้วทำ feature ตามเดิม (`lookback_features` ตัวเดิม → parity)
  - MJO หาย → impute (มิเรอร์ regional) + ตั้ง flag warning

### 4.4 แก้ — `scripts/predict_provinces.py`
- arg `operational` (เหมือน regional `predict.py operational`)
- `operational=True`: `build_forecast` ใช้ `build_provinces_features(operational=True)`, risk จาก per-province base_rate แช่แข็ง, `model` field เติม suffix หรือ warning ถ้า MJO impute
- issue_date = วันล่าสุดที่ feature ครบในหน้าต่าง recent

### 4.5 แก้ — frontend `components/forecast/HistoricalRunBanner.tsx`
- รับ `generatedAt` เพิ่ม; โชว์ banner เฉพาะเมื่อ `(generatedAt - issueDate) > 14 วัน`
- ข้อมูลสด (issue 2026-06-10, generated วันนี้) → ไม่ขึ้น banner

## 5. Data flow (operational)
```
[prereq] download_era5_hourly_aggregate.py recent   → raw_recent/{tmax,soil}
[prereq] download_indices.py                          → MJO/Niño34 ล่าสุด
   → build_provinces_features(operational=True)
        recent grid + thr90 แช่แข็ง + MJO impute(ถ้าหาย)
   → โมเดล .pkl เดิม (lead 2–6) predict_proba → Platt calibrate
   → risk_level(p, base_rate[province][lead])         ← per-province
   → forecast_provinces.json (issue_date ~2026-06-10, + warning ถ้า impute)
   → validate_contract.py (hard gate)  → publish_bridge (Part 2 ทำให้อัตโนมัติ)
```

## 6. Error handling
- ขาด `raw_recent/` หรือ `climatology_provinces.pkl` → **fail ชัดเจน** (FileNotFoundError ข้อความบอกวิธีแก้) — ห้าม silently fallback ไป grid เก่า (กันออก forecast เก่าโดยไม่รู้ตัว)
- MJO ล่าสุดหาย → impute + warning ใน JSON (ไม่ fail)
- หน้าต่าง recent สั้นจน feature 30 วันไม่ครบบางจังหวัด → `dropna` เดิมตัดแถวนั้น; ถ้าทุกจังหวัดว่าง → fail
- Niño34 lag-1m หายช่วงล่าสุด → ใช้ค่า lag ที่มี; ถ้าหายหมด → fail ชัดเจน (ดีกว่า impute ดัชนีใหญ่เงียบ ๆ)

## 7. Testing / acceptance
1. `freeze_provinces_climatology.py` self-test: thr90_grid มิติถูก, base_rate ทุกจังหวัด×lead ∈ [0,1] (77×5 ค่า)
2. **operational parity self-test** (มิเรอร์ regional `predict.py` test): feature ของวันที่อยู่ทั้งใน recent และ historical grid → ค่าตรงกัน (พิสูจน์ operational ไม่ทำ feature เพี้ยน)
3. **regime sanity**: probability ของ issue_date 2026-06-10 ทุกจังหวัด/lead ∈ (0,1), base_rate per-province ต่างกันจริง (ไม่ใช่ค่าเดียวทั้ง 77)
4. **end-to-end**: `predict_provinces.py operational` → JSON มี issue_date ปี 2026 → `validate_contract.py` exit 0
5. banner: unit test (vitest) โชว์เมื่อเก่า >14 วัน, ซ่อนเมื่อสด
6. **regression**: pipeline เดิมยังรันได้ — `build_provinces_features(operational=False)` + self-test เดิมทุกตัวผ่าน, `predict_provinces.py` (ไม่มี arg) ยังออกผลเหมือนเดิม

## 8. Reuse vs ใหม่

| reuse (ไม่แตะ) | สร้าง/แก้ |
|---|---|
| `.pkl` โมเดลรายจังหวัด, `lookback_features`, `mjo_features`, `nino_lagged_daily`, `risk_level`, `doy_window_percentile`/`hot_days`, regional `predict.py operational` (เป็น template), `validate_contract.py` | ใหม่: `climatology_provinces.pkl`, `freeze_provinces_climatology.py` ; แก้: `build_provinces_features(operational=)`, `predict_provinces.py`(arg), `HistoricalRunBanner.tsx`(conditional) |

## 9. ลำดับงาน
1. `freeze_provinces_climatology.py` + artifact + self-test
2. `build_provinces_features(operational=True)` + operational parity self-test
3. `predict_provinces.py operational` + regime sanity + end-to-end (ออก JSON ปี 2026)
4. frontend banner conditional + vitest
5. regression: ยืนยัน pipeline เดิมไม่พัง
6. (จบ Part 1) → spec Part 2 (GitHub Actions cron)
