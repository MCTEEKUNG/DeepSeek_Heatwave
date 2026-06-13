# ดีไซน์: โมเดล sub-seasonal heatwave รายจังหวัด (pooled) — DeepSeek_Heatwave

วันที่: 2026-06-13
สถานะ: ร่างเพื่อ review (ยังไม่อนุมัติ)
บริบท: งานวิจัย/วิทยานิพนธ์ — **subsystem #1 ของงานพอร์ต webapp (calm-authority) จาก Heatwave_AI**
แนวทางโมเดล: **Pooled + province features** (โมเดลเดียว train ข้ามทุกจังหวัด) ; **เสริม ไม่แทนที่** pipeline regional ที่ validate แล้ว

> #2 (webapp port: calm-authority + province map ต่อ forecast นี้) เป็น spec แยกหลังจาก #1 เสร็จ. spec นี้ไม่ครอบ webapp.

---

## 1. เป้าหมาย & ขอบเขต

เพิ่มความสามารถพยากรณ์ **ความน่าจะเป็น heatwave 2–6 สัปดาห์ล่วงหน้า รายจังหวัด (77 จังหวัด)** ด้วยโมเดล pooled ตัวเดียว
เพื่อให้หน้า MAP ของ webapp (#2) มีข้อมูลรายจังหวัดมาเติม ; คงมาตรฐาน rigor เดียวกับ pipeline regional
(leakage-safe CV, ชนะ climatology+persistence, calibration, bootstrap CI)

ผลลัพธ์หลัก: (ก) โมเดล pooled + artifacts รายlead, (ข) ตัวเลขสกิล pooled + **รายจังหวัด**, (ค) ไฟล์ forecast รายจังหวัดสำหรับ webapp

## 2. Non-goals (กันงานบาน)

- **ไม่แทนที่** pipeline regional เดิม (แกนวิทยานิพนธ์ + validation เดิมต้องอยู่ครบ รันได้เหมือนเดิม)
- ไม่สร้าง webapp/MAP ใน spec นี้ (= #2)
- ไม่ดาวน์โหลด ERA5 ใหม่ (ใช้ `.nc` grid เดิม)
- ไม่ทำโมเดลแยกรายจังหวัด (77×5) — เลือก pooled แล้ว
- ไม่เพิ่ม data source ใหม่ (NDVI/SST/elevation) ในรอบนี้ — future work

## 3. Province → grid cell

- พอร์ต `data/provinces.csv` จาก Heatwave_AI (77 จังหวัด: `id, code, name_th, name_en, region, lat, lon`)
- แต่ละจังหวัด map → **cell 0.25° ที่ใกล้ centroid ที่สุด** (grid เดิม lat 5–21 / lon 97–106, 65×37=2405 cell)
- predictors local ของจังหวัด = ค่าของ cell นั้น (Tmax, soil L1/L3) ดึงจาก `.nc` เดิม
- *หมายเหตุที่ทราบแล้ว:* 77 จังหวัด → 76 cell (1 คู่ในเขต กทม./ปริมณฑล ใช้ cell ร่วม → local feature เหมือนกัน ต่างที่ province-static lat/lon/region) — ยอมรับได้, บันทึกไว้
- *future refinement (ไม่ทำตอนนี้):* ค่าเฉลี่ย neighborhood แทน nearest-cell สำหรับจังหวัดใหญ่

## 4. Label รายจังหวัด (reuse กลไกเดิม)

ใช้นิยามเดียวกับ regional **ต่อจังหวัด**:
- `thr90_cell = doy_window_percentile(t_grid, q=90, window=15)` (per-cell — **มีอยู่แล้ว** ใน area-fraction path ของ `build_dataset`)
- `hw_cell = flag_heatwaves(hot_days(t_grid, thr90_cell), min_len=3)` (per-cell, fwd+bwd — ถูกต้องสำหรับ label)
- ต่อจังหวัด: เลือก series ของ cell ตัวเอง → `weekly_event_targets` → `y_rm_l{2..6}` รายจังหวัด
- target รายจังหวัด = heatwave ของ cell จังหวัดนั้น (p90 ราย doy ของ cell + ≥3 วันติด + weekly event) — กลไกเดียวกับ per-cell ใน area-fraction ที่มีอยู่ ; p95 เป็น ablation ได้ภายหลัง
- **จุดยืน R1 (สืบทอด):** threshold freeze-all-history ตอน serve (ถูกต้อง operational) ; leak อยู่ใน CI — **รัน R1 gate ซ้ำบนชุด pooled** (reuse `leak_check_r1` แนวเดิม) เพื่อยืนยันก่อนปิด

## 5. Features (antecedent-only — สืบทอด R2)

**Local (ราย cell จังหวัด):** `sm1, sm1_mean7, sm1_mean30, sm1_trend, sm3, sm3_mean7, sm3_mean30, sm3_trend, tmax, tmax_mean7, in_hw_today (trailing-only ตาม R2), hot_frac7`
**Shared (ทุกจังหวัดค่าเดียวกันต่อวัน):** `mjo_rmm1, mjo_rmm2, mjo_amp, mjo_sin, mjo_cos, nino34_lag1m, doy_sin, doy_cos`
**Province-static (ใหม่, geographic/leak-free):** `lat, lon, region` (region = one-hot 6 ภูมิภาค) — โมเดลเรียนความเสี่ยงพื้นฐานรายจังหวัดจากพิกัด+ภูมิภาค+ฟีเจอร์ climatology local ได้เอง. **ไม่ใช้ `prov_base_rate` ดิบเป็น feature** (เป็นค่าจาก label → เสี่ยง target-leak ถ้าคิดข้ามปี test) ; ถ้าจะทดลองให้คิด train-only ต่อ fold = ablation เสริมเท่านั้น

กติกา leakage (สืบทอด): ทุก local feature คำนวณจากข้อมูล ≤ t (rolling ย้อนหลัง/trailing) ; `in_hw_today` = trailing streak ≥3 (leak-free) ; `nino34_lag1m` = lag 1 เดือน ; **ห้าม** ใช้ค่าวันเป้าหมายของ cell เป็น feature

## 6. Pooled dataset

- 1 แถว = (จังหวัด × วันออกพยากรณ์) ; คอลัมน์ = features (ข้อ 5) + `province_id` + `date` + targets `y_rm_l{2..6}`
- ขนาด ~76 cell × ~10,900 วัน ≈ **~828k แถว/lead** (รวม targets ในแถวเดียวกัน)
- builder ใหม่ `build_provinces_dataset.py` → `data/processed/dataset_provinces.parquet`
- **เพิ่ม dep `pyarrow`** (parquet — เร็ว/เล็กกว่า csv ที่ขนาดนี้ ; `data/` gitignore อยู่แล้ว) — pin ใน requirements.txt
- เซฟ climatology รายจังหวัด (threshold + base_rate) แบบ freeze เหมือน regional (สำหรับ serving)

## 7. Model

- โมเดล pooled ตัวเดียว ; prod = `logistic_balanced_cal` (ครอบ StandardScaler) ; เทียบ `lgbm` (+Platt)
- reuse `train.fit_calibrated_model` (core/calib split แยกเวลา + gap) — ขั้นตอน deploy = ขั้นตอนที่วัดผล
- province-static + features ป้อนเข้าโมเดลเป็นคอลัมน์ปกติ ; **region = one-hot 6 ภูมิภาค** (ไม่ใช้ ordinal — กัน false ordering)

## 8. Validation

- **CV: rolling-origin บล็อกตามเวลา** gap=49 วัน — แบ่ง fold ตาม "ช่วงวันที่" (ทุกจังหวัดในช่วงเดียวกันอยู่ fold เดียวกัน) เพื่อกัน temporal leakage ; **ไม่ hold-out จังหวัด** (ต้องการครบทุกจังหวัด)
- baseline: seasonal climatology + conditional persistence **รายจังหวัด** (เรียนจาก train ของจังหวัดนั้น) ; reuse `evaluate.*`
- เมตริก: pooled BSS/Brier/AUC เทียบ baseline + bootstrap CI (moving-block) + paired test (FDR) + calibration decomposition
- **per-province BSS** (สกิลรายจังหวัด — ให้ webapp โชว์จังหวัดไหน skill ดี/อ่อน + guard n<50 = ไม่น่าเชื่อถือ)
- **Gate:** pooled ต้องชนะ climatology และ persistence อย่างมีนัยสำคัญ (q<0.05) ที่ lead ที่เคลม ; รายงานตรงๆ ถ้า lead/จังหวัดไหนอ่อน
- *หมายเหตุ compute:* ~828k แถว → CV + bootstrap ช้ากว่า regional หลายเท่า (เป็นนาที–สิบนาที) ; lgbm/logistic ไหว, bootstrap อาจลด B หรือ block ใหญ่ขึ้น

## 9. Output contract (รอยต่อไป #2 webapp)

`scripts/predict_provinces.py` (ขยายแนว `predict.py`) → `docs/forecast_provinces.json`:
```
{
  "schema_version": 1, "issue_date": "...", "generated_at": "...",
  "model": "logistic_balanced_cal", "n_provinces": 77,
  "provinces": [
    {"id":1,"code":"BKK","name_th":"...","name_en":"...","region":"...","lat":..,"lon":..,
     "forecasts":[{"lead_weeks":2,"probability":..,"climatology_base_rate":..,"ratio_vs_normal":..,
                   "risk_level_th":"..","risk_level_en":".."}, ... lead 3-6]}
  ],
  "skill": [{"province_id":1,"lead":2,"bss":..,"reliable":true}, ...]   // จาก validation
}
```
- train/serve parity: feature รายจังหวัดสร้างจาก builder ตัวเดียวกับตอน train (reuse — จุดพังอันดับ 1 ของ pipeline ทำนาย) + self-test parity
- operational: climatology จังหวัดแบบ freeze + ข้อมูลล่าสุด (reuse แนว `raw_recent` ที่มีอยู่)

## 10. Reuse vs ใหม่

| reuse (ไม่แตะ) | สร้างใหม่ |
|---|---|
| `heatwave_target` (per-cell percentile/flag), `cv`, `evaluate`, `models`, `train.fit_calibrated_model`, R2 trailing, จุดยืน R1, pipeline regional ทั้งหมด | `data/provinces.csv` (พอร์ต), `src/province_grid.py` (centroid→cell + extract), `scripts/build_provinces_dataset.py`, `scripts/train_provinces.py` (+ validation pooled/per-province), `scripts/predict_provinces.py`, R1 gate variant บน pooled |

## 11. Risks & mitigations

- **Compute (828k แถว):** bootstrap/CV ช้า → ลด B / block ใหญ่ขึ้น / รัน per-lead แยก ; lgbm เร็วพออยู่แล้ว
- **Cell collision (1 คู่):** local feature ซ้ำ → ต่างที่ province-static ; บันทึกไว้ ยอมรับ
- **จังหวัดเหตุการณ์น้อย (rare):** pooled ช่วย (เรียนข้ามจังหวัด) ; per-province BSS ใส่ guard n<50=unreliable
- **Spatial autocorrelation ใน CV/bootstrap:** ทุกจังหวัดวันเดียวกัน correlated → block ตามเวลา (ไม่ใช่สุ่มแถว) ; bootstrap แบบ block-by-date ; รายงานอย่างระวัง (CI อาจแคบเกินถ้าไม่ทำ)
- **R1 leak รายจังหวัด:** รัน gate ซ้ำบน pooled ก่อนปิด (เหมือน regional)
- **ทับ pipeline เดิม:** ไฟล์/สคริปต์ใหม่แยกชื่อ (`*_provinces`) ไม่แตะของเดิม → regional ยังรันได้

## 12. Acceptance criteria

1. `data/provinces.csv` + `src/province_grid.py` (centroid→cell) + self-test: 77 จังหวัด map ครบ, 76 unique cell
2. `build_provinces_dataset.py` → `dataset_provinces.parquet` (~828k แถว/lead) + self-test leakage (อนาคตไม่กระทบอดีต) + train/serve parity
3. `train_provinces.py`: ตาราง pooled BSS เทียบ clim+persist + bootstrap CI + per-province BSS ; **gate ผ่าน** (ชนะ baseline q<0.05)
4. R1 gate บน pooled: บันทึก ΔBSS + verdict
5. `predict_provinces.py` → `docs/forecast_provinces.json` ตาม schema ; parity self-test ผ่าน
6. pipeline regional เดิมยังรันได้ครบ (self-test ทุกโมดูลเดิมผ่าน)

## 13. ลำดับงาน & branch

branch `feat/per-province-model` (แตกจาก main แล้ว) ; commit spec ก่อน
1. provinces.csv + province_grid.py (+ self-test)
2. build_provinces_dataset.py (+ leakage/parity self-test) → สร้าง dataset
3. train_provinces.py + validation (pooled + per-province) → gate
4. R1 gate บน pooled
5. predict_provinces.py + forecast_provinces.json (+ parity self-test)
6. (จบ #1) → spec #2 webapp แยก

## 14. การตัดสินใจที่ล็อกแล้ว

- Pooled + province features (ไม่ใช่ per-province models)
- Additive — ไม่แทน regional
- nearest-cell ต่อจังหวัด (neighborhood = future)
- parquet + เพิ่ม dep `pyarrow`
- province-static = lat/lon/region (one-hot) เท่านั้น ; **prov_base_rate ดิบไม่ใช้** (target-leak) — ถ้าทดลองให้ train-only
- CV บล็อกตามเวลา ; ไม่ hold-out จังหวัด ; per-province BSS มี guard n<50
- รัน R1 gate ซ้ำบน pooled ก่อนปิด
