# Production-Readiness Audit & Gate — Design

วันที่: 2026-06-16
สถานะ: approved (brainstorming) → รอ implementation plan

## ปัญหา (motivation)

ไฟล์ที่เผยแพร่สาธารณะตอนนี้ (`https://mcteekung.github.io/heatwave-contract/forecast_provinces.json`)
แสดง `generated_at` เป็นปัจจุบัน (2026-06-14) แต่ `issue_date` = **2023-12-31** และ
ทุกจังหวัดทุก lead = "สูงมาก / High" (กรุงเทพฯ lead2 prob 0.386, ratio 3.4) —
เป็นข้อมูล backtest ช่วง El Niño ปลายปี 2023 ที่ถูก publish ราวกับเป็นพยากรณ์ของ "อีก 2–6 สัปดาห์ข้างหน้า".
ความเสี่ยง: ประชาชนตื่นตระหนกจากพยากรณ์เก่าที่ผิดบริบท.

### Root cause (ยืนยันแล้ว ไม่ใช่แค่อาการ)
1. `predict_provinces.build_forecast()` ใช้ `valid.sort_values("date").iloc[-1]` จาก dataset เทรนเต็ม
   → ได้วันล่าสุดในคลัง = 2023-12-31 **เสมอ**. ต่างจาก regional `predict.py` ที่มี `operational` mode
   (clim แช่แข็ง + `data/raw_recent/`). **per-province ไม่มี operational mode เลย** = สร้างได้แค่ demo.
2. `validate_contract.check_staleness()` ตรวจแค่ `generated_at` ไม่ตรวจ `issue_date`
   → "ข้อมูลเก่า 3 ปีที่ generate ใหม่วันนี้" ผ่าน gate ฉลุย.

## เป้าหมาย

ทำให้ระบบ per-province แสดง "พยากรณ์ของจริง ณ ปัจจุบัน" + มีระบบตรวจ (audit) และด่านอัตโนมัติ (gate)
ที่กันของเสียไม่ให้ขึ้นเว็บได้อีก. ครอบคลุมทั้ง pipeline: ข้อมูลดิบ → โมเดล → เผยแพร่.

## ขอบเขต

- **In scope:** per-province operational mode, ชุดเช็คความพร้อม 5 หมวด (audit + gate), ต่อ gate เข้า publish path, รัน audit จริง, แก้ถ้อยคำ/การแสดงผลบนเว็บเล็กน้อยถ้า audit พบ.
- **Out of scope (กันบวม):**
  - ไม่ retrain โมเดล — ใช้ artifact เดิมใน `models/` (ตัวเลข thesis อ้างอิงตัวนี้ + ผ่าน leakage audit แล้ว).
  - ไม่ทำ CI/CD เต็มรูปแบบ — gate ตอน publish ให้ความปลอดภัยพอสำหรับงานเดี่ยว; มี GitHub Actions cron อยู่แล้ว.
  - ไม่เพิ่ม feature ใหม่บนเว็บ — แค่ตรวจ/ปรับการสื่อสารที่มีอยู่.

## สถาปัตยกรรม

โมดูลใหม่ `scripts/readiness/` (numpy/pandas-only ตาม pattern `scripts/analysis/`, ทุกไฟล์มี `_selftest`):

```
scripts/readiness/
  checks.py        # CheckResult(name, category, status, detail, blocking)
                   #   status ∈ {PASS, WARN, FAIL}; blocking=True = ห้าม publish ถ้า FAIL
  data_quality.py  # เช็ค upstream: NaN, domain guard (in_training_domain), ช่วงค่า feature, MJO-impute
  audit.py         # รันทุกเช็ค → docs/readiness/AUDIT-YYYY-MM-DD.md (go/no-go report)
  gate.py          # รันเฉพาะ blocking subset → exit 0/1
```

**CheckResult** = dataclass เดียว ใช้ได้ทั้ง audit (แสดงทุกหมวด) และ gate (กรอง `blocking=True`).
audit/gate **อ่าน** `docs/forecast.json`, `docs/forecast_provinces.json`, `outputs/analysis/*`, `models/*`
— **ไม่ retrain** (เร็ว, deterministic).

## 5 หมวดการตรวจ

| หมวด | คำถาม | เช็คตัวอย่าง | blocking? |
|---|---|---|---|
| 1. Freshness | ข้อมูลสดจริงไหม | วัด **`generated_at - issue_date`** ≤ 30 วัน (ข้อมูลล้าหลังตอนสร้าง) — แยก demo(gap ~898 วัน) จากพยากรณ์สด(ERA5 ล่าช้า ~16 วัน) ; ไม่ใช่ `วันนี้-issue_date` ที่บล็อกพยากรณ์สดผิดๆ. + generated_recent (WARN) | ✅ |
| 2. Plausibility | ตัวเลขดูแปลกไหม | สัดส่วนจังหวัด High พร้อมกัน (WARN เท่านั้น ‼️ ไม่ block — อาจเป็นสัญญาณ El Niño จริง), `ratio_vs_normal` ≤ เพดาน (WARN) | WARN |
| 3. Data quality | วัตถุดิบครบ/ดีไหม | NaN เกินเกณฑ์, domain guard (`in_training_domain`), feature อยู่ในช่วงเทรน, MJO-impute flag | บางตัว ✅ |
| 4. Skill/calibration | โมเดลแม่นกว่าเดาไหม | BSS vs climatology ยังเป็นบวก (อ่านจาก outputs/analysis); reliability ไม่ bias สูงผิดปกติ | WARN |
| 5. Risk communication | เว็บสื่อสารไม่ชวนเข้าใจผิดไหม | UI ระบุ "ความน่าจะเป็นการเกิด ไม่ใช่ความรุนแรง"; โชว์ `issue_date` เด่น | WARN→fix |

## แผนการทำงาน (3 เฟส ตามลำดับ)

### เฟส 1 — per-province operational mode (ซ่อม root cause)
มิเรอร์ regional: ให้ `build_provinces_features()` / `build_forecast()` รับโหมด operational
(climatology แช่แข็ง + `data/raw_recent/`) แทนการใช้ `iloc[-1]` จาก dataset เทรนเต็ม.
ต้องมี **operational parity selftest** (เหมือน regional: feature row จาก clim+ข้อมูลย่อย = dataset) กัน train/serve mismatch.
ผล: `forecast_provinces.json` มี `issue_date` ปัจจุบัน + `data_through`.

### เฟส 2 — ชุดเช็ค 5 หมวด (audit + gate)
เขียน `checks.py`, `data_quality.py`, `audit.py`, `gate.py` ตามสถาปัตยกรรมข้างบน. ทุกตัวมี `_selftest`.

### เฟส 3 — ต่อ gate เข้า publish + รัน audit จริง
- `validate_contract.py` / `publish_bridge.py` เรียก `gate.py` (gate ใหม่ = superset ของ validator เดิม).
- abort การ distribute ถ้า gate FAIL (hard gate มีอยู่แล้ว แค่เสริมเช็ค).
- รัน `audit.py` → review รายงาน; แก้จนเขียวก่อน publish ของจริง.

## เกณฑ์ "พร้อม Production" (success criteria)
1. `issue_date` เป็นปัจจุบัน (≤ ~10 วัน) — ไม่ใช่ 2023.
2. ไม่ใช่ทุกจังหวัด/ทุก lead = "สูงมาก" พร้อมกันแบบ degenerate.
3. ข้อมูลดิบครบ ไม่มี NaN ผิดปกติ; feature อยู่ในโดเมนเทรน.
4. BSS vs climatology ยังเป็นบวก (อ่านจาก analysis ที่มีอยู่).
5. เว็บสื่อ "โอกาสเกิด" + โชว์ issue_date เด่น.
6. **gate พิสูจน์ได้:** ป้อน contract เสีย (เช่น issue_date 2023, all-High) แล้ว gate ต้อง FAIL/บล็อก (มี selftest ครอบ).

## Testing
- ทุกโมดูล readiness มี `_selftest` (รวม negative tests: ป้อนข้อมูลเสียต้อง FAIL).
- operational parity selftest ของ per-province (กัน train/serve mismatch).
- รัน `validate_contract.py` + app tests เดิมต้องยังผ่าน (ไม่ทำของเดิมพัง).

## Manual steps (ผู้ใช้ทำเอง — ตามที่ตั้งไว้แล้ว)
git push, secret `CDSAPIRC`, GitHub Pages — ไม่เปลี่ยนจากเดิม.
