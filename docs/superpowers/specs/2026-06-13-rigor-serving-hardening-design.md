# ดีไซน์: Rigor + Serving Hardening — DeepSeek_Heatwave

วันที่: 2026-06-13
สถานะ: ร่างเพื่อ review (ยังไม่อนุมัติ)
บริบท: งานวิจัย / วิทยานิพนธ์ — ปักหลักพัฒนาที่โปรเจกต์นี้ (เลือกแล้วเทียบกับ Heatwave_AI)
แนวทาง: **A — Rigor-first** (แก้ leak ให้จบ → re-validate → แล้วทำ serving บางๆ ทับบนโมเดลที่แก้แล้ว)

---

## 1. เป้าหมาย

ทำให้ DeepSeek_Heatwave:
- **(ก) ป้องกันได้เชิงวิชาการ** — ไม่มี leakage ที่ทำให้สกิลที่รายงานเฟ้อ ; ตัวเลขที่เคลม (ชนะ climatology + persistence) ยืนได้หลังตัด leak
- **(ข) มี serving path ที่ run ออก forecast ปัจจุบันได้จริง** แบบ on-demand (รันมือ) — ไม่ต้องมี automation/scheduler

ทั้งสองอย่างต้องอยู่บน pipeline เดียวกัน (train/serve parity) ที่มีอยู่แล้ว

## 2. Non-goals (กันงานบาน — YAGNI)

- ไม่ดึงโค้ด/ข้อมูล/ฟีเจอร์ใดๆ จาก Heatwave_AI
- ไม่แตะ Web UI / Streamlit (`app/`) — output contract (`docs/forecast.json`) คงเดิม
- ไม่สร้าง automation/cron/CI scheduler (รันมือพอ ; โครง `predict.py operational` มีอยู่แล้ว)
- ไม่ดาวน์โหลด ERA5 archive 30 ปีใหม่ (raw .nc เดิมครบ — rebuild จากของเดิมได้)
- ไม่เพิ่ม data source ใหม่ (SST / geopotential / NDVI = future work ตาม spec 2026-06-08)

## 3. สภาพปัจจุบัน (ผลสำรวจ ระดับโค้ด)

- `dataset.csv`: 10,957 วัน, 30 ปีเต็ม (1994–2023, ทุกเดือน), missing สูงสุด 0.44% (NaN เชิงโครงสร้างจาก lookback), ไม่มีวันซ้ำ
- raw ครบสำหรับ rebuild: Tmax 60 ไฟล์ (30 ปี × h1+h2), soil 120 ไฟล์, indices (`mjo_rmm.csv`, `nino34.csv`) อัปเดตล่าสุด
- `predict.py` สุกแล้ว: operational mode (frozen climatology + ข้อมูลล่าสุด), reuse `build_feature_table` (train/serve parity self-test ผ่าน), impute MJO ที่ค้าง, เตือน out-of-domain, risk band เทียบ base rate, เขียน `docs/forecast.json`
- โมเดล `models/*.pkl` ลงวันที่ปัจจุบัน (เต็มปีแล้ว) **แต่ยังเทรนบนข้อมูลที่ติด leak ทั้งสองตัว** → ต้อง retrain หลังแก้
- git: branch `main`, working tree สะอาด (เหลือ PNG untracked)

## 4. Leakage ที่ต้องแก้ (Phase 1)

### R2 — `in_hw_today` เป็น feature ที่มองอนาคต *(สำคัญสุด)*

`heatwave_target.flag_heatwaves` (heatwave_target.py:70–93) นับความยาว run แบบ `fwd + bwd - arr`
คือใช้วัน **t+1, t+2 …** มาตัดสินว่าวัน t อยู่ในคลื่นความร้อนหรือไม่

- ถูกต้องสำหรับ **label/target** (ground truth ว่าคลื่นความร้อนเกิดจริงในหน้าต่างอนาคต) → **คงไว้**
- แต่ output เดียวกันถูกใช้เป็น **feature** `in_hw_today` (build_dataset.py:175, `f["in_hw_today"] = d["hw_rm"]`)
  ซึ่ง corr กับ target = +0.20 (อันดับ 3) → ปัญหา:
  1. **สกิลที่รายงานเฟ้อ** (ใช้ข้อมูลที่ ณ วันออกพยากรณ์ยังไม่รู้)
  2. **บั๊ก train/serve**: ค่าของวันล่าสุดใน `predict.py` คำนวณไม่ได้/ไม่ตรง — self-test ปัจจุบันต้องตัด 7 วันท้ายทิ้งเพราะเหตุนี้

**⚠️ ขอบเขตที่กว้างกว่าที่เห็น:** `in_hw_today` ไม่ได้เป็นแค่ feature โมเดล — มันยังป้อน **persistence baseline**
(train.py:161 `state = sub["in_hw_today"]` → :173 `persistence_probs(state[tr], y_tr, state[te])`)
→ การแก้ต้องทำที่ **ต้นทาง** (`build_dataset`) เพื่อให้ทั้ง feature โมเดล **และ** persistence state ใช้นิยามใหม่พร้อมกัน
(ห้ามจบลงที่ตัวหนึ่ง trailing อีกตัว fwd+bwd)

**การแก้ (eliminate outright — ไม่ gate):** นิยาม feature `in_hw_today` ใหม่เป็น **trailing-only** —
"ความยาว hot-streak ที่ต่อเนื่องและจบ ณ วัน t (ใช้เฉพาะวัน ≤ t) ≥ MIN_RUN (3)"
→ คำนวณได้ ณ วันออกพยากรณ์จริง, leak-free, ความหมายซื่อสัตย์ ("ยืนยันแล้วว่ากำลังอยู่ในคลื่นความร้อน ณ วันนี้")
*ทำไมไม่ gate แบบ R1:* นี่เป็นบั๊ก serve consistency จริง (ค่าวันล่าสุดผิด/คำนวณไม่ได้) → ต้องแก้ไม่ว่าผลต่อสกิลจะเท่าไหร่

- เพิ่ม helper `trailing_hot_streak()` ใน `heatwave_target.py` (แยกจาก `flag_heatwaves` ที่ใช้ทำ label)
- `hot_frac7` (build_dataset.py:176) ใช้ `hot_rm` = same-day hot flag + rolling ย้อนหลัง → **leak-free อยู่แล้ว ไม่แตะ**
- เพิ่ม self-test ใน `build_dataset.py`: "แก้ค่า Tmax วันอนาคต → `in_hw_today` ของวันอดีตต้องไม่ขยับ" (ตอนนี้ test นี้จะ fail = หลักฐานว่ามี leak)

### R1 — Percentile-label leak (เกณฑ์ p90 เห็น test ปนมา)

`doy_window_percentile` (build_dataset.py:237) คำนวณเกณฑ์ p90/p95 ราย doy บนข้อมูล **30 ปีเต็มรวมปี test**
แล้ว bake label (`y_rm*`) ลง `dataset.csv` ; `train.py` อ่าน label ที่ bake มาใช้ใน CV
→ label ของ fold ทดสอบถูกนิยามด้วยเกณฑ์ที่ "เห็น" test (low-variance climatological leak)
*หมายเหตุ:* leak ระดับเดียวกับที่ Heatwave_AI เปิดเผยใน `DATASET_PROFILE.md` ; DeepSeek ยังไม่ได้บันทึก

**เหตุผลที่ R1 ต่างจาก R2 (ทำไม gate ได้):** BSS วัดเทียบ **climatology baseline ที่ใช้ label ชุดเดียวกัน** กับโมเดล
→ leak ที่ระดับ "นิยาม threshold" หักล้างกันเป็นส่วนใหญ่ใน skill **ratio** (มันขยับ Brier สัมบูรณ์ ไม่ใช่ BSS)
prior ว่า ΔBSS แทบเป็นศูนย์จึงแข็งแรง → **วัดก่อน ค่อยตัดสินว่าต้อง refactor ไหม** (rigorous กว่าการ refactor มั่วเพราะได้ตัวเลขจริง, ถูกกว่า, เสี่ยงน้อยกว่า)

**การแก้ — measure-first / gated:**
- **Serving:** freeze เกณฑ์จาก history ทั้งหมด = **ถูกต้องเชิง operational** (predict.py ทำอยู่แล้วผ่าน `load_climatology`/`CLIM_FILE`) → คงไว้ + ระบุว่าตั้งใจ
- **Gate (วัดครั้งเดียว):** rebuild label หนึ่งชุดด้วยเกณฑ์ p90 แบบ **leave-each-test-block-out** (fit threshold โดยตัดบล็อก test ของแต่ละ fold ออก) → recompute pooled BSS → เทียบกับ bootstrap CI ที่มีใน `results_master`
  - **ΔBSS อยู่ใน CI** (คาดว่าใช่) → **document ขนาดของ leak + คง frozen-all-history labels** ไม่ต้อง refactor
  - **ΔBSS หลุด CI** → ค่อยทำ per-fold elimination เต็มรูป (helper "label จาก Tmax series" reuse ทั้ง build+CV ; `dataset.csv` มีคอลัมน์ `tmax_rm` รายวันอยู่แล้ว recompute per-fold ได้โดยไม่โหลด NetCDF ซ้ำ)
- *gate นี้เป็น "decision procedure" ที่นิยามชัด ไม่ใช่ความกำกวม* — ผลลัพธ์คือตัวเลข ΔBSS + ข้อสรุปที่ตามมาแน่นอน

## 5. Re-validation (ปิดท้าย Phase 1)

1. rebuild `dataset.csv` (หลังแก้ R2 — `in_hw_today` trailing) + เซฟ climatology freeze ใหม่
2. rerun `train.py` + ชุด analysis เดิมทั้งหมด + รัน R1 gate (วัด ΔBSS leave-block-out หนึ่งครั้ง)
   (`scripts/analysis/`: bootstrap_ci, stats/FDR, ablation, calibration, feature_selection, permutation_importance, skill_by_season, regime_strat, make_report)
3. **Gate:** โมเดลหลัก/production ยังต้องชนะ **climatology และ persistence** อย่างมีนัยสำคัญ (q < 0.05 หลัง BH-FDR) ที่ lead ที่เคยเคลม
   - ⚠️ **persistence baseline จะขยับ** เพราะ R2 เปลี่ยน `in_hw_today` ที่มันพึ่งอยู่ → "beats persistence" รอบนี้เทียบของที่สะอาดทั้งคู่ ; ตัวเลข persistence ที่ต่างจากเดิม = **คาดได้ ไม่ใช่ regression** (เดิม baseline ก็ติด forward leak)
   - ถ้า lead ใดหลุด significance หลังตัด leak → **บันทึกตรงๆ** (นั่นคือสกิลจริง ไม่ปิดบัง) แล้วปรับ README/รายงานตาม
4. อัปเดต `outputs/analysis/results_master.md` + เพิ่ม **integrity note** (เปิดเผย R1/R2 + การตัดสินใจ + ΔBSS ที่วัดได้)

## 6. Serving (Phase 2 — ทับบนโมเดลที่แก้แล้วเท่านั้น)

| รหัส | งาน | รายละเอียด |
|------|-----|-----------|
| S1 | `scripts/download_recent.py` | ดึง ERA5 Tmax+soil ล่าสุด ~90 วัน (เผื่อ ERA5T latency ~5 วัน) เข้า `data/raw_recent/` + MJO/Niño ล่าสุด ; reuse logic hourly→daily เดิม ; resume + validate ได้ |
| S2 | retrain final | `train_final.py` บน dataset ที่แก้แล้ว → ยืนยัน `train_issue_doy_min/max` ครอบเต็มปี (เลิกเตือน out-of-domain ครึ่งปีหลัง) |
| S3 | end-to-end smoke | รัน `predict.py operational` จาก `raw_recent/` → `forecast.json` ; **กระชับ parity self-test ให้ไม่ต้องตัด 7 วันท้าย** (หลักฐานว่า R2 หาย) |
| S4 | runbook + docs | `docs/RUNBOOK.md` ขั้นตอน "ออก forecast ปัจจุบันด้วยมือ" + อัปเดต README สถานะ + integrity note |

## 7. Acceptance criteria

1. self-test ใหม่ผ่านครบ: `in_hw_today` backward-only ; `predict.py test` parity ผ่าน **โดยไม่ตัด 7 วันท้าย**
2. ตาราง BSS **ก่อน/หลัง** ตัด leak + ข้อสรุป gate (beats climatology & persistence, q-FDR) ใน `results_master.md`
3. `python scripts/download_recent.py` → `python scripts/predict.py operational` วิ่งจบ → `forecast.json` issue date ล่าสุด **in_training_domain = true** ทุก lead
4. `docs/RUNBOOK.md` + integrity note + README สถานะใหม่ commit แล้ว
5. self-test ทุกโมดูล (`build_dataset`, `cv`, `train`, `train_final`, `predict`, `heatwave_target`) ผ่าน

## 8. ลำดับงาน & branch

ทำเป็นเส้นตรง **P1 → gate → P2** (serving พึ่งโมเดลที่แก้ leak แล้ว ห้ามสลับ)
แตก branch `feat/harden-rigor-serving` จาก `main` ; commit spec นี้ก่อนเริ่ม

1. R2: helper `trailing_hot_streak` + เปลี่ยน `in_hw_today` ที่ต้นทาง (`build_dataset`) ให้กระทบทั้ง feature โมเดล + persistence state + self-test leakage
2. rebuild dataset + rerun train + analysis ; รัน R1 gate (วัด ΔBSS leave-block-out) → ตัดสินตาม gate
3. integrity note + ปรับ README/results ตามผล gate
4. S2 retrain final → S1 download_recent → S3 smoke + ปรับ parity test → S4 runbook/docs

## 9. Risks & mitigations

- **R1 per-fold refactor บานปลาย** → คุมด้วย helper เดียว reuse ทั้ง build+CV ; ขอบเขตจำกัดที่อนุกรม regional-mean (1D, ถูก, ไม่แตะ grid/area-fraction ของ ablation)
- **`in_hw_today` ใหม่ทำสกิลลดเล็กน้อย** → เป็นความจริงที่ honest ; ถ้าลดมากผิดคาด แสดงว่าสกิลเดิมพึ่ง leak — ต้องรายงาน
- **S1 ต้องมี CDS key** (`~/.cdsapirc`) บนเครื่องที่รัน + ERA5T latency ~5 วัน → predict ใช้ issue date = วันล่าสุดที่ feature ครบ (มีตรรกะอยู่แล้ว)
- **โมเดล/ผลเดิมถูกทับ** → อยู่บน branch แยก + commit ก่อน/หลังแก้ เทียบ before/after ได้

## 10. การตัดสินใจที่ล็อกแล้ว (ไม่เปิด fork)

- R1 = **measure-first / gated** (วัด ΔBSS leave-block-out → ใน CI ก็ document, หลุด CI ค่อย eliminate per-fold) + คง frozen-all-history สำหรับ serving เสมอ
- R2 = `in_hw_today` → **trailing-only, eliminate outright** (แก้ที่ต้นทางให้กระทบทั้ง feature + persistence) ; label `flag_heatwaves` (fwd+bwd) **คงเดิม**
- Serving = **on-demand รันมือ** (ไม่มี scheduler)
- ไม่ re-download archive ; rebuild จาก raw เดิม
