# ดีไซน์: Pipeline Bridge (เชื่อมผลทำนายจริงของ AI → Frontend ของผู้ใช้)

วันที่: 2026-06-14
สถานะ: อนุมัติแล้ว (รอ review สเปกก่อนเขียน plan)
บริบท: **subsystem #3** — pipeline ที่ทำให้ผลทำนายจริงของ DeepSeek (per-province, lead 2–6)
ส่งถึง frontend (`HeatMAP_Frontend`) ได้อย่าง **ทำซ้ำได้ + ตรวจสอบแล้ว** ทั้ง dev และ prod

> สถาปัตยกรรม decoupled เดิม (static `forecast_provinces.json`, ไม่มี API server) ถูกล็อกไว้แล้ว
> งานนี้คือ "กลไกท่อ" ที่ขาด: **predict → validate → distribute** + เส้นทาง publish ขึ้น prod

---

## 1. เป้าหมาย
มีคำสั่งเดียวที่ regenerate ผลทำนายจริง, **ตรวจ contract ให้ถูกต้องก่อนถึงผู้ใช้**, แล้วส่งไป
- **dev sink:** `HeatMAP_Frontend/public/forecast_provinces.json` (same-origin, ใช้ตอนรัน local)
- **prod sink:** repo สาธารณะเฉพาะ `heatwave-contract` → GitHub Pages → frontend บน Vercel fetch ที่ runtime

## 2. Non-goals
- ไม่เพิ่ม live ERA5 ingestion — `issue_date` คงที่ที่ **2023-12-31** (วันล่าสุดที่ feature ครบ); งานนี้คือ delivery ไม่ใช่ data freshness
- ไม่ตั้ง backend/API server (คงหลัก decoupled static)
- ไม่ push `DeepSeek_Heatwave` ขึ้นสาธารณะ — repo วิจัยอยู่ local/private; เผยแพร่เฉพาะ contract JSON
- ไม่ออกแบบ UI ใหม่ (แตะ frontend แค่ version guard + env + deploy)

## 3. สถานะที่ verify แล้ว (2026-06-14)
- `python scripts/predict_provinces.py` รันซ้ำได้ → เขียน `docs/forecast_provinces.json` (77 จังหวัด, model `logistic_balanced_cal`, `generated_at` อัปเดตเป็นปัจจุบัน)
- self-test parity ผ่าน: 28 feature × 843,689 แถว ตรง `dataset_provinces.parquet`
- prerequisites ครบ: `models/heatwave_prov_lead{2..6}.pkl`, `data/provinces.csv`, `data/processed/dataset_provinces.parquet`, `outputs/analysis/provinces_per_province_bss.csv`
- ทั้งสอง repo เป็น git local-only (ยังไม่มี remote); `gh` auth = `MCTEEKUNG`

## 4. Components

### 4.1 Validator — `DeepSeek_Heatwave/scripts/validate_contract.py`
gate ที่กันผลผิดไม่ให้ถึงผู้ใช้ (importable function `validate_contract(obj) -> list[str]` + CLI ที่รับ path).
ตรวจ:
- `schema_version == 1` ; `model` เป็น str ไม่ว่าง ; `generated_at` parse เป็น datetime ได้
- `n_provinces == len(provinces) == 77`
- ต่อจังหวัด: มี key `id,code,name_th,name_en,region,lat,lon,issue_date,forecasts` ครบและ type ถูก ; `lat∈[-90,90]`, `lon∈[-180,180]` ; `issue_date` parse เป็น date ได้ ; `id` ไม่ซ้ำ
- `forecasts`: lead เป็นเซ็ต `{2,3,4,5,6}` พอดี (ไม่ขาดไม่เกิน)
- ต่อ forecast: `probability∈[0,1]`, `climatology_base_rate∈[0,1]`, `ratio_vs_normal≥0`, `risk_level_en∈{Low,Normal,Elevated,High}`, `risk_level_th` ไม่ว่าง
- **soft warning** (ไม่ fail): `generated_at` เก่ากว่า 7 วัน
CLI: รวบรวม error ทั้งหมดแล้ว print ; exit `1` ถ้ามี error, `0` ถ้าผ่าน
**ขั้นแรกของ implementation:** รัน validator กับ `docs/forecast_provinces.json` ปัจจุบัน → ต้องผ่าน (กัน gate เข้มเกินจนตี valid data ตก)

### 4.2 Bridge orchestrator — `DeepSeek_Heatwave/scripts/publish_bridge.py`
ลำดับ: `predict_provinces.predict()` → `validate_contract` (มี error = abort, ไม่ sync) → copy ไป sink
- path ของ sink **config ได้** (ไม่ hardcode coupling): dev default `../HeatMAP_Frontend/public/forecast_provinces.json` (override ด้วย env `BRIDGE_FRONTEND_DIR` หรือ arg) ; prod default `../heatwave-contract/forecast_provinces.json`
- flags:
  - (default) predict + validate + sync → dev sink
  - `--no-predict` : ข้าม predict, validate + sync ของไฟล์ที่มีอยู่
  - `--publish` : sync เข้า contract repo ด้วย แล้ว `git add forecast_provinces.json && git commit && git push` (ดัน Pages)
- print สรุปทุกขั้น (จำนวนจังหวัด, ผล validate, sink ที่เขียน)

### 4.3 Contract repo — `C:\Users\ASUS\heatwave-contract` (ใหม่, สาธารณะ)
ไฟล์: `forecast_provinces.json`, `CONTRACT.md` (เอกสาร schema v1 + ความหมาย field + risk mapping), `README.md`, `index.html` (landing สั้น optional).
ขั้นตั้ง: `gh repo create heatwave-contract --public` → push → เปิด Pages (branch `main`, root) → URL `https://mcteekung.github.io/heatwave-contract/forecast_provinces.json`
**verify ก่อนใช้:** `curl -I <url>` ต้อง 200 + header `access-control-allow-origin: *` (Pages ส่งให้ static) ก่อน repoint frontend

### 4.4 Consumer — `HeatMAP_Frontend`
- `services/deepseekContract.ts`: เพิ่ม **version guard** — ถ้า `schema_version !== 1` ให้ throw ข้อความชัด (กัน schema ใหม่ทำ UI พังเงียบ)
- prod `EXPO_PUBLIC_FORECAST_URL` = Pages URL (เอกสารใน `.env.example` / `.env.production`) ; dev คง same-origin `public/`
- deploy `dist/` (`bunx expo export -p web`) ขึ้น Vercel

### 4.5 Docs
- DeepSeek `docs/RUNBOOK.md`: คำสั่ง bridge (dev sync, `--publish`)
- frontend `README.md`: วิธีตั้ง `EXPO_PUBLIC_FORECAST_URL` (dev=same-origin, prod=Pages URL) + deploy

## 5. Data flow
```
predict_provinces.predict()  ──>  docs/forecast_provinces.json
        │                                   │
        └── validate_contract (GATE) ───────┤  (fail → abort, ไม่มีอะไรถูกส่ง)
                                            │
              ┌─────────────────────────────┴───────────────────────────┐
        dev sink                                                    prod sink (--publish)
HeatMAP_Frontend/public/forecast_provinces.json          heatwave-contract/ → git push → Pages
        │                                                                 │
   expo web (local, same-origin)                          Vercel frontend fetch (EXPO_PUBLIC_FORECAST_URL)
```

## 6. Error handling
- predict fail → orchestrator exit non-zero, ไม่แตะ sink ใด ๆ
- validate fail → print error ทั้งหมด, exit non-zero, ไม่ sync
- sink path ไม่มี → error ชัด (บอก path ที่คาด + วิธี override)
- `--publish` กับ contract repo ที่ไม่มี/ไม่มี remote → error ชัด ก่อน push
- consumer fetch fail / `schema_version` ผิด → throw ข้อความ user-readable (มี `loadContract` catch + guard ใหม่)

## 7. Testing / verification
- **validator unit (pytest):** valid contract ผ่าน ; แต่ละ mutation (lead ขาด, prob>1, n ไม่ตรง, risk แปลก, schema!=1) ต้องโดนจับ
- **orchestrator:** รัน dev sync จริง → diff dev sink กับ `docs/` ต้องตรง byte ; `--no-predict` ไม่เปลี่ยน `generated_at`
- **consumer:** vitest adapter เดิม + เคส version guard (schema!=1 → throw)
- **prod:** `curl -I` Pages URL 200 + ACAO:* ; เปิด Vercel URL ใน chrome-devtools → console 0 error, map 77 จังหวัด, network request ไป Pages URL สำเร็จ
- **go/no-go (stale dates):** ก่อน deploy สาธารณะ เปิด browser ดู `target_date` ที่ render (จะเป็น ม.ค.–ก.พ. 2024) → ผู้ใช้ตัดสิน: ship as-is / ใส่ banner "historical model run" / เปลี่ยนเป็น label "+2…+6 สัปดาห์" ซ่อนวันที่สัมบูรณ์

## 8. Sequencing
1. **Local-first** (4.1 + 4.2 dev-sink): validator + orchestrator → verify dev sync ครบ (deps ศูนย์, ของหลัก)
2. **Prod** (4.3 + 4.4): contract repo + Pages + version guard + Vercel — outward-facing, ทำหลัง, verify ด้วย curl/browser
3. go/no-go เรื่อง stale dates คั่นก่อน "public deploy" เท่านั้น (ไม่บล็อกการ build)

## 9. การตัดสินใจที่ล็อก
- full bridge = local sync command + prod publish
- prod publish = dedicated public **contract repo → Pages** ; DeepSeek คง private/local ; frontend บน Vercel fetch Pages URL ที่ runtime
- validate เป็น hard gate ก่อน distribute ; staleness เป็น soft warning เท่านั้น
- orchestrator อยู่ฝั่ง producer (DeepSeek) ; sink path config ได้ (ไม่ hardcode coupling)
