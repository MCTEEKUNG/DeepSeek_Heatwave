# LINE OA — แจ้งเตือนความเสี่ยงคลื่นความร้อนรายสัปดาห์

วันที่: 2026-06-20
สถานะ: อนุมัติ design — รอเขียน implementation plan

## 1. วัตถุประสงค์

LINE Official Account ทำหน้าที่เป็น **ช่องแจ้งเตือนภาพรวมรายสัปดาห์** ของความเสี่ยงคลื่นความร้อนรายจังหวัด
ส่งหา **ผู้ติดตามทุกคน** (ไม่มีการติดตามรายจังหวัด/รายบุคคล) เพื่อบอกว่า "สัปดาห์นี้จังหวัดใดมีความเสี่ยงต่อสุขภาพ"
อธิบายความเสี่ยงสั้น ๆ และ **ดันผู้ใช้เข้าเว็บไซต์** เพื่อดูภาพรวมเต็มรูปแบบ

หลักการ:
- สอดคล้องวัตถุประสงค์โครงการ "ไม่ก่อความวุ่นวายโดยมิใช่เหตุ" — เตือนแบบคาดเดาได้ (รายสัปดาห์) ไม่สแปม
- เก็บข้อมูลส่วนบุคคลให้น้อยที่สุด: **ไม่เก็บ LINE user ID เลย** (ระบบ stateless 100%)
- LINE = ตัวกระตุ้น/สรุป ; เว็บไซต์ = แหล่งภาพรวมเต็ม (`https://heat-map-frontend.vercel.app/map`)

## 2. ขอบเขต

**ทำ:**
- Broadcast รายสัปดาห์หาผู้ติดตามทุกคน (LINE Broadcast API) — รายชื่อจังหวัดเสี่ยง + คำอธิบาย + ลิงก์เว็บ
- Webhook ตอบ on-demand (stateless) ผ่าน Rich menu 3 ปุ่ม
- ภาษาไทย, ใช้ horizon lead 2–4 เท่านั้น (5–6 ไม่แสดง ตามนโยบายสกิลของระบบ)

**ไม่ทำ (YAGNI):**
- ไม่มี subscription / ฐานข้อมูล / Supabase
- ไม่มี flow เลือกจังหวัดรายตัว
- ไม่มีการแจ้งเตือนด่วนระหว่างสัปดาห์ (เฉพาะสรุปรายสัปดาห์)
- ไม่เก็บประวัติผู้ใช้

## 3. สถาปัตยกรรม

```
GitHub Actions (รายสัปดาห์, ต่อจาก forecast.yml)
   forecast JSON ─► scripts/broadcast_weekly.py ─► LINE Broadcast API ─► ผู้ติดตามทุกคน

ผู้ใช้แตะ Rich menu ─► LINE Platform ─► Vercel Webhook (api/webhook.py, Python, stateless)
   ├─ "ความเสี่ยงสัปดาห์นี้"  → อ่าน forecast JSON ล่าสุด → ตอบรายชื่อจังหวัดเสี่ยง + ลิงก์เว็บ
   ├─ "ดูแผนที่ภาพรวม"       → URI action เปิดเว็บไซต์ตรง ๆ (ไม่ผ่าน server)
   └─ "วิธีอ่าน / เกี่ยวกับ"  → ข้อความอธิบายระดับความเสี่ยง + ข้อจำกัด
```

ทั้ง webhook และ broadcast **อ่าน forecast JSON ล่าสุดจาก URL คงที่** (publish ผ่าน GitHub Pages
หรือ raw URL ของ repo เช่น `forecast-latest.json`) → ข้อมูลตรงกับที่ Actions รันเสมอ ไม่มี state ฝั่ง server

เหตุผลที่แยก 2 ส่วน: webhook ต้องออนไลน์ตลอดแต่ทำงานเบามาก (เหมาะ serverless ฟรี) ;
broadcast เป็นงานตามเวลา ใช้ GitHub Actions ที่มี workflow รายสัปดาห์อยู่แล้วคุ้มกว่า

## 4. Components

```
api/
  webhook.py            # Vercel entrypoint: verify signature → route events → reply
linebot/
  __init__.py
  config.py             # อ่าน env (LINE token/secret, FORECAST_URL, WEBSITE_URL) — ไม่มี secret ใน repo
  forecast_source.py    # ดึง forecast JSON ล่าสุด (HTTP) + cache ในหน่วยความจำต่อ invocation
  risk.py               # logic คัด/จัดอันดับจังหวัดเสี่ยง (pure, ไม่มี I/O)
  messages.py           # สร้าง Flex/text จากผลของ risk.py (pure)
  provinces.py          # โหลด 77 จังหวัดจาก data/provinces.csv + จัดกลุ่มภาค (pure)
  handlers.py           # logic ต่อ event type (follow / postback)
  richmenu.py           # สร้าง + ผูก rich menu (รันครั้งเดียวตอน setup)
  tests/
    test_risk.py
    test_messages.py
    test_provinces.py
    test_webhook_signature.py
scripts/
  broadcast_weekly.py   # งานรายสัปดาห์ (GitHub Actions เรียก): อ่าน JSON → risk.py → Broadcast API
vercel.json             # route /api/webhook
requirements-linebot.txt
```

หลักการแยกความรับผิดชอบ:
- `risk.py`, `messages.py`, `provinces.py` = **pure functions** (input → output) เทสต์ง่ายไม่ต้อง mock
- `forecast_source.py` = I/O เดียว (HTTP) → mock ในเทสต์
- `handlers.py` = ประกอบ logic ; `webhook.py` = แค่ verify signature + route + reply
- `broadcast_weekly.py` ใช้ `risk.py` + `messages.py` ซ้ำ (ตรรกะคัดจังหวัดชุดเดียวกับ webhook)

## 5. ตรรกะการคัดจังหวัดเสี่ยง (risk.py)

อ้างนิยามระดับความเสี่ยงจาก `scripts/predict.py` (`RISK_BANDS`):

| ระดับ (th) | เงื่อนไข | ลำดับ (ordinal) |
|---|---|---|
| ต่ำ | prob < 0.05 หรือ ratio < 0.75 | 0 |
| ปกติ | ratio < 1.5 | 1 |
| ค่อนข้างสูง | ratio < 2.5 | 2 |
| สูง | ratio ≥ 2.5 | 3 |

อัลกอริทึม:
1. ต่อจังหวัด: พิจารณาเฉพาะ `forecasts` ที่ `lead_weeks ∈ {2,3,4}`
2. หาค่า **ระดับความเสี่ยงสูงสุด** ในช่วง lead 2–4 (ใช้ ordinal เป็นตัวเทียบ ; tie-break ด้วย ratio สูงสุด)
3. คัดเฉพาะจังหวัดที่ระดับสูงสุด ≥ "ค่อนข้างสูง" (ordinal ≥ 2)
4. เรียง: สูง ก่อน ค่อนข้างสูง ; ภายในระดับเดียวกันเรียง ratio มาก→น้อย
5. ผลลัพธ์: list ของ `{province_id, name_th, level_th, max_ratio, lead_at_max}`

จำกัดการแสดงผล: ถ้าเกิน `MAX_LIST = 12` จังหวัด → แสดง 12 รายการแรก + ต่อท้าย
"…และอีก N จังหวัด — ดูทั้งหมดบนเว็บ"

## 6. รูปแบบข้อความ

### 6.1 Broadcast รายสัปดาห์ (Flex หรือ text)
```
🌡️ เฝ้าระวังคลื่นความร้อน — สัปดาห์ที่ [issue_date]
จังหวัดที่มีความเสี่ยงต่อสุขภาพในอีก 2–4 สัปดาห์:

🔴 สูง: เชียงใหม่, ลำพูน, ตาก, ...
🟠 ค่อนข้างสูง: น่าน, แพร่, ... (และอีก N จังหวัด)

ℹ️ "สูง" = โอกาสเกิดคลื่นความร้อนมากกว่าปกติ ≥ 2.5 เท่า
👉 ดูแผนที่ภาพรวมทุกจังหวัด: https://heat-map-frontend.vercel.app/map
```
- กรณีไม่มีจังหวัดถึงเกณฑ์: "✅ สัปดาห์นี้ไม่มีจังหวัดที่ความเสี่ยงสูงผิดปกติ ดูรายละเอียดได้ที่เว็บไซต์"
- ถ้า forecast JSON มี `warnings` (เช่น MJO ไม่อัปเดต) → แนบหมายเหตุท้ายข้อความ

### 6.2 ตอบ "ความเสี่ยงสัปดาห์นี้" (webhook)
เนื้อหาเดียวกับ 6.1 (ใช้ `messages.py` ตัวเดียวกัน)

### 6.3 "วิธีอ่าน / เกี่ยวกับ"
อธิบายระดับความเสี่ยง (ต่ำ/ปกติ/ค่อนข้างสูง/สูง = เทียบอัตราต่อค่าปกติ) +
ข้อจำกัด (พยากรณ์ sub-seasonal 2–4 สัปดาห์ เป็นการบ่งชี้ความเสี่ยง ไม่ใช่การฟันธง) + ลิงก์เว็บ

## 7. Rich menu

3 ปุ่ม (สร้างครั้งเดียวด้วย `richmenu.py`):

| ปุ่ม | action |
|---|---|
| 🌡️ ความเสี่ยงสัปดาห์นี้ | `postback` data=`action=weekly` |
| 🗺️ ดูแผนที่ภาพรวม | `uri` = `https://heat-map-frontend.vercel.app/map` |
| ℹ️ วิธีอ่าน / เกี่ยวกับ | `postback` data=`action=about` |

เมื่อมี follow event → ส่งข้อความต้อนรับ + (rich menu ผูกเป็น default menu ของ OA อยู่แล้ว)

## 8. Error handling

- **Signature**: ตรวจ `X-Line-Signature` (HMAC-SHA256 ด้วย channel secret) ทุก request ; ไม่ผ่าน → HTTP 401, ไม่ประมวลผล
- **forecast JSON ดึงไม่ได้/ว่าง/parse ไม่ได้**: webhook ตอบ "ขออภัย ข้อมูลยังไม่พร้อม ดูที่เว็บไซต์ [ลิงก์]" + log ; broadcast → ไม่ส่ง, exit non-zero ให้ Actions เห็น
- **broadcast ล้มเหลว**: เป็น step แยกใน workflow ที่ `continue-on-error` ไม่ให้ทำพยากรณ์/commit พัง
- **ไม่มีจังหวัดถึงเกณฑ์**: ข้อความเชิงบวก (ไม่ใช่ error)
- **MJO/warnings**: แนบหมายเหตุ ไม่ปิดบัง
- **LINE API error (rate/4xx/5xx)**: log + retry เบื้องต้นใน broadcast (ไม่ retry ใน webhook — LINE ส่ง event ซ้ำเอง)

## 9. Testing

- `test_risk.py`: คัด/จัดอันดับจาก forecast JSON ตัวอย่างจริง (`forecast_2026-05-31.json`) — ตรวจ max lead 2–4, เกณฑ์ ≥ ค่อนข้างสูง, การเรียง, การตัดที่ MAX_LIST
- `test_messages.py`: รูปแบบข้อความถูก (มีลิงก์เว็บ, หมายเหตุ warning, กรณีไม่มีจังหวัด)
- `test_provinces.py`: โหลด 77 จังหวัด, จับคู่ id↔ชื่อถูก
- `test_webhook_signature.py`: valid → ผ่าน, invalid → 401
- `forecast_source.py`: mock HTTP (สำเร็จ / 404 / JSON เสีย)
- e2e เบา: ทดสอบยิงจริงผ่าน `line-bot` MCP หรือผู้ใช้กดเองหลัง deploy

## 10. Deploy & secrets

- **Vercel**: เพิ่ม `api/webhook.py` (แยก project หรือรวมกับ frontend เดิมก็ได้ — ตัดสินตอน implement)
- **env vars** (Vercel + GitHub Actions secrets — ไม่ลง git):
  - `LINE_CHANNEL_SECRET`, `LINE_CHANNEL_ACCESS_TOKEN`
  - `FORECAST_URL` (URL ของ forecast JSON ล่าสุด)
  - `WEBSITE_URL` = `https://heat-map-frontend.vercel.app/map`
- `.gitignore` เพิ่ม `.env`
- **ขั้นตอน setup (ผู้ใช้ทำบนเว็บ + ผู้ช่วยไกด์):**
  1. สร้าง Provider + Messaging API channel ที่ developers.line.biz
  2. คัดลอก Channel secret + Issue Channel access token
  3. เปิด Use webhook = ON, ปิด Auto-reply/Greeting
  4. ใส่ env ใน Vercel → deploy → ตั้ง Webhook URL = endpoint ที่ได้
  5. รัน `richmenu.py` ครั้งเดียวเพื่อสร้าง+ผูก rich menu
  6. ทดสอบ webhook (verify) + ทดสอบ broadcast (dry-run ก่อนจริง)
- ⚠️ token/secret = รหัสผ่าน : ไม่วางในแชต/ไม่ commit ; เพิกถอน/หมุนได้เสมอ

## 11. การเผยแพร่ forecast JSON

webhook/broadcast ต้องเข้าถึง forecast JSON ล่าสุดผ่าน URL คงที่ ทางเลือก (ตัดสินตอน implement):
- (ก) GitHub Actions เขียน `forecast-latest.json` ไป GitHub Pages (`docs/`) — ใช้ Pages ที่มีอยู่
- (ข) อ่านจาก raw.githubusercontent ของไฟล์ใน repo

เลือกแนวที่เสถียร + cache ได้ และไม่ผูกกับ path ที่เปลี่ยนตามวันที่ (ใช้ชื่อ `-latest`)

## 12. คำถามค้าง (เคลียร์ตอน implement)
- Vercel: แยก project ใหม่ หรือรวม `api/` กับ frontend เดิม
- forecast JSON URL: GitHub Pages vs raw — เลือกตอนต่อ pipeline
- Flex layout ละเอียด (สี/bubble) — ปรับช่วง implement
