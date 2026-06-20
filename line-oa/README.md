# LINE OA — แจ้งเตือนความเสี่ยงคลื่นความร้อนรายสัปดาห์

Stateless LINE bot: webhook (Vercel) ตอบ on-demand + GitHub Actions broadcast รายสัปดาห์
อ่าน `docs/forecast_provinces.json` ไม่มีฐานข้อมูล/ไม่เก็บ user id

## โครงสร้าง
- `bot/` — โมดูล pure + I/O (config, risk, messages, forecast_source, line_api, handlers, richmenu)
- `api/webhook.py` — Vercel entrypoint (`/api/webhook`)
- `scripts/broadcast_weekly.py` — งานรายสัปดาห์ (เรียกโดย .github/workflows/forecast.yml)

## รัน test
```
cd line-oa
pip install -r requirements.txt pytest
python -m pytest tests/ -v
```

## Setup ครั้งแรก
1. LINE Developers Console → สร้าง Provider + Messaging API channel
2. เก็บ **Channel secret** (Basic settings) + **Issue Channel access token** (Messaging API)
3. ปิด Auto-reply/Greeting, เปิด Use webhook = ON
4. Deploy Vercel:
   - New Project → import repo → **Root Directory = `line-oa`**
   - Env: `LINE_CHANNEL_SECRET`, `LINE_CHANNEL_ACCESS_TOKEN`,
     `WEBSITE_URL=https://heat-map-frontend.vercel.app/map`
     (`FORECAST_URL` ปล่อย default ได้)
   - Deploy → ได้ URL → ตั้ง **Webhook URL = `https://<project>.vercel.app/api/webhook`**
   - กด **Verify** ในคอนโซล LINE ต้องได้ Success
5. ตั้ง GitHub Actions secret: `LINE_CHANNEL_ACCESS_TOKEN`
6. Rich menu (ครั้งเดียว): เตรียมรูป 2500×843 PNG 3 ช่อง แล้ว
   ```
   cd line-oa
   LINE_CHANNEL_ACCESS_TOKEN=xxx python -m bot.richmenu /path/to/menu.png
   ```

## ทดสอบ broadcast (manual)
```
cd line-oa
LINE_CHANNEL_ACCESS_TOKEN=xxx python scripts/broadcast_weekly.py --file ../docs/forecast_provinces.json
```
(ระวัง: broadcast ส่งหา **ผู้ติดตามทุกคน** — ทดสอบตอนยังมีผู้ติดตามน้อย/บัญชีทดสอบ)
