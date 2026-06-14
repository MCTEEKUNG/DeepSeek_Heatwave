# RUNBOOK — ออก forecast ปัจจุบันด้วยมือ (on-demand)

## A. Forecast จากข้อมูลเทรนเต็ม (ไม่ต้องมี CDS key)
ออกพยากรณ์จาก "วันล่าสุดที่ feature ครบ" ในข้อมูลเทรน (ใช้ตรวจ/เดโม):
```
python scripts/predict.py
```
ผล -> `docs/forecast.json`

## B. Forecast "ปัจจุบันจริง" (ต้องมี CDS key ที่ ~/.cdsapirc)
1. ดึงข้อมูล ERA5 ล่าสุด ~70 วัน เข้า data/raw_recent/ (โหมด recent มีอยู่แล้ว):
   ```
   python scripts/download_era5_hourly_aggregate.py recent
   ```
2. รีเฟรช indices ล่าสุด (MJO/Niño):
   ```
   python scripts/download_indices.py
   ```
3. ออกพยากรณ์แบบ operational (climatology แช่แข็ง + raw_recent/):
   ```
   python scripts/predict.py operational
   ```
ผล -> `docs/forecast.json` (issue date = วันล่าสุดที่ feature ครบ ; MJO ที่ค้างถูก impute + ใส่ warning)

## หมายเหตุ
- ERA5 ล่าช้า ~5-6 วัน (ตั้งไว้ใน ERA5_LATENCY_DAYS) -> issue date จะตามหลังวันนี้เล็กน้อย
- ถ้า retrain โมเดล/เปลี่ยนนิยาม feature: รัน build_dataset.py -> train_final.py -> predict.py ตามลำดับ
- ทุกโมดูลมี self-test: `python scripts/<module>.py test`

## C. Pipeline bridge (forecast รายจังหวัด -> frontend)
ส่ง `docs/forecast_provinces.json` ไปยัง frontend แบบ predict -> validate (hard gate) -> distribute
```
python scripts/publish_bridge.py            # predict + validate + sync -> HeatMAP_Frontend/public/ (dev)
python scripts/publish_bridge.py --no-predict   # ใช้ไฟล์เดิม: validate + sync (ไม่ predict ใหม่)
python scripts/publish_bridge.py --publish      # + sync เข้า heatwave-contract แล้ว git push -> GitHub Pages (prod)
python scripts/validate_contract.py             # ตรวจ contract อย่างเดียว (exit 1 ถ้าไม่ผ่าน)
```
- validate เป็น **hard gate**: ถ้า contract ผิด จะ abort ไม่ distribute
- override path ปลายทาง: env `BRIDGE_FRONTEND_DIR`, `BRIDGE_CONTRACT_DIR`
- prod contract เผยแพร่ที่ `https://mcteekung.github.io/heatwave-contract/forecast_provinces.json` (schema_version 1)
