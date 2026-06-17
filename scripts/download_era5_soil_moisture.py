"""
ดึง ERA5 soil moisture (volumetric soil water) เฉพาะกรอบไทย ย้อนหลัง 30 ปี (ม.ค.-ก.ค.)
- feature หลัก (land-atmosphere coupling) ตัวทำนาย sub-seasonal ที่แข็งแรงสุด
- ดึง "ทีละชั้น (1 ตัวแปร/คำขอ)" เพราะ 4 ชั้นพร้อมกันชน cost limit ของ CDS
- รอบแรกใช้ 2 ชั้น: layer 1 (0-7cm, ผิว/coupling เร็ว) + layer 3 (28-100cm, memory ราก)
  เพิ่ม layer 2/4 ภายหลังได้แค่แก้ SOIL_LAYERS
- daily_mean, หน่วย m^3/m^3 (0-1), แบ่งทีละปี+ชั้น, resume ได้

ใช้งาน:
  python download_era5_soil_moisture.py                                             # ดึงครบ 1994-2023
  python download_era5_soil_moisture.py --year-start 2024 --year-end 2025          # backtest 2024-2025
  python download_era5_soil_moisture.py --out-dir data/raw_backtest/soil_moisture/  # กำหนด output เอง
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

import cdsapi
import xarray as xr

THAILAND_AREA = [21, 97, 5, 106]  # N, W, S, E
YEAR_START, YEAR_END = 1994, 2023
DATASET = "derived-era5-single-levels-daily-statistics"

# รอบแรก: 2 ชั้น (เพิ่ม 2,4 ภายหลังได้)
SOIL_LAYERS = [1, 3]
VAR_NAME = {n: f"volumetric_soil_water_layer_{n}" for n in (1, 2, 3, 4)}
NC_NAME = {n: f"swvl{n}" for n in (1, 2, 3, 4)}

SM_MIN, SM_MAX = 0.0, 1.0  # ช่วงค่าความชื้นดินเชิงปริมาตรที่สมเหตุสมผล

OUT_DIR = Path(__file__).resolve().parent.parent / "data" / "raw" / "soil_moisture_thailand"
ALL_DAYS = [f"{d:02d}" for d in range(1, 32)]
HOT_MONTHS = [f"{m:02d}" for m in range(1, 8)]  # ม.ค.-ก.ค. ให้ตรงกับ Tmax


def layer_request(year: int, layer: int) -> dict:
    return {
        "product_type": "reanalysis",
        "variable": [VAR_NAME[layer]],  # 1 ตัวแปร/คำขอ = ขนาดเท่า Tmax (พิสูจน์แล้วว่าผ่าน)
        "year": str(year),
        "month": HOT_MONTHS,
        "day": ALL_DAYS,
        "daily_statistic": "daily_mean",
        "time_zone": "utc+00:00",
        "frequency": "1_hourly",
        "area": THAILAND_AREA,
    }


def is_valid(path: Path, layer: int) -> bool:
    if not path.exists() or path.stat().st_size < 1000:
        return False
    try:
        with xr.open_dataset(path) as d:
            v = NC_NAME[layer]
            if v not in d:
                return False
            vmin, vmax = float(d[v].min()), float(d[v].max())
            if vmin < SM_MIN or vmax > SM_MAX:
                return False
        return True
    except Exception:
        return False


def main(year_start: int = YEAR_START, year_end: int = YEAR_END, out_dir: Path = OUT_DIR) -> int:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    client = cdsapi.Client()
    years = list(range(year_start, year_end + 1))
    jobs = [(y, n) for y in years for n in SOIL_LAYERS]

    print(f"=== ดึง Soil Moisture ชั้น {SOIL_LAYERS} เฉพาะไทย {year_start}-{year_end} (ม.ค.-ก.ค.) ===")
    print(f"จำนวนคำขอ: {len(jobs)} (={len(years)} ปี x {len(SOIL_LAYERS)} ชั้น)")
    print(f"ปลายทาง: {out_dir}\n")

    done, skipped, failed = [], [], []
    for i, (year, layer) in enumerate(jobs, 1):
        out_file = out_dir / f"era5_sm_l{layer}_thailand_{year}.nc"
        tag = f"[{i:3d}/{len(jobs)}] {year} L{layer}"

        if is_valid(out_file, layer):
            print(f"{tag}  ข้าม (มีไฟล์ valid แล้ว)")
            skipped.append((year, layer))
            continue

        print(f"{tag}  กำลังดึง ...", flush=True)
        t0 = time.time()
        try:
            client.retrieve(DATASET, layer_request(year, layer), str(out_file))
        except Exception as exc:
            print(f"{tag}  [FAIL] {exc!r}")
            failed.append((year, layer))
            continue

        if is_valid(out_file, layer):
            mb = out_file.stat().st_size / 1e6
            print(f"{tag}  [OK] {mb:.2f} MB ({time.time()-t0:.0f}s)")
            done.append((year, layer))
        else:
            print(f"{tag}  [FAIL] ดาวน์โหลดมาแต่ไม่ผ่านด่านตรวจ")
            failed.append((year, layer))

    print("\n=== สรุป ===")
    print(f"ดึงใหม่สำเร็จ : {len(done)}")
    print(f"ข้าม (มีอยู่) : {len(skipped)}")
    print(f"ล้มเหลว      : {len(failed)} -> {failed}")
    return 1 if failed else 0


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="ดึง ERA5 daily Soil Moisture สำหรับประเทศไทย (ทีละปี+ชั้น, resume ได้)"
    )
    parser.add_argument(
        "--year-start", type=int, default=YEAR_START,
        help=f"ปีเริ่มต้น (default: {YEAR_START})",
    )
    parser.add_argument(
        "--year-end", type=int, default=YEAR_END,
        help=f"ปีสิ้นสุด (default: {YEAR_END})",
    )
    parser.add_argument(
        "--out-dir", type=Path, default=OUT_DIR,
        help=f"โฟลเดอร์ปลายทาง (default: {OUT_DIR})",
    )
    args = parser.parse_args()
    sys.exit(main(args.year_start, args.year_end, args.out_dir))
