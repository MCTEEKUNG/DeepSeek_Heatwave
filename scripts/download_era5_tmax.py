"""
ดึง ERA5 daily-maximum 2m temperature (Tmax) เฉพาะกรอบประเทศไทย ย้อนหลัง 30 ปี
กลยุทธ์กัน limit + ทนทาน:
  - แบ่งดึง "ทีละปี" (1 request/ปี ~2 MB) ไม่ใช่ก้อนใหญ่ก้อนเดียว
  - resume ได้: ปีไหนมีไฟล์ valid อยู่แล้ว จะข้าม
  - ตรวจไฟล์ทุกปีด้วยด่านหน่วย (units_utils) หลังดาวน์โหลด

ใช้งาน:
  python download_era5_tmax.py                                                    # 1994-2023 (default)
  python download_era5_tmax.py --year-start 2024 --year-end 2025 --out-dir data/raw_backtest/tmax_thailand/
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

sys.path.insert(0, str(Path(__file__).resolve().parent))
from units_utils import convert_temperature_to_celsius, assert_temperature_celsius_plausible

THAILAND_AREA = [21, 97, 5, 106]  # N, W, S, E
YEAR_START = 1994
YEAR_END = 2023
DATASET = "derived-era5-single-levels-daily-statistics"

OUT_DIR = Path(__file__).resolve().parent.parent / "data" / "raw" / "tmax_thailand"
ALL_DAYS = [f"{d:02d}" for d in range(1, 32)]
# รอบแรก: เฉพาะฤดูร้อน ม.ค.-ก.ค. (เดือน 01-07) เพื่อลดเวลาดาวน์โหลด ~ครึ่ง
# ครอบคลุมคลื่นความร้อนไทย (ก.พ.-มิ.ย.) + buffer ให้ feature ล่วงหน้า ; ขยายเป็นทั้งปีภายหลังได้
ALL_MONTHS = [f"{m:02d}" for m in range(1, 8)]


def year_request(year: int) -> dict:
    return {
        "product_type": "reanalysis",
        "variable": ["2m_temperature"],
        "year": str(year),
        "month": ALL_MONTHS,
        "day": ALL_DAYS,  # CDS จะข้ามวันที่ไม่มีจริง (เช่น 30 ก.พ.) ให้เอง
        "daily_statistic": "daily_maximum",
        "time_zone": "utc+00:00",
        "frequency": "1_hourly",
        "area": THAILAND_AREA,
    }


def is_valid(path: Path) -> bool:
    """เช็คว่าไฟล์เปิดได้และค่าอุณหภูมิสมเหตุสมผล (ผ่านด่านหน่วย)."""
    if not path.exists() or path.stat().st_size < 1000:
        return False
    try:
        with xr.open_dataset(path) as d:
            t = convert_temperature_to_celsius(d["t2m"])
            assert_temperature_celsius_plausible(t)
        return True
    except Exception:
        return False


def main(year_start: int = YEAR_START, year_end: int = YEAR_END, out_dir: Path = OUT_DIR) -> int:
    out_dir = Path(out_dir)
    if year_start > year_end:
        raise ValueError(f"year_start ({year_start}) must be <= year_end ({year_end})")
    out_dir.mkdir(parents=True, exist_ok=True)
    client = cdsapi.Client()
    years = list(range(year_start, year_end + 1))

    print(f"=== ดึง Tmax เฉพาะไทย {year_start}-{year_end} ({len(years)} ปี) ===")
    print(f"ปลายทาง: {out_dir}\n")

    done, skipped, failed = [], [], []
    for i, year in enumerate(years, 1):
        out_file = out_dir / f"era5_tmax_thailand_{year}.nc"
        tag = f"[{i:2d}/{len(years)}] {year}"

        if is_valid(out_file):
            print(f"{tag}  ข้าม (มีไฟล์ valid แล้ว)")
            skipped.append(year)
            continue

        print(f"{tag}  กำลังดึง ...", flush=True)
        t0 = time.time()
        try:
            client.retrieve(DATASET, year_request(year), str(out_file))
        except Exception as exc:
            print(f"{tag}  [FAIL] {exc!r}")
            failed.append(year)
            continue

        if is_valid(out_file):
            mb = out_file.stat().st_size / 1e6
            print(f"{tag}  [OK] {mb:.2f} MB ({time.time()-t0:.0f}s)")
            done.append(year)
        else:
            print(f"{tag}  [FAIL] ไฟล์ดาวน์โหลดมาแต่ไม่ผ่านด่านตรวจ")
            failed.append(year)

    print("\n=== สรุป ===")
    print(f"ดึงใหม่สำเร็จ : {len(done)} ปี")
    print(f"ข้าม (มีอยู่) : {len(skipped)} ปี")
    print(f"ล้มเหลว      : {len(failed)} ปี -> {failed}")
    return 1 if failed else 0


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="ดึง ERA5 daily Tmax สำหรับประเทศไทย (ทีละปี, resume ได้)"
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
