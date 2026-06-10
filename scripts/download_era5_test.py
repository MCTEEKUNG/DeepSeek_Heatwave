"""
de-risk test: ดึง ERA5 ก้อนจิ๋วเพื่อยืนยันว่า CDS API ใช้งานได้
- ชุดข้อมูล: derived-era5-single-levels-daily-statistics (ค่าสูงสุดรายวันมาตรงๆ -> เล็ก)
- ขอบเขต: กรอบประเทศไทย, เมษายน 2023 (เดือนร้อนสุด), ตัวแปรเดียว = 2m temperature (Tmax)
ออกแบบให้ "เล็กที่สุด" เพื่อไม่ชน request limit ของ CDS
"""

import sys
from pathlib import Path

# บังคับ stdout เป็น UTF-8 เพื่อให้พิมพ์ภาษาไทยบน Windows console ได้
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

import cdsapi

# กรอบประเทศไทยโดยประมาณ: [North, West, South, East]
THAILAND_AREA = [21, 97, 5, 106]

OUT_DIR = Path(__file__).resolve().parent.parent / "data" / "raw"
OUT_FILE = OUT_DIR / "era5_tmax_thailand_2023-04_TEST.nc"


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    request = {
        "product_type": "reanalysis",
        "variable": ["2m_temperature"],
        "year": "2023",
        "month": ["04"],
        "day": [f"{d:02d}" for d in range(1, 31)],  # เมษายนมี 30 วัน
        "daily_statistic": "daily_maximum",
        "time_zone": "utc+00:00",
        "frequency": "1_hourly",
        "area": THAILAND_AREA,
    }

    print("=== ERA5 de-risk test download ===")
    print(f"dataset : derived-era5-single-levels-daily-statistics")
    print(f"area    : {THAILAND_AREA} (N,W,S,E)")
    print(f"period  : 2023-04 (daily maximum 2m temperature)")
    print(f"output  : {OUT_FILE}")
    print("กำลังส่ง request ... (อาจรอคิวสักครู่)")

    client = cdsapi.Client()
    try:
        client.retrieve(
            "derived-era5-single-levels-daily-statistics",
            request,
            str(OUT_FILE),
        )
    except Exception as exc:  # จับ error เพื่อแปลเป็นภาษาที่เข้าใจง่าย
        print("\n[ERROR] ดึงข้อมูลไม่สำเร็จ:", repr(exc))
        msg = str(exc).lower()
        if "licence" in msg or "license" in msg or "403" in msg:
            print(
                "\n>>> สาเหตุที่พบบ่อย: ยังไม่ได้กดยอมรับเงื่อนไข (licence) ของชุดข้อมูลนี้บนเว็บ CDS\n"
                "    วิธีแก้: ล็อกอิน cds.climate.copernicus.eu เปิดหน้า dataset นี้ แล้วกดยอมรับ Terms ที่แท็บ Download"
            )
        return 1

    size_mb = OUT_FILE.stat().st_size / 1e6
    print(f"\n[OK] ดาวน์โหลดสำเร็จ! ขนาดไฟล์ = {size_mb:.2f} MB")
    print("ขั้นต่อไป: เปิดไฟล์ด้วย xarray เพื่อตรวจโครงสร้างข้อมูล")
    return 0


if __name__ == "__main__":
    sys.exit(main())
