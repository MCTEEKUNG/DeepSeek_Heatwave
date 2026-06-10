"""
ดึง "ดัชนี teleconnection" ที่เป็น feature ตัวแปรเปลี่ยนช้า (memory ของระบบ)
- MJO (RMM1, RMM2, phase, amplitude) จาก BoM ออสเตรเลีย : รายวัน
- Niño3.4 (ENSO) anomaly จาก NOAA PSL : รายเดือน
ไฟล์เล็ก สาธารณะ ไม่เกี่ยวกับคิว CDS -> ดึงได้ทันที

แปลงเป็น CSV สะอาด (มี index เป็นวันที่) เก็บที่ data/processed/indices/
การจับคู่ Niño3.4 (รายเดือน) -> รายวัน จะทำตอนประกอบ feature dataset (forward-fill) ไม่ใช่ที่นี่
"""
from __future__ import annotations

import sys
import urllib.request
from pathlib import Path

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

import numpy as np
import pandas as pd

RAW_DIR = Path(__file__).resolve().parent.parent / "data" / "raw" / "indices"
OUT_DIR = Path(__file__).resolve().parent.parent / "data" / "processed" / "indices"

MJO_URL = "http://www.bom.gov.au/climate/mjo/graphics/rmm.74toRealtime.txt"
NINO34_URL = "https://psl.noaa.gov/data/correlation/nina34.anom.data"


def _fetch(url: str, dest: Path) -> list[str]:
    dest.parent.mkdir(parents=True, exist_ok=True)
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    text = urllib.request.urlopen(req, timeout=60).read().decode("utf-8", "replace")
    dest.write_text(text, encoding="utf-8")
    return text.splitlines()


def parse_mjo(lines: list[str]) -> pd.DataFrame:
    """คอลัมน์: year month day RMM1 RMM2 phase amplitude (ข้าม 2 บรรทัดหัว)."""
    rows = []
    for ln in lines[2:]:
        parts = ln.split()
        if len(parts) < 7:
            continue
        try:
            y, m, d = int(parts[0]), int(parts[1]), int(parts[2])
            rmm1, rmm2 = float(parts[3]), float(parts[4])
            phase, amp = int(parts[5]), float(parts[6])
        except ValueError:
            continue
        # ค่า missing = 1.E36 หรือ 999
        if amp > 100 or abs(rmm1) > 100 or abs(rmm2) > 100:
            continue
        rows.append((pd.Timestamp(y, m, d), rmm1, rmm2, phase, amp))
    df = pd.DataFrame(rows, columns=["date", "mjo_rmm1", "mjo_rmm2", "mjo_phase", "mjo_amplitude"])
    return df.set_index("date").sort_index()


def parse_nino34(lines: list[str]) -> pd.DataFrame:
    """บรรทัดแรก = ปีเริ่ม ปีจบ ; จากนั้นแต่ละบรรทัด = ปี + 12 ค่ารายเดือน (missing=-99.99)."""
    rows = []
    for ln in lines[1:]:
        parts = ln.split()
        if len(parts) != 13:
            continue  # ข้ามบรรทัดท้ายไฟล์/คำอธิบาย
        try:
            year = int(parts[0])
            vals = [float(x) for x in parts[1:]]
        except ValueError:
            continue
        for month, v in enumerate(vals, start=1):
            value = np.nan if v <= -99.0 else v
            rows.append((pd.Timestamp(year, month, 1), value))
    df = pd.DataFrame(rows, columns=["date", "nino34_anom"])
    return df.set_index("date").sort_index().dropna()


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    print("=== ดึงดัชนี teleconnection (MJO, Niño3.4) ===\n")

    print("MJO RMM (BoM) ...")
    mjo = parse_mjo(_fetch(MJO_URL, RAW_DIR / "mjo_rmm_raw.txt"))
    mjo.to_csv(OUT_DIR / "mjo_rmm.csv")
    print(f"  -> {len(mjo)} วัน | {mjo.index.min().date()} ถึง {mjo.index.max().date()}")
    print(f"  amplitude range: {mjo['mjo_amplitude'].min():.2f} - {mjo['mjo_amplitude'].max():.2f}")

    print("\nNiño3.4 (NOAA PSL) ...")
    nino = parse_nino34(_fetch(NINO34_URL, RAW_DIR / "nino34_raw.txt"))
    nino.to_csv(OUT_DIR / "nino34.csv")
    print(f"  -> {len(nino)} เดือน | {nino.index.min().date()} ถึง {nino.index.max().date()}")
    print(f"  anomaly range: {nino['nino34_anom'].min():.2f} - {nino['nino34_anom'].max():.2f} °C")

    print(f"\n[OK] เก็บที่ {OUT_DIR}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
