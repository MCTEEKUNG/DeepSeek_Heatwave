"""
ดึง ERA5 จากชุด "หลัก" reanalysis-era5-single-levels (รายชั่วโมง) แล้ว aggregate เป็น daily เองในเครื่อง
เหตุผล: ชุดสำเร็จรูป derived-...-daily-statistics ล่ม (HTTP 500) ฝั่งเซิร์ฟเวอร์ CDS

ได้ไฟล์ daily หน้าตาเหมือนเดิม (var t2m / swvlN, dim valid_time) -> downstream + ไฟล์เดิมใช้ต่อได้
  - Tmax  : daily maximum ของ 2m_temperature  -> tmax_thailand/era5_tmax_thailand_{year}.nc
  - Soil  : daily mean ของ volumetric_soil_water_layer_{1,3} -> soil_moisture_thailand/era5_sm_l{N}_thailand_{year}.nc

กลยุทธ์: ต่อปี ดึง hourly -> aggregate -> ตรวจ -> ลบ hourly ทิ้ง ; resume ได้ ; retry ต่ำ (กันค้างยาว)

ใช้งาน:
  python download_era5_hourly_aggregate.py            # ครบ 1994-2023
  python download_era5_hourly_aggregate.py 2001       # เฉพาะปีที่ระบุ (ทดสอบ)
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

THAILAND_AREA = [21, 97, 5, 106]
YEAR_START, YEAR_END = 1994, 2023
DATASET = "reanalysis-era5-single-levels"
HOT_MONTHS = [f"{m:02d}" for m in range(1, 8)]
ALL_DAYS = [f"{d:02d}" for d in range(1, 32)]
ALL_HOURS = [f"{h:02d}:00" for h in range(24)]

RAW = Path(__file__).resolve().parent.parent / "data" / "raw"
TMAX_DIR = RAW / "tmax_thailand"
SOIL_DIR = RAW / "soil_moisture_thailand"
TMP_DIR = RAW / "_hourly_tmp"
SOIL_LAYERS = [1, 3]

# สเปกของแต่ละ "ผลผลิต": (era5_variable, nc_var, วิธี aggregate, โฟลเดอร์, รูปแบบชื่อไฟล์)
SPECS = [
    ("2m_temperature", "t2m", "max", TMAX_DIR, "era5_tmax_thailand_{year}.nc"),
]
for _n in SOIL_LAYERS:
    SPECS.append(
        (f"volumetric_soil_water_layer_{_n}", f"swvl{_n}", "mean", SOIL_DIR,
         f"era5_sm_l{_n}_thailand_{{year}}.nc")
    )


def safe_unlink(path: Path) -> None:
    """ลบไฟล์แบบไม่ทำให้โปรแกรมล้มถ้าลบไม่ได้ (Windows อาจปล่อย handle ช้า)."""
    try:
        path.unlink(missing_ok=True)
    except Exception:
        pass


def hourly_request(variable: str, year: int) -> dict:
    return {
        "product_type": "reanalysis",
        "variable": variable,
        "year": str(year),
        "month": HOT_MONTHS,
        "day": ALL_DAYS,
        "time": ALL_HOURS,
        "data_format": "netcdf",
        "area": THAILAND_AREA,
    }


def is_valid(path: Path, nc_var: str) -> bool:
    if not path.exists() or path.stat().st_size < 1000:
        return False
    try:
        with xr.open_dataset(path) as d:
            if nc_var not in d:
                return False
            if nc_var == "t2m":
                t = convert_temperature_to_celsius(d[nc_var])
                assert_temperature_celsius_plausible(t)
            else:  # soil moisture m^3/m^3 (ยอมรับค่าติดลบจิ๋วใกล้ศูนย์ที่ ERA5 มีปกติ)
                v = d[nc_var]
                if float(v.min()) < -0.05 or float(v.max()) > 1.0:
                    return False
        return True
    except Exception:
        return False


def aggregate_to_daily(hourly_path: Path, nc_var: str, how: str) -> xr.Dataset:
    """เปิด hourly -> resample เป็น daily (max/mean) -> คืน Dataset dim 'valid_time' (ให้ตรงไฟล์เดิม).

    ใช้ context manager + .load() เพื่อ "อ่านเข้าหน่วยความจำแล้วปิดไฟล์" ก่อนลบ hourly
    (กัน PermissionError บน Windows ที่ไฟล์ยังถูกจับค้าง)
    """
    with xr.open_dataset(hourly_path) as d:
        tdim = "valid_time" if "valid_time" in d.dims else "time"
        da = d[nc_var]
        if tdim != "time":
            da = da.rename({tdim: "time"})
        daily = da.resample(time="1D").max() if how == "max" else da.resample(time="1D").mean()
        daily = daily.rename({"time": "valid_time"}).load()  # บังคับ compute เข้า RAM
        daily.attrs = dict(d[nc_var].attrs)  # คงหน่วยไว้ (เช่น K)
    return daily.to_dataset(name=nc_var)


def main(years: list[int]) -> int:
    for p in (TMAX_DIR, SOIL_DIR, TMP_DIR):
        p.mkdir(parents=True, exist_ok=True)
    client = cdsapi.Client(retry_max=4)  # retry ต่ำ -> ถ้าเซิร์ฟเวอร์ล่มจะ fail เร็ว ไม่ค้างยาว

    jobs = [(y, spec) for y in years for spec in SPECS]
    print(f"=== ดึงชุดหลัก (hourly) + aggregate เอง : {years[0]}-{years[-1]} ม.ค.-ก.ค. ===")
    print(f"งานทั้งหมด: {len(jobs)} (={len(years)} ปี x {len(SPECS)} ตัวแปร)\n")

    done, skipped, failed = 0, 0, []
    for i, (year, (variable, nc_var, how, out_dir, fname_tpl)) in enumerate(jobs, 1):
        out_file = out_dir / fname_tpl.format(year=year)
        tag = f"[{i:3d}/{len(jobs)}] {year} {nc_var}"

        if is_valid(out_file, nc_var):
            print(f"{tag}  ข้าม (มีไฟล์ valid แล้ว)")
            skipped += 1
            continue

        tmp = TMP_DIR / f"hourly_{nc_var}_{year}.nc"
        print(f"{tag}  ดึง hourly ...", flush=True)
        t0 = time.time()
        try:
            client.retrieve(DATASET, hourly_request(variable, year), str(tmp))
            daily = aggregate_to_daily(tmp, nc_var, how)
            daily.to_netcdf(out_file)
            daily.close()
            safe_unlink(tmp)  # ลบ hourly ทิ้งประหยัดดิสก์
        except Exception as exc:
            print(f"{tag}  [FAIL] {repr(exc)[:120]}")
            safe_unlink(tmp)
            failed.append((year, nc_var))
            continue

        if is_valid(out_file, nc_var):
            mb = out_file.stat().st_size / 1e6
            print(f"{tag}  [OK] daily {mb:.2f} MB ({time.time()-t0:.0f}s)")
            done += 1
        else:
            print(f"{tag}  [FAIL] aggregate แล้วไม่ผ่านด่านตรวจ")
            failed.append((year, nc_var))

    print(f"\n=== สรุป === ดึงใหม่ {done} | ข้าม {skipped} | ล้มเหลว {len(failed)} -> {failed}")
    return 1 if failed else 0


if __name__ == "__main__":
    yrs = [int(a) for a in sys.argv[1:]] if len(sys.argv) > 1 else list(range(YEAR_START, YEAR_END + 1))
    sys.exit(main(yrs))
