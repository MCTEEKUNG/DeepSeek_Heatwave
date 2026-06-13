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
import numpy as np
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


SECOND_HALF_MONTHS = [f"{m:02d}" for m in range(8, 13)]   # ส.ค.-ธ.ค. (ขยายเป็นทั้งปี)


def hourly_request(variable: str, year: int, months: list[str] = HOT_MONTHS) -> dict:
    return {
        "product_type": "reanalysis",
        "variable": variable,
        "year": str(year),
        "month": months,
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


RECENT_DIR = RAW.parent / "raw_recent"
RECENT_TMAX_DIR = RECENT_DIR / "tmax_thailand"
RECENT_SOIL_DIR = RECENT_DIR / "soil_moisture_thailand"
ERA5_LATENCY_DAYS = 6      # ERA5 ล่าช้า ~5 วัน -> เผื่อ 6 กันข้อมูลยังไม่ขึ้น
RECENT_WINDOW_DAYS = 70    # ต้อง > 30 (sm_mean30) + เผื่อ lookback/บริบท run


def _recent_out_dir(out_dir: Path) -> Path:
    """แมพโฟลเดอร์เทรน -> โฟลเดอร์ raw_recent ที่ชื่อย่อยเดียวกัน."""
    return RECENT_DIR / out_dir.name


def download_recent(n_days: int = RECENT_WINDOW_DAYS, latency: int = ERA5_LATENCY_DAYS) -> int:
    """ดึง ERA5 'หน้าต่างล่าสุด ~n_days วัน' เข้า raw_recent/ สำหรับ operational predict (cron).

    ไม่ต้องมีข้อมูล 30 ปี — predict ใช้ climatology แช่แข็ง + หน้าต่างนี้พอ.
    ขอเป็น 'เดือนเต็ม' ที่คาบเกี่ยวหน้าต่าง (ราย year) แล้ว aggregate + trim ให้พอดี.
    ไฟล์ตั้งชื่อ era5_*_thailand_recent.nc (เข้า glob เดิม -> build_feature_table อ่านได้).
    """
    import datetime as _dt
    import pandas as _pd

    end = _pd.Timestamp(_dt.date.today()) - _pd.Timedelta(days=latency)
    start = end - _pd.Timedelta(days=n_days)
    window = _pd.date_range(start, end, freq="D")
    by_year: dict[int, list[str]] = {}
    for ts in window:
        by_year.setdefault(ts.year, set()).add(f"{ts.month:02d}")
    by_year = {y: sorted(ms) for y, ms in by_year.items()}

    for p in (RECENT_TMAX_DIR, RECENT_SOIL_DIR, TMP_DIR):
        p.mkdir(parents=True, exist_ok=True)
    client = cdsapi.Client(retry_max=4)
    print(f"=== ดึงข้อมูลล่าสุด {start.date()}..{end.date()} ({n_days} วัน) เข้า raw_recent/ ===")
    print(f"    ปี/เดือนที่ขอ: {by_year}\n")

    failed = []
    for variable, nc_var, how, out_dir, fname_tpl in SPECS:
        recent_dir = _recent_out_dir(out_dir)
        # ใช้ template เดียวกับไฟล์เทรน (year="recent") -> ชื่อเข้า glob ของ loader เป๊ะ
        # เช่น era5_tmax_thailand_recent.nc, era5_sm_l1_thailand_recent.nc
        out_file = recent_dir / fname_tpl.format(year="recent")
        try:
            parts = []
            for year, months in by_year.items():
                tmp = TMP_DIR / f"recent_{nc_var}_{year}.nc"
                client.retrieve(DATASET, hourly_request(variable, year, months), str(tmp))
                parts.append(aggregate_to_daily(tmp, nc_var, how))
                safe_unlink(tmp)
            daily = parts[0] if len(parts) == 1 else xr.concat(parts, dim="valid_time")
            daily = daily.sortby("valid_time")
            # trim ให้พอดีหน้าต่าง [start, end] — slice บน coord ที่เรียงแล้ว (label-based)
            daily = daily.sel(valid_time=slice(np.datetime64(start), np.datetime64(end)))
            daily.to_netcdf(out_file)
            daily.close()
            if is_valid(out_file, nc_var):
                print(f"  [OK] {out_file.name} ({out_file.stat().st_size/1e6:.2f} MB)")
            else:
                print(f"  [FAIL] {nc_var} aggregate ไม่ผ่านด่านตรวจ"); failed.append(nc_var)
        except Exception as exc:
            print(f"  [FAIL] {nc_var}: {repr(exc)[:120]}"); failed.append(nc_var)

    print(f"\n=== สรุป recent === สำเร็จ {len(SPECS)-len(failed)}/{len(SPECS)} | ล้มเหลว {failed}")
    return 1 if failed else 0


def main(years: list[int], months: list[str] = HOT_MONTHS, suffix: str = "") -> int:
    """ดึง + aggregate ราย (ปี, ตัวแปร).

    months/suffix: รองรับการขยายเป็นทั้งปี โดยดึง 'ครึ่งหลัง' (ส.ค.-ธ.ค.) แยกไฟล์
    ลงท้าย _h2 (ไม่ดึง ม.ค.-ก.ค. ที่มีแล้วซ้ำ) ; ชื่อยังเข้า glob เดิม -> downstream
    ต่อ time ให้เองด้วย open_mfdataset(by_coords). suffix="" = พฤติกรรมเดิม (ม.ค.-ก.ค.).
    """
    for p in (TMAX_DIR, SOIL_DIR, TMP_DIR):
        p.mkdir(parents=True, exist_ok=True)
    client = cdsapi.Client(retry_max=4)  # retry ต่ำ -> ถ้าเซิร์ฟเวอร์ล่มจะ fail เร็ว ไม่ค้างยาว

    jobs = [(y, spec) for y in years for spec in SPECS]
    span = f"เดือน {months[0]}-{months[-1]}" + (f" (ไฟล์{suffix})" if suffix else "")
    print(f"=== ดึงชุดหลัก (hourly) + aggregate เอง : {years[0]}-{years[-1]} {span} ===")
    print(f"งานทั้งหมด: {len(jobs)} (={len(years)} ปี x {len(SPECS)} ตัวแปร)\n")

    done, skipped, failed = 0, 0, []
    for i, (year, (variable, nc_var, how, out_dir, fname_tpl)) in enumerate(jobs, 1):
        out_file = out_dir / fname_tpl.format(year=year).replace(".nc", f"{suffix}.nc")
        tag = f"[{i:3d}/{len(jobs)}] {year} {nc_var}{suffix}"

        if is_valid(out_file, nc_var):
            print(f"{tag}  ข้าม (มีไฟล์ valid แล้ว)")
            skipped += 1
            continue

        tmp = TMP_DIR / f"hourly_{nc_var}{suffix}_{year}.nc"
        print(f"{tag}  ดึง hourly ...", flush=True)
        t0 = time.time()
        try:
            client.retrieve(DATASET, hourly_request(variable, year, months), str(tmp))
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
    args = sys.argv[1:]
    # โหมด recent = ดึงหน้าต่างล่าสุดเข้า raw_recent/ (operational/cron) : python script.py recent [n_days]
    if args and args[0] == "recent":
        n = int(args[1]) if len(args) > 1 else RECENT_WINDOW_DAYS
        sys.exit(download_recent(n_days=n))
    # โหมด h2 = ครึ่งหลัง ส.ค.-ธ.ค. (ขยายเป็นทั้งปี) : python script.py h2 [years...]
    if args and args[0] == "h2":
        yrs = [int(a) for a in args[1:]] or list(range(YEAR_START, YEAR_END + 1))
        sys.exit(main(yrs, months=SECOND_HALF_MONTHS, suffix="_h2"))
    yrs = [int(a) for a in args] if args else list(range(YEAR_START, YEAR_END + 1))
    sys.exit(main(yrs))
