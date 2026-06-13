"""
คำนวณ "target" คลื่นความร้อน ตามนิยามของงานวิจัย:
  วันร้อน = Tmax เกินเปอร์เซ็นไทล์ที่ q (90/95) ของภูมิอากาศฐาน
            โดยเกณฑ์แปรตามวันในปี (moving window ±W วัน รวมทุกปีในช่วงฐาน)
  คลื่นความร้อน = วันร้อน "ติดต่อกัน >= 3 วัน"

ทุกอุณหภูมิผ่านด่านหน่วย (units_utils) -> °C ก่อนคำนวณเสมอ
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import xarray as xr

sys.path.insert(0, str(Path(__file__).resolve().parent))
from units_utils import convert_temperature_to_celsius, assert_temperature_celsius_plausible


def load_tmax_celsius(paths: list[Path]) -> xr.DataArray:
    """เปิดไฟล์ Tmax หลายปี ต่อกันตามเวลา และแปลงเป็น °C (ผ่านด่านหน่วย)."""
    ds = xr.open_mfdataset([str(p) for p in paths], combine="by_coords")
    t = ds["t2m"]
    t = convert_temperature_to_celsius(t.load())
    assert_temperature_celsius_plausible(t)
    # ใช้ valid_time เป็นแกนเวลา ตั้งชื่อให้เป็น 'time' เพื่อความสะดวก
    if "valid_time" in t.dims:
        t = t.rename({"valid_time": "time"})
    if "number" in t.coords:
        t = t.drop_vars("number")
    return t.sortby("time")


def doy_window_percentile(da: xr.DataArray, q: float, window: int = 15) -> xr.DataArray:
    """เกณฑ์เปอร์เซ็นไทล์ที่ q (0-100) ต่อ 'วันในปี' (1..366) ต่อ grid cell.

    สำหรับวันในปี d: รวมค่าทุกวันที่อยู่ในช่วง ±window วัน (วนรอบปฏิทิน) ของทุกปี
    แล้วหาเปอร์เซ็นไทล์ -> ได้เกณฑ์ที่ลื่นไหลตามฤดูกาล
    คืนค่า DataArray มิติ (dayofyear, lat, lon)
    """
    doy = da["time"].dt.dayofyear
    spatial_dims = [d for d in da.dims if d != "time"]
    thresholds = []
    for d in range(1, 367):
        offsets = {((d - 1 + k) % 366) + 1 for k in range(-window, window + 1)}
        sample = da.where(doy.isin(list(offsets)), drop=True)
        if sample.sizes.get("time", 0) == 0:
            # ไม่มีข้อมูลในหน้าต่างของ doy นี้เลย (เช่น ส.ค.-ธ.ค. ที่ไม่ได้โหลด)
            # -> เกณฑ์ = NaN (วันเหล่านี้ไม่มีข้อมูลให้เทียบอยู่แล้ว)
            thr = xr.full_like(da.isel(time=0, drop=True), np.nan)
        else:
            thr = sample.quantile(q / 100.0, dim="time").drop_vars("quantile")
        thresholds.append(thr)
    out = xr.concat(thresholds, dim="dayofyear")
    out = out.assign_coords(dayofyear=np.arange(1, 367))
    out.attrs["units"] = "degC"
    out.attrs["percentile"] = q
    out.attrs["window_days"] = window
    return out.transpose("dayofyear", *spatial_dims)


def hot_days(da: xr.DataArray, threshold_by_doy: xr.DataArray) -> xr.DataArray:
    """boolean: Tmax ของวันนั้น > เกณฑ์ของวันในปีนั้น (per cell)."""
    doy = da["time"].dt.dayofyear
    thr_for_each_day = threshold_by_doy.sel(dayofyear=doy)
    return (da > thr_for_each_day).rename("hot_day")


def flag_heatwaves(hot_bool: xr.DataArray, min_len: int = 3) -> xr.DataArray:
    """flag วันที่อยู่ในช่วงวันร้อนติดต่อกัน >= min_len (ตามแกน time, per cell).

    วิธี: นับความยาว run ที่แต่ละวันสังกัด = (นับไปข้างหน้า)+(นับถอยหลัง)-ตัวเอง
    แล้ว heatwave = run length >= min_len
    """
    b = hot_bool.transpose("time", ...).astype(int)
    arr = b.values  # (time, *spatial)
    n = arr.shape[0]

    fwd = np.zeros_like(arr)
    fwd[0] = arr[0]
    for i in range(1, n):
        fwd[i] = np.where(arr[i] > 0, fwd[i - 1] + 1, 0)

    bwd = np.zeros_like(arr)
    bwd[-1] = arr[-1]
    for i in range(n - 2, -1, -1):
        bwd[i] = np.where(arr[i] > 0, bwd[i + 1] + 1, 0)

    run_len = fwd + bwd - arr  # ความยาว run ที่แต่ละ "วันร้อน" สังกัด
    hw = (run_len >= min_len)
    return xr.DataArray(hw, coords=hot_bool.transpose("time", ...).coords,
                        dims=hot_bool.transpose("time", ...).dims, name="heatwave")


def trailing_run_length(hot_bool) -> np.ndarray:
    """ความยาว run ของ "วันร้อน" ที่ต่อเนื่องและ **จบ ณ ตำแหน่งนั้น** (มองเฉพาะ index <= t).

    รับ 1D (array/Series ของ 0/1/NaN) คืน np.ndarray ความยาวเท่ากัน.
    ค่า != 1 (รวม 0 และ NaN) = ตัด streak (รีเซ็ตเป็น 0).
    leak-free โดยนิยาม: ไม่มองอนาคต — ต่างจาก flag_heatwaves ที่นับ fwd+bwd (ใช้ทำ label).
    """
    arr = np.asarray(hot_bool, dtype=float)
    out = np.zeros(arr.shape[0], dtype=float)
    run = 0.0
    for i in range(arr.shape[0]):
        run = run + 1.0 if arr[i] == 1.0 else 0.0
        out[i] = run
    return out


if __name__ == "__main__":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

    # --- unit test ตรรกะนับวันติดต่อกัน (ไม่ต้องใช้ข้อมูลจริง) ---
    seq = np.array([0, 1, 1, 0, 1, 1, 1, 0, 1, 0, 1, 1, 1, 1])  # runs: 2, 3, 1, 4
    expected = np.array([0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 1, 1, 1, 1])
    da = xr.DataArray(seq, dims=["time"], coords={"time": np.arange(len(seq))})
    got = flag_heatwaves(da, min_len=3).astype(int).values
    print("ทดสอบตรรกะ >=3 วันติดกัน:")
    print("  input   :", seq.tolist())
    print("  expected:", expected.tolist())
    print("  got     :", got.tolist())
    assert (got == expected).all(), "ตรรกะนับวันติดต่อกันผิด!"
    print("  [OK] ตรรกะถูกต้อง\n")

    # --- unit test: trailing run length (มองย้อนหลังเท่านั้น = leak-free) ---
    seq2 = np.array([0, 1, 1, 0, 1, 1, 1, 0, 1, 0, 1, 1, 1, 1])
    exp_run = np.array([0, 1, 2, 0, 1, 2, 3, 0, 1, 0, 1, 2, 3, 4])
    got_run = trailing_run_length(seq2)
    assert (got_run == exp_run).all(), got_run.tolist()
    # in_hw (trailing >= 3): ติด 1 ตั้งแต่ "วันที่ 3" ของ streak เป็นต้นไป (ไม่ย้อนติดให้ 2 วันแรก)
    assert ((exp_run >= 3).astype(int).tolist()
            == [0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 1])
    print("  [OK] trailing_run_length + in_hw(trailing>=3) ถูกต้อง\n")

    # --- ถ้ามีไฟล์ปีจริงแล้ว สาธิต climatology + heatwave จริง ---
    ddir = Path(__file__).resolve().parent.parent / "data" / "raw" / "tmax_thailand"
    files = sorted(ddir.glob("era5_tmax_thailand_*.nc"))
    print(f"พบไฟล์ปีจริง {len(files)} ไฟล์")
    if len(files) >= 1:
        t = load_tmax_celsius(files)
        print("  ช่วงเวลา:", str(t.time.min().values)[:10], "->", str(t.time.max().values)[:10],
              "| n_days =", t.sizes["time"])
        thr90 = doy_window_percentile(t, q=90, window=15)
        hot = hot_days(t, thr90)
        hw = flag_heatwaves(hot, min_len=3)
        frac_hot = float(hot.mean())
        frac_hw = float(hw.mean())
        print(f"  สัดส่วนวันร้อน (>p90)      : {frac_hot:.3f}")
        print(f"  สัดส่วนวันที่เป็นคลื่นความร้อน: {frac_hw:.3f}")
        print("  [OK] คำนวณ target จากข้อมูลจริงได้")
    else:
        print("  (ยังไม่มีไฟล์ปีจริง - รอดาวน์โหลด แล้วรันสคริปต์นี้ซ้ำเพื่อทดสอบเต็ม)")
