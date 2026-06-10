"""
ยูทิลิตี้จัดการ "หน่วย" ให้เป็นมาตรฐานเดียว เพื่อกันบั๊ก Kelvin/Celsius ปนกัน
หลักการ:
  1) แปลงหน่วยตาม metadata (attribute `units`) ไม่ใช่เดาจากค่า
  2) ด่านตรวจช่วงค่าที่ fail loudly (ถ้าลืมแปลง -> error ทันที ไม่ปล่อยค่าเพี้ยน)

หน่วยมาตรฐานภายในโปรเจกต์ = องศาเซลเซียส (°C)
"""
from __future__ import annotations

import xarray as xr

# ชื่อหน่วยที่ยอมรับ (ทำให้เป็นตัวพิมพ์เล็กก่อนเทียบ)
_KELVIN_NAMES = {"k", "kelvin"}
_CELSIUS_NAMES = {"c", "degc", "celsius", "deg_c", "°c"}

# ช่วงอุณหภูมิพื้นผิวที่ "เป็นไปได้จริง" บนโลก (°C) — ใช้เป็นด่านตรวจ
PLAUSIBLE_C_MIN = -90.0
PLAUSIBLE_C_MAX = 65.0


def convert_temperature_to_celsius(da: xr.DataArray) -> xr.DataArray:
    """แปลง DataArray อุณหภูมิเป็น °C โดยอ่านจาก attribute `units`.

    - K / kelvin     -> ลบ 273.15
    - C / degC / ...  -> ปล่อยไว้ (เป็น °C อยู่แล้ว)
    - ไม่มี units หรือไม่รู้จัก -> error (ไม่เดา)
    """
    units = str(da.attrs.get("units", "")).strip().lower()

    if units in _KELVIN_NAMES:
        out = da - 273.15
    elif units in _CELSIUS_NAMES:
        out = da.copy()
    elif units == "":
        raise ValueError(
            f"ตัวแปร '{da.name}' ไม่มี attribute 'units' -> ปฏิเสธการเดาหน่วย "
            "(โปรดระบุหน่วยให้ชัดเจนก่อนนำเข้าโมเดล)"
        )
    else:
        raise ValueError(
            f"ตัวแปร '{da.name}' มีหน่วยที่ไม่รู้จัก: '{units}' "
            "(รองรับเฉพาะ Kelvin/Celsius)"
        )

    out.attrs = dict(da.attrs)
    out.attrs["units"] = "degC"
    return out


def assert_temperature_celsius_plausible(
    da: xr.DataArray,
    lo: float = PLAUSIBLE_C_MIN,
    hi: float = PLAUSIBLE_C_MAX,
) -> None:
    """ด่านตรวจ: ถ้าค่าหลุดช่วง °C ที่สมเหตุสมผล -> error ดังๆ.

    จับบั๊กยอดฮิต: ลืมแปลง Kelvin (ค่า ~250-320) จะทะลุ hi ทันที.
    """
    vmin = float(da.min())
    vmax = float(da.max())
    if vmin < lo or vmax > hi:
        raise ValueError(
            f"ตัวแปร '{da.name}' มีค่านอกช่วงอุณหภูมิที่สมเหตุสมผล "
            f"[{lo}, {hi}] °C : พบ min={vmin:.2f}, max={vmax:.2f}. "
            "สงสัยว่าลืมแปลงหน่วย (เช่น ยังเป็น Kelvin) หรือข้อมูลผิดปกติ"
        )


if __name__ == "__main__":
    import sys
    from pathlib import Path

    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

    test_file = Path(__file__).resolve().parent.parent / "data" / "raw" / "era5_tmax_thailand_2023-04_TEST.nc"
    d = xr.open_dataset(test_file)
    t2m = d["t2m"]

    print("ก่อนแปลง: units =", t2m.attrs.get("units"),
          "| min/max =", round(float(t2m.min()), 2), "/", round(float(t2m.max()), 2))

    t2m_c = convert_temperature_to_celsius(t2m)
    print("หลังแปลง: units =", t2m_c.attrs.get("units"),
          "| min/max =", round(float(t2m_c.min()), 2), "/", round(float(t2m_c.max()), 2))

    assert_temperature_celsius_plausible(t2m_c)
    print("[OK] ผ่านด่านตรวจช่วงค่า °C")

    # สาธิตว่า "ถ้าลืมแปลง" ด่านตรวจจะดักได้จริง (ส่งค่า Kelvin ดิบเข้าไป)
    try:
        assert_temperature_celsius_plausible(t2m)  # ยังเป็น Kelvin
        print("[BUG] ด่านตรวจควร error แต่ไม่ error!")
    except ValueError as e:
        print("[OK] ด่านตรวจดักค่า Kelvin ที่ลืมแปลงได้:", str(e)[:70], "...")
