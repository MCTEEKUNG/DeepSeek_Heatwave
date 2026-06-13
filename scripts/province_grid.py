"""Map 77 Thai provinces -> nearest 0.25° ERA5 grid cell + extract per-province series.

centroid จาก data/provinces.csv (พอร์ตจาก Heatwave_AI) -> cell ใกล้สุดในกริด ERA5
(lat 5-21 / lon 97-106). 77 จังหวัด -> 76 unique cell (กทม./ปริมณฑล 1 คู่ใช้ cell ร่วม).
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

ROOT = Path(__file__).resolve().parent.parent
PROVINCES_CSV = ROOT / "data" / "provinces.csv"
REGIONS = ["Central", "North", "Northeast", "East", "West", "South"]


def load_provinces(path: Path | None = None) -> pd.DataFrame:
    df = pd.read_csv(path or PROVINCES_CSV)
    need = {"id", "code", "name_th", "name_en", "region", "lat", "lon"}
    missing = need - set(df.columns)
    if missing:
        raise ValueError(f"provinces.csv ขาดคอลัมน์: {missing}")
    return df


def _lat_lon_names(da: xr.DataArray) -> tuple[str, str]:
    """ชื่อ dim ละติจูด/ลองจิจูด (รองรับ latitude/lat, longitude/lon) — fail loudly ถ้าไม่เจอ."""
    for latn in ("latitude", "lat"):
        if latn in da.dims:
            break
    else:
        raise KeyError(f"หา dim ละติจูดไม่เจอใน {list(da.dims)}")
    for lonn in ("longitude", "lon"):
        if lonn in da.dims:
            break
    else:
        raise KeyError(f"หา dim ลองจิจูดไม่เจอใน {list(da.dims)}")
    return latn, lonn


def nearest_cell(da: xr.DataArray, lat: float, lon: float) -> tuple[float, float]:
    """พิกัด (lat, lon) ของ cell ที่ใกล้ centroid ที่สุดในกริดของ da."""
    latn, lonn = _lat_lon_names(da)
    la = np.asarray(da[latn].values, dtype=float)
    lo = np.asarray(da[lonn].values, dtype=float)
    return float(la[np.abs(la - lat).argmin()]), float(lo[np.abs(lo - lon).argmin()])


def province_series(da: xr.DataArray, lat: float, lon: float) -> pd.Series:
    """อนุกรมรายวันของ cell ใกล้ centroid -> pandas Series (index=date normalize)."""
    latn, lonn = _lat_lon_names(da)
    cell = da.sel({latn: lat, lonn: lon}, method="nearest")
    idx = pd.DatetimeIndex(cell["time"].values).normalize()
    return pd.Series(np.asarray(cell.values, dtype=float), index=idx)


if __name__ == "__main__":
    import sys
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

    # 1) โหลด provinces ครบ 77 + region อยู่ในชุดที่รู้จัก
    pv = load_provinces()
    assert len(pv) == 77, len(pv)
    assert set(pv["region"]) <= set(REGIONS), set(pv["region"]) - set(REGIONS)
    print(f"[OK] provinces.csv: {len(pv)} จังหวัด, regions={sorted(set(pv['region']))}")

    # 2) nearest_cell บนกริดสังเคราะห์ที่ตรงกริดจริง (lat 5..21, lon 97..106, 0.25°)
    lat = np.arange(5.0, 21.0001, 0.25)[::-1]
    lon = np.arange(97.0, 106.0001, 0.25)
    da = xr.DataArray(np.zeros((1, len(lat), len(lon))), dims=["time", "latitude", "longitude"],
                      coords={"time": [pd.Timestamp("2020-01-01")], "latitude": lat, "longitude": lon})
    cy, cx = nearest_cell(da, 13.7563, 100.5018)  # กรุงเทพฯ
    assert abs(cy - 13.75) < 1e-9 and abs(cx - 100.50) < 1e-9, (cy, cx)
    # 77 จังหวัด -> 76 cell unique (1 คู่ใช้ร่วม)
    cells = {nearest_cell(da, r.lat, r.lon) for r in pv.itertuples()}
    assert len(cells) == 76, len(cells)
    print(f"[OK] nearest_cell: BKK->({cy},{cx}); 77 จังหวัด -> {len(cells)} cell unique")

    # 3) province_series คืน Series รายวัน index=date ของ cell
    s = province_series(da.isel(time=[0]).assign_coords(time=[pd.Timestamp("2020-01-01")]), 13.7563, 100.5018)
    assert isinstance(s, pd.Series) and len(s) == 1
    print("[OK] province_series คืน pandas Series")
    print("[OK] self-test ผ่าน")
