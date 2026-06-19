import xarray as xr, numpy as np

files = [
    ("data/raw_backtest/soil_moisture_thailand/era5_sm_l1_thailand_2024.nc", "swvl1"),
    ("data/raw_backtest/soil_moisture_thailand/era5_sm_l3_thailand_2024.nc", "swvl3"),
    ("data/raw_backtest/soil_moisture_thailand/era5_sm_l1_thailand_2025.nc", "swvl1"),
]
for path, var in files:
    ds = xr.open_dataset(path)
    v = ds[var].values.flatten().astype(float)
    finite = v[np.isfinite(v)]
    nan_pct = np.isnan(v).mean() * 100
    print(f"{path.split('/')[-1]}  min={finite.min():.4f}  max={finite.max():.4f}  nan={nan_pct:.1f}%  n_valid={len(finite)}")
    ds.close()
