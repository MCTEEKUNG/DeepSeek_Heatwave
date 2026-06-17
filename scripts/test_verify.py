"""Verify correctness of streak-crossing-window-boundary label logic.

All tests are purely synthetic — no ERA5 files needed.
Run with:  python scripts/test_verify.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

sys.path.insert(0, str(Path(__file__).resolve().parent))
from heatwave_target import flag_heatwaves
from build_dataset import weekly_event_targets, LEADS


def _make_hot_da(dates: pd.DatetimeIndex, hot_indices: list[int]) -> xr.DataArray:
    """Return a 1-D (time,) DataArray of 0/1 hot-day flags."""
    arr = np.zeros(len(dates), dtype=int)
    for i in hot_indices:
        arr[i] = 1
    return xr.DataArray(arr, dims=["time"], coords={"time": dates})


def test1_streak_straddles_boundary() -> None:
    """3-day hot run at indices 19/20/21 straddles window-2 and window-3.

    Proves that flag_heatwaves detects the run across the full series AND that
    weekly_event_targets propagates it into both windows.
    """
    dates = pd.date_range("2020-01-15", periods=60, freq="D")
    hot_da = _make_hot_da(dates, hot_indices=[19, 20, 21])

    hw = flag_heatwaves(hot_da, min_len=3)
    hw_vals = hw.astype(int).values

    # flag_heatwaves must mark exactly positions 19, 20, 21 as heatwave
    expected_hw = np.zeros(60, dtype=int)
    expected_hw[19] = 1
    expected_hw[20] = 1
    expected_hw[21] = 1
    assert (hw_vals == expected_hw).all(), (
        f"Test1: flag_heatwaves wrong.\n  got={hw_vals.tolist()}"
    )

    hw_series = pd.Series(hw.astype(float).values, index=dates)
    targets = weekly_event_targets(hw_series, leads=LEADS)

    # issue_date = dates[0]
    # window-2: days 14..20 relative to dates[0]  (indices 14..20) — hits idx 19, 20
    # window-3: days 21..27 relative to dates[0]  (indices 21..27) — hits idx 21
    l2 = targets.loc[dates[0], "lead2"]
    l3 = targets.loc[dates[0], "lead3"]
    assert l2 == 1.0, f"Test1: lead2 should be 1, got {l2}"
    assert l3 == 1.0, f"Test1: lead3 should be 1, got {l3}"
    print("[OK] Test1: 3-day run straddling window-2/window-3 boundary → lead2=1, lead3=1")


def test2_two_day_run_no_flag() -> None:
    """A 2-day run must NOT be flagged as heatwave (min_len=3 not met)."""
    dates = pd.date_range("2020-01-15", periods=60, freq="D")
    hot_da = _make_hot_da(dates, hot_indices=[14, 15])

    hw = flag_heatwaves(hot_da, min_len=3)
    hw_vals = hw.astype(int).values

    assert hw_vals.sum() == 0, (
        f"Test2: 2-day run should produce all zeros, got {hw_vals.tolist()}"
    )
    print("[OK] Test2: 2-day hot run → flag_heatwaves produces all zeros (min_len=3 not met)")


def test3_streak_fully_inside_window2() -> None:
    """3-day run at indices 16/17/18 falls entirely inside window-2 (days 14..20).

    lead2 = 1 ; all other leads at dates[0] = 0 (series is 60 days so all windows
    are complete — window-6 ends at index 0+42+6=48 < 60).
    """
    dates = pd.date_range("2020-01-15", periods=60, freq="D")
    hot_da = _make_hot_da(dates, hot_indices=[16, 17, 18])

    hw = flag_heatwaves(hot_da, min_len=3)
    hw_vals = hw.astype(int).values

    # Confirm flag_heatwaves marks exactly 16/17/18
    expected_hw = np.zeros(60, dtype=int)
    expected_hw[16] = 1
    expected_hw[17] = 1
    expected_hw[18] = 1
    assert (hw_vals == expected_hw).all(), (
        f"Test3: flag_heatwaves wrong.\n  got={hw_vals.tolist()}"
    )

    hw_series = pd.Series(hw.astype(float).values, index=dates)
    targets = weekly_event_targets(hw_series, leads=LEADS)

    l2 = targets.loc[dates[0], "lead2"]
    assert l2 == 1.0, f"Test3: lead2 should be 1, got {l2}"

    for L in [3, 4, 5, 6]:
        v = targets.loc[dates[0], f"lead{L}"]
        assert v == 0.0, (
            f"Test3: lead{L} should be 0 (streak not in that window), got {v}"
        )
    print("[OK] Test3: 3-day run fully inside window-2 → lead2=1, leads 3-6=0")


def test4_flag_heatwaves_sequence() -> None:
    """Canonical sequence: runs of 2, 3, 1, 4 → only runs >=3 are flagged."""
    seq = np.array([0, 1, 1, 0, 1, 1, 1, 0, 1, 0, 1, 1, 1, 1])
    expected = np.array([0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 1, 1, 1, 1])
    da = xr.DataArray(seq, dims=["time"], coords={"time": np.arange(len(seq))})
    got = flag_heatwaves(da, min_len=3).astype(int).values
    assert (got == expected).all(), (
        f"Test4: sequence wrong.\n  got     : {got.tolist()}\n  expected: {expected.tolist()}"
    )
    print("[OK] Test4: canonical run sequence [2,3,1,4] → flags only runs >=3")


if __name__ == "__main__":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

    test1_streak_straddles_boundary()
    test2_two_day_run_no_flag()
    test3_streak_fully_inside_window2()
    test4_flag_heatwaves_sequence()

    print("\nALL TESTS PASSED")
