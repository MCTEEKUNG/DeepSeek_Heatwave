import sys
from pathlib import Path
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import data_loader as dl


def test_load_bss_ci_returns_dataframe():
    df = dl.load_bss_ci("y_rm")
    assert isinstance(df, pd.DataFrame)
    assert not df.empty
    assert {"lead", "model", "model_label", "point", "lo95", "hi95"} <= set(df.columns)
    assert set(df["lead"].unique()) == {2, 3, 4, 5, 6}


def test_load_bss_ci_metric_is_bss_only():
    df = dl.load_bss_ci("y_rm")
    # function must filter to BSS only (no auc/brier rows)
    assert "metric" not in df.columns or (df["metric"] == "bss").all()


def test_load_regime_bss_has_three_regimes():
    df = dl.load_regime_bss("y_rm")
    assert isinstance(df, pd.DataFrame)
    assert {"elnino", "neutral", "lanina"} <= set(df["regime"].unique())
    assert "model_label" in df.columns


def test_load_permutation_importance_has_groups():
    df = dl.load_permutation_importance("y_rm", "lgbm")
    assert isinstance(df, pd.DataFrame)
    assert {"feature", "lead", "group", "color", "mean_drop_bss"} <= set(df.columns)
    # permutation_importance.csv only has lead=2 (script default)
    assert 2 in df["lead"].unique()
    assert set(df["lead"].unique()) <= {2, 3, 4, 5, 6}


def test_load_calibration_decomp_has_key_columns():
    df = dl.load_calibration_decomp("y_rm")
    assert isinstance(df, pd.DataFrame)
    assert {"target", "lead", "model", "model_label", "REL", "RES", "brier", "ece"} <= set(df.columns)


def test_load_predictions_columns():
    df = dl.load_predictions("y_rm")
    assert {"y", "p", "lead", "model", "date"} <= set(df.columns)
    assert (df["p"].between(0, 1)).all()
