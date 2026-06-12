import sys
from pathlib import Path
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import data_loader as dl
import charts


def test_fig_bss_by_lead_returns_figure():
    import plotly.graph_objects as go
    df = dl.load_bss_ci("y_rm")
    fig = charts.fig_bss_by_lead(df, ["Logistic (bal+cal)", "Climatology"])
    assert isinstance(fig, go.Figure)
    assert len(fig.data) == 2  # one trace per selected model


def test_fig_bss_by_lead_empty_selection_returns_figure():
    import plotly.graph_objects as go
    df = dl.load_bss_ci("y_rm")
    fig = charts.fig_bss_by_lead(df, [])
    assert isinstance(fig, go.Figure)
    assert len(fig.data) == 0


def test_fig_enso_base_rate_has_three_regimes():
    import plotly.graph_objects as go
    df = dl.load_regime_bss("y_rm")
    fig = charts.fig_enso_base_rate(df)
    assert isinstance(fig, go.Figure)
    # three bar traces: El Niño, Neutral, La Niña
    assert len(fig.data) == 3


def test_fig_enso_bss_returns_figure():
    import plotly.graph_objects as go
    df = dl.load_regime_bss("y_rm")
    fig = charts.fig_enso_bss(df, ["Logistic (bal+cal)", "Climatology"])
    assert isinstance(fig, go.Figure)
    assert len(fig.data) == 3  # 3 regime groups
