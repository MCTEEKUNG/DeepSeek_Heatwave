"""Return Plotly Figure objects from cleaned DataFrames (no Streamlit imports here)."""
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import plotly.graph_objects as go

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

MODEL_COLORS = {
    "Logistic": "#1f77b4",
    "Logistic (bal+cal)": "#0099cc",
    "LightGBM (raw)": "#d62728",
    "LightGBM (Platt)": "#2ca02c",
    "LightGBM (bal+cal)": "#98df8a",
    "Balanced RF (cal)": "#e377c2",
    "Climatology": "#7f7f7f",
    "Persistence": "#bcbd22",
}

DEFAULT_MODELS = [
    "Logistic (bal+cal)", "LightGBM (Platt)", "Climatology", "Persistence",
]


def fig_bss_by_lead(df: pd.DataFrame, selected_models: list) -> go.Figure:
    """BSS vs lead time with 95% CI error bars.

    df: from data_loader.load_bss_ci() — columns: lead, model_label, point, lo95, hi95
    """
    fig = go.Figure()
    for label in selected_models:
        g = df[df["model_label"] == label].sort_values("lead")
        if g.empty:
            continue
        color = MODEL_COLORS.get(label, "#888888")
        fig.add_trace(go.Scatter(
            x=g["lead"].tolist(),
            y=g["point"].tolist(),
            error_y=dict(
                type="data",
                symmetric=False,
                array=(g["hi95"] - g["point"]).tolist(),
                arrayminus=(g["point"] - g["lo95"]).tolist(),
                visible=True,
            ),
            mode="lines+markers",
            name=label,
            line=dict(color=color, width=2),
            marker=dict(size=8),
        ))
    fig.add_hline(y=0, line_dash="dash", line_color="black", line_width=1)
    fig.update_layout(
        title="Brier Skill Score vs Lead Time (95% CI)",
        xaxis_title="Lead Time (weeks)",
        yaxis_title="BSS (higher = better; 0 = climatology baseline)",
        xaxis=dict(tickvals=[2, 3, 4, 5, 6]),
        legend=dict(orientation="h", y=-0.25),
        height=450,
        margin=dict(b=80),
    )
    return fig


_REGIME_COLORS = {"elnino": "#d62728", "neutral": "#ff7f0e", "lanina": "#1f77b4"}
_REGIME_LABELS = {"elnino": "El Niño", "neutral": "Neutral", "lanina": "La Niña"}


def fig_enso_base_rate(df: pd.DataFrame) -> go.Figure:
    """Grouped bar chart: heatwave base rate by ENSO regime across lead times.

    df: from data_loader.load_regime_bss() — uses climatology rows for base rate
    """
    clim = df[df["model"] == "climatology"].drop_duplicates(["lead", "regime"])
    fig = go.Figure()
    for regime in ["elnino", "neutral", "lanina"]:
        g = clim[clim["regime"] == regime].sort_values("lead")
        fig.add_trace(go.Bar(
            x=g["lead"].tolist(),
            y=g["base_rate"].tolist(),
            name=_REGIME_LABELS[regime],
            marker_color=_REGIME_COLORS[regime],
        ))
    fig.update_layout(
        title="Heatwave Base Rate by ENSO Regime",
        xaxis_title="Lead Time (weeks)",
        yaxis_title="Observed heatwave frequency",
        barmode="group",
        xaxis=dict(tickvals=[2, 3, 4, 5, 6]),
        legend=dict(orientation="h", y=-0.25),
        height=400,
        margin=dict(b=80),
    )
    return fig


def fig_enso_bss(df: pd.DataFrame, selected_models: list) -> go.Figure:
    """Grouped bar chart: BSS by ENSO regime at lead 2, comparing selected models.

    df: from data_loader.load_regime_bss()
    """
    sub = df[(df["lead"] == 2) & df["model_label"].isin(selected_models)]
    fig = go.Figure()
    for regime in ["elnino", "neutral", "lanina"]:
        g = sub[sub["regime"] == regime]
        fig.add_trace(go.Bar(
            x=g["model_label"].tolist(),
            y=g["bss"].tolist(),
            name=_REGIME_LABELS[regime],
            marker_color=_REGIME_COLORS[regime],
        ))
    fig.add_hline(y=0, line_dash="dash", line_color="black", line_width=1)
    fig.update_layout(
        title="BSS by ENSO Regime (Lead 2 weeks)",
        xaxis_title="Model",
        yaxis_title="BSS (vs climatology)",
        barmode="group",
        legend=dict(orientation="h", y=-0.25),
        height=400,
        margin=dict(b=80),
    )
    return fig
