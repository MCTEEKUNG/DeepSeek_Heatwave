# Phase 2: Streamlit Web Demo Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a Streamlit web app that reads pre-computed `outputs/` files and shows interactive forecast skill, ENSO context, feature importance, and calibration charts — ready to demo for TAIEC 2026 portfolio.

**Architecture:** Static demo pattern — app reads `outputs/analysis/bootstrap_ci.csv`, `regime_by_enso.csv`, `permutation_importance.csv`, `calibration_decomp.csv`, and `predictions.csv` at startup. No ERA5 calls, no re-training. Separated into `data_loader.py` (pure data functions, testable) and `charts.py` (pure Plotly figure functions, testable), wired together by `streamlit_app.py`.

**Tech Stack:** Python 3.12, pandas, numpy (existing), streamlit>=1.31, plotly>=5.18, pytest (existing or via pip)

---

## File Map

| Action | Path | Responsibility |
|--------|------|----------------|
| Create | `app/data_loader.py` | Load + label pre-computed CSVs |
| Create | `app/charts.py` | Return Plotly `Figure` objects |
| Create | `app/streamlit_app.py` | 4-tab Streamlit UI |
| Create | `app/tests/__init__.py` | Make tests/ a package |
| Create | `app/tests/test_data_loader.py` | Unit tests for data_loader |
| Create | `app/tests/test_charts.py` | Unit tests for charts |
| Modify | `requirements.txt` | Add streamlit, plotly |

---

## Task 1: Setup — dependencies and directory skeleton

**Files:**
- Modify: `requirements.txt`
- Create: `app/__init__.py` (empty)
- Create: `app/tests/__init__.py` (empty)

- [ ] **Step 1: Add streamlit and plotly to requirements.txt**

Open `requirements.txt` and append these two lines at the end:

```
streamlit>=1.31.0
plotly>=5.18.0
```

- [ ] **Step 2: Install the new dependencies**

Run:
```
pip install "streamlit>=1.31.0" "plotly>=5.18.0"
```

Expected: installs without errors. Verify with:
```
python -c "import streamlit, plotly; print('ok')"
```
Expected output: `ok`

- [ ] **Step 3: Create empty package files**

Create `app/__init__.py` — empty file (0 bytes).
Create `app/tests/__init__.py` — empty file (0 bytes).

- [ ] **Step 4: Commit**

```bash
git add requirements.txt app/__init__.py app/tests/__init__.py
git commit -m "feat: add streamlit + plotly deps; scaffold app/ package"
```

---

## Task 2: data_loader.py — load pre-computed outputs

**Files:**
- Create: `app/data_loader.py`
- Create: `app/tests/test_data_loader.py`

- [ ] **Step 1: Write the failing tests**

Create `app/tests/test_data_loader.py`:

```python
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
    assert set(df["lead"].unique()) == {2, 3, 4, 5, 6}


def test_load_calibration_decomp_has_key_columns():
    df = dl.load_calibration_decomp("y_rm")
    assert isinstance(df, pd.DataFrame)
    assert {"target", "lead", "model", "model_label", "REL", "RES", "brier", "ece"} <= set(df.columns)


def test_load_predictions_columns():
    df = dl.load_predictions("y_rm")
    assert {"y", "p", "lead", "model", "date"} <= set(df.columns)
    assert (df["p"].between(0, 1)).all()
```

- [ ] **Step 2: Run tests to verify they fail**

```
cd C:\Users\ASUS\DeepSeek_Heatwave
python -m pytest app/tests/test_data_loader.py -v
```

Expected: all 6 tests FAIL with `ModuleNotFoundError: No module named 'data_loader'`

- [ ] **Step 3: Implement data_loader.py**

Create `app/data_loader.py`:

```python
"""Load pre-computed outputs/ files into clean DataFrames for the Streamlit demo."""
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
OUTPUTS = ROOT / "outputs"
ANALYSIS = OUTPUTS / "analysis"

MODEL_LABELS = {
    "logistic": "Logistic",
    "logistic_balanced_cal": "Logistic (bal+cal)",
    "lgbm": "LightGBM (raw)",
    "lgbm_cal": "LightGBM (Platt)",
    "lgbm_balanced_cal": "LightGBM (bal+cal)",
    "balanced_rf_cal": "Balanced RF (cal)",
    "climatology": "Climatology",
    "persistence": "Persistence",
}

FEATURE_GROUPS = {
    "sm1": "Soil", "sm1_mean7": "Soil", "sm1_mean30": "Soil", "sm1_trend": "Soil",
    "sm3": "Soil", "sm3_mean7": "Soil", "sm3_mean30": "Soil", "sm3_trend": "Soil",
    "tmax_rm": "Thermal", "tmax_mean7": "Thermal",
    "in_hw_today": "Thermal", "hot_frac7": "Thermal",
    "mjo_rmm1": "MJO", "mjo_rmm2": "MJO", "mjo_amp": "MJO",
    "mjo_sin": "MJO", "mjo_cos": "MJO",
    "nino34_lag1m": "ENSO",
    "doy_sin": "Seasonal", "doy_cos": "Seasonal",
}

GROUP_COLORS = {
    "Soil": "#2ca02c", "ENSO": "#1f77b4", "MJO": "#9467bd",
    "Thermal": "#d62728", "Seasonal": "#ff7f0e",
}


def load_bss_ci(target: str = "y_rm") -> pd.DataFrame:
    """Bootstrap CI table filtered to target + BSS metric, with friendly model labels."""
    df = pd.read_csv(ANALYSIS / "bootstrap_ci.csv")
    df = df[(df["target"] == target) & (df["metric"] == "bss")].copy()
    df["model_label"] = df["model"].map(MODEL_LABELS).fillna(df["model"])
    return df.reset_index(drop=True)


def load_regime_bss(target: str = "y_rm") -> pd.DataFrame:
    """ENSO regime BSS and base rate filtered to target, with friendly model labels."""
    df = pd.read_csv(ANALYSIS / "regime_by_enso.csv")
    df = df[df["target"] == target].copy()
    df["model_label"] = df["model"].map(MODEL_LABELS).fillna(df["model"])
    return df.reset_index(drop=True)


def load_permutation_importance(target: str = "y_rm", model: str = "lgbm") -> pd.DataFrame:
    """Permutation importance with feature group labels and group colors."""
    df = pd.read_csv(ANALYSIS / "permutation_importance.csv")
    df = df[(df["target"] == target) & (df["model"] == model)].copy()
    df["group"] = df["feature"].map(FEATURE_GROUPS).fillna("Other")
    df["color"] = df["group"].map(GROUP_COLORS).fillna("#7f7f7f")
    return df.reset_index(drop=True)


def load_calibration_decomp(target: str = "y_rm") -> pd.DataFrame:
    """Brier score decomposition filtered to target, with friendly model labels."""
    df = pd.read_csv(ANALYSIS / "calibration_decomp.csv")
    df = df[df["target"] == target].copy()
    df["model_label"] = df["model"].map(MODEL_LABELS).fillna(df["model"])
    return df.reset_index(drop=True)


def load_predictions(target: str = "y_rm") -> pd.DataFrame:
    """Raw per-day predictions for reliability curve computation."""
    df = pd.read_csv(OUTPUTS / "predictions.csv", parse_dates=["date"])
    return df[df["target"] == target].copy().reset_index(drop=True)
```

- [ ] **Step 4: Run tests to verify they pass**

```
python -m pytest app/tests/test_data_loader.py -v
```

Expected: all 6 tests PASS.

- [ ] **Step 5: Commit**

```bash
git add app/data_loader.py app/tests/test_data_loader.py
git commit -m "feat: add data_loader — load pre-computed outputs for web demo"
```

---

## Task 3: charts.py — BSS skill chart

**Files:**
- Create: `app/charts.py` (initial version with `fig_bss_by_lead`)
- Modify: `app/tests/test_charts.py` (create the file)

- [ ] **Step 1: Write the failing test for fig_bss_by_lead**

Create `app/tests/test_charts.py`:

```python
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
```

Run to see fail:
```
python -m pytest app/tests/test_charts.py::test_fig_bss_by_lead_returns_figure -v
```
Expected: FAIL with `ModuleNotFoundError: No module named 'charts'`

- [ ] **Step 2: Implement fig_bss_by_lead in charts.py**

Create `app/charts.py`:

```python
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
```

- [ ] **Step 3: Run tests to verify they pass**

```
python -m pytest app/tests/test_charts.py::test_fig_bss_by_lead_returns_figure app/tests/test_charts.py::test_fig_bss_by_lead_empty_selection_returns_figure -v
```

Expected: both PASS.

- [ ] **Step 4: Commit**

```bash
git add app/charts.py app/tests/test_charts.py
git commit -m "feat: charts.py scaffold + fig_bss_by_lead with CI error bars"
```

---

## Task 4: charts.py — ENSO context figures

**Files:**
- Modify: `app/charts.py` (add fig_enso_base_rate, fig_enso_bss)
- Modify: `app/tests/test_charts.py` (add ENSO tests)

- [ ] **Step 1: Write the failing tests**

Append to `app/tests/test_charts.py`:

```python
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
```

Run to see fail:
```
python -m pytest app/tests/test_charts.py::test_fig_enso_base_rate_has_three_regimes -v
```
Expected: FAIL with `AttributeError: module 'charts' has no attribute 'fig_enso_base_rate'`

- [ ] **Step 2: Add fig_enso_base_rate and fig_enso_bss to charts.py**

Append to `app/charts.py` (after the existing `fig_bss_by_lead` function):

```python
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
```

- [ ] **Step 3: Run tests to verify they pass**

```
python -m pytest app/tests/test_charts.py -v
```

Expected: all 4 tests PASS.

- [ ] **Step 4: Commit**

```bash
git add app/charts.py app/tests/test_charts.py
git commit -m "feat: add ENSO base rate + BSS-by-regime charts"
```

---

## Task 5: charts.py — feature importance and reliability

**Files:**
- Modify: `app/charts.py` (add fig_feature_importance, fig_reliability)
- Modify: `app/tests/test_charts.py` (add tests)

- [ ] **Step 1: Write the failing tests**

Append to `app/tests/test_charts.py`:

```python
def test_fig_feature_importance_returns_figure():
    import plotly.graph_objects as go
    df = dl.load_permutation_importance("y_rm", "lgbm")
    fig = charts.fig_feature_importance(df, lead=2)
    assert isinstance(fig, go.Figure)
    assert len(fig.data) == 1  # single horizontal bar trace
    assert len(fig.data[0].y) > 0  # has features


def test_fig_reliability_returns_figure():
    import plotly.graph_objects as go
    pred = dl.load_predictions("y_rm")
    fig = charts.fig_reliability(pred, lead=2, models=["climatology", "logistic"])
    assert isinstance(fig, go.Figure)
    # perfect diagonal + one trace per model = 3 traces
    assert len(fig.data) == 3
```

Run to see fail:
```
python -m pytest app/tests/test_charts.py::test_fig_feature_importance_returns_figure -v
```
Expected: FAIL with `AttributeError`

- [ ] **Step 2: Add fig_feature_importance and fig_reliability to charts.py**

Append to `app/charts.py`:

```python
def fig_feature_importance(df: pd.DataFrame, lead: int) -> go.Figure:
    """Horizontal bar chart of permutation feature importance at a given lead.

    df: from data_loader.load_permutation_importance()
        columns: feature, lead, mean_drop_bss, std_drop_bss, color
    """
    g = df[df["lead"] == lead].sort_values("mean_drop_bss", ascending=True)
    fig = go.Figure(go.Bar(
        x=g["mean_drop_bss"].tolist(),
        y=g["feature"].tolist(),
        orientation="h",
        marker_color=g["color"].tolist(),
        error_x=dict(
            type="data",
            array=g["std_drop_bss"].tolist(),
            visible=True,
        ),
    ))
    fig.update_layout(
        title=f"Feature Importance via Permutation (LightGBM, Lead {lead} weeks)",
        xaxis_title="Mean drop in BSS when feature shuffled (higher = more important)",
        yaxis_title="Feature",
        height=max(380, len(g) * 24),
    )
    return fig


def fig_reliability(pred: pd.DataFrame, lead: int, models: list) -> go.Figure:
    """Reliability diagram for selected models at a given lead time.

    pred: from data_loader.load_predictions() — columns: y, p, lead, model
    Uses evaluate.reliability_curve to bin probabilities into 10 bins.
    """
    from evaluate import reliability_curve  # scripts/evaluate.py

    sub = pred[pred["lead"] == lead]
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=[0, 1], y=[0, 1], mode="lines",
        line=dict(dash="dash", color="black", width=1),
        name="Perfect calibration",
    ))
    _colors = {
        "climatology": "#7f7f7f", "logistic": "#1f77b4",
        "logistic_balanced_cal": "#0099cc", "lgbm_cal": "#2ca02c",
        "lgbm": "#d62728", "persistence": "#bcbd22",
    }
    for model in models:
        g = sub[sub["model"] == model]
        if g.empty:
            continue
        mp, of, ct = reliability_curve(g["y"].to_numpy(), g["p"].to_numpy(), n_bins=10)
        ok = ct > 0
        color = _colors.get(model, "#888888")
        label = MODEL_COLORS.get(model, model)
        fig.add_trace(go.Scatter(
            x=mp[ok].tolist(), y=of[ok].tolist(),
            mode="lines+markers",
            name=model,
            line=dict(color=color, width=2),
            marker=dict(size=8),
        ))
    fig.update_layout(
        title=f"Reliability Diagram (Lead {lead} weeks)",
        xaxis_title="Forecast probability",
        yaxis_title="Observed heatwave frequency",
        xaxis=dict(range=[0, 1]),
        yaxis=dict(range=[0, 1]),
        height=450,
    )
    return fig
```

- [ ] **Step 3: Run all chart tests to verify they pass**

```
python -m pytest app/tests/test_charts.py -v
```

Expected: all 6 tests PASS.

- [ ] **Step 4: Commit**

```bash
git add app/charts.py app/tests/test_charts.py
git commit -m "feat: add feature importance + reliability diagram charts"
```

---

## Task 6: streamlit_app.py — wire all tabs together

**Files:**
- Create: `app/streamlit_app.py`

- [ ] **Step 1: Create streamlit_app.py**

Create `app/streamlit_app.py`:

```python
"""Thailand Sub-Seasonal Heatwave Prediction — Streamlit web demo.

Run from project root:
    streamlit run app/streamlit_app.py
"""
import sys
from pathlib import Path

import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent))
import data_loader as dl
import charts

st.set_page_config(
    page_title="Thailand Heatwave Forecast",
    page_icon="🌡️",
    layout="wide",
)

st.title("🌡️ Thailand Sub-Seasonal Heatwave Prediction")
st.caption(
    "Probabilistic forecast of heatwave events 2–6 weeks ahead · "
    "ERA5 reanalysis 1994–2023 · Classical ML (Logistic Regression / LightGBM)"
)

tab1, tab2, tab3, tab4 = st.tabs(
    ["📈 Forecast Skill", "🌊 ENSO Context", "🔍 Feature Importance", "📐 Calibration"]
)

# ── Tab 1: Forecast Skill ──────────────────────────────────────────────────

with tab1:
    st.header("How well do models beat the baselines?")
    st.markdown(
        "**BSS > 0** = better than seasonal climatology. "
        "**BSS > persistence** = the model adds value beyond simply repeating today's state."
    )

    @st.cache_data
    def _bss_ci():
        return dl.load_bss_ci("y_rm")

    df_bss = _bss_ci()
    available_models = sorted(df_bss["model_label"].unique().tolist())
    selected = st.multiselect(
        "Select models to compare",
        options=available_models,
        default=[m for m in charts.DEFAULT_MODELS if m in available_models],
    )
    if selected:
        st.plotly_chart(charts.fig_bss_by_lead(df_bss, selected), use_container_width=True)
        st.caption(
            "Error bars = 95% moving-block bootstrap CI (block length L=28 days, B=2000 resamples). "
            "CI entirely above 0 (✓) = statistically beats climatology."
        )
        with st.expander("Show data table"):
            cols = ["lead", "model_label", "point", "lo95", "hi95"]
            sub = df_bss[df_bss["model_label"].isin(selected)][cols].rename(
                columns={"point": "BSS", "lo95": "CI lo 95%", "hi95": "CI hi 95%",
                         "model_label": "Model"}
            )
            st.dataframe(sub.round(3).reset_index(drop=True), use_container_width=True)
    else:
        st.info("Select at least one model above.")

# ── Tab 2: ENSO Context ───────────────────────────────────────────────────

with tab2:
    st.header("ENSO strongly modulates heatwave probability")
    st.markdown(
        "During **El Niño**, heatwave base rate reaches ~45%. "
        "During **La Niña**, it drops to ~5%. "
        "Models that capture this shift can be far more useful than climatology alone."
    )

    @st.cache_data
    def _regime():
        return dl.load_regime_bss("y_rm")

    df_reg = _regime()
    col1, col2 = st.columns(2)

    with col1:
        st.plotly_chart(charts.fig_enso_base_rate(df_reg), use_container_width=True)

    with col2:
        available_reg = sorted(df_reg["model_label"].unique().tolist())
        sel_reg = st.multiselect(
            "Models for BSS-by-regime chart",
            options=available_reg,
            default=[m for m in charts.DEFAULT_MODELS if m in available_reg],
            key="enso_models",
        )
        if sel_reg:
            st.plotly_chart(charts.fig_enso_bss(df_reg, sel_reg), use_container_width=True)
            st.caption("Lead 2 weeks shown. Skill is highest during El Niño — the most predictable regime.")

# ── Tab 3: Feature Importance ────────────────────────────────────────────

with tab3:
    st.header("Which features drive heatwave predictability?")
    st.markdown(
        "Each bar shows how much BSS drops when that feature is randomly shuffled. "
        "Larger drop = feature is more critical."
    )

    @st.cache_data
    def _perm():
        return dl.load_permutation_importance("y_rm", "lgbm")

    df_perm = _perm()
    lead_sel = st.select_slider(
        "Lead time (weeks)", options=[2, 3, 4, 5, 6], value=2
    )
    st.plotly_chart(charts.fig_feature_importance(df_perm, lead_sel), use_container_width=True)

    with st.expander("Feature group legend"):
        st.markdown("""
| Color | Group | Features |
|-------|-------|---------|
| 🟢 Green | Soil moisture | sm1, sm3, 7-day/30-day means & trends |
| 🔵 Blue | ENSO | nino34_lag1m |
| 🟣 Purple | MJO | RMM1, RMM2, amplitude, sin/cos phase |
| 🔴 Red | Thermal/State | tmax, hot fraction, in-heatwave indicator |
| 🟠 Orange | Seasonal | doy_sin, doy_cos |
        """)

# ── Tab 4: Calibration ────────────────────────────────────────────────────

with tab4:
    st.header("Are forecast probabilities trustworthy?")
    st.markdown(
        "A well-calibrated model should produce probability p≈0.3 for events that occur "
        "roughly 30% of the time. Points close to the diagonal = good calibration."
    )

    @st.cache_data
    def _pred():
        return dl.load_predictions("y_rm")

    @st.cache_data
    def _decomp():
        return dl.load_calibration_decomp("y_rm")

    pred = _pred()
    df_decomp = _decomp()

    lead_c = st.select_slider(
        "Lead time (weeks)", options=[2, 3, 4, 5, 6], value=2, key="calib_lead"
    )
    rel_models = ["climatology", "logistic", "logistic_balanced_cal", "lgbm_cal"]
    st.plotly_chart(
        charts.fig_reliability(pred, lead_c, rel_models),
        use_container_width=True,
    )

    st.subheader("Brier Score Decomposition")
    st.markdown(
        "**REL** (lower = better): penalty for miscalibration. "
        "**RES** (higher = better): reward for sharpness/discrimination. "
        "**ECE**: expected calibration error (lower = better)."
    )
    sub_decomp = df_decomp[df_decomp["lead"] == lead_c][
        ["model_label", "REL", "RES", "UNC", "brier", "ece"]
    ].rename(columns={
        "model_label": "Model", "brier": "Brier Score", "ece": "ECE"
    }).sort_values("Brier Score")
    st.dataframe(sub_decomp.round(4).reset_index(drop=True), use_container_width=True)
```

- [ ] **Step 2: Run the app to verify it starts**

```
cd C:\Users\ASUS\DeepSeek_Heatwave
streamlit run app/streamlit_app.py
```

Expected: browser opens at http://localhost:8501 showing the app title and 4 tabs. No Python errors in the terminal.

- [ ] **Step 3: Check each tab manually**

In the browser:
- Tab 1: Select "Logistic (bal+cal)" + "Persistence" → line chart with error bars appears ✓
- Tab 2: Base rate bar chart shows El Niño >> La Niña ✓
- Tab 3: Move lead slider from 2 to 6 → chart updates ✓
- Tab 4: Move lead slider → reliability diagram updates ✓

- [ ] **Step 4: Commit**

```bash
git add app/streamlit_app.py
git commit -m "feat: complete Streamlit web demo with 4 interactive tabs"
```

---

## Task 7: Polish and final verification

**Files:**
- Modify: `app/streamlit_app.py` (minor fixes found in Task 6 testing)

- [ ] **Step 1: Run the full test suite to confirm nothing is broken**

```
cd C:\Users\ASUS\DeepSeek_Heatwave
python -m pytest app/tests/ -v
```

Expected: all 8 tests PASS.

- [ ] **Step 2: Check for import error on reliability chart**

The `fig_reliability` function does `from evaluate import reliability_curve`. This import path requires `scripts/` to be on `sys.path`. Verify it works:

```
python -c "
import sys; sys.path.insert(0, 'scripts')
from evaluate import reliability_curve
import numpy as np
mp, of, ct = reliability_curve(np.array([0,0,1,1]), np.array([0.1,0.4,0.6,0.9]))
print('ok', mp.round(2))
"
```

Expected: `ok [0.1  0.4  0.6  0.9]` (or similar binned values)

If the import path fails because `scripts/` is not found, fix the `sys.path.insert` line in `charts.py` to use an absolute path:

```python
# Replace the sys.path.insert line at the top of charts.py with:
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
```

- [ ] **Step 3: Verify the app title and all 4 tabs are working**

Open http://localhost:8501 and confirm:
1. Title shows "🌡️ Thailand Sub-Seasonal Heatwave Prediction"
2. Tab 1 shows BSS chart with error bars and a data table expander
3. Tab 2 shows base rate bar chart + BSS-by-regime chart
4. Tab 3 shows horizontal feature importance chart with legend expander
5. Tab 4 shows reliability diagram + decomposition table

- [ ] **Step 4: Final commit**

```bash
git add -A
git commit -m "feat: Phase 2 complete — Streamlit static demo app for TAIEC portfolio"
```

---

## Self-Review

**Spec coverage check:**
- "แสดงผล" (display results) → ✓ Tabs 1–4 display BSS, ENSO, importance, calibration
- "Phase 2 นอกขอบเขตหลัก" — app reads pre-computed outputs, does NOT re-run training
- Portfolio use (TAIEC July 25) → ✓ local `streamlit run` demo, no cloud deploy needed
- Beginner-friendly — ✓ single command to run, no Docker, no DB

**Placeholder scan:**
- No TBD/TODO left in code steps ✓
- All column names verified against actual CSV files (bootstrap_ci.csv, regime_by_enso.csv, permutation_importance.csv, calibration_decomp.csv) ✓

**Type consistency:**
- `model_label` used consistently between data_loader and charts throughout ✓
- `lead` is integer in all DataFrames; charts use `.tolist()` to avoid numpy int serialisation issues ✓
- `reliability_curve` returns `(mp, of, ct)` — same signature used in `train.py` and in `charts.py` ✓
