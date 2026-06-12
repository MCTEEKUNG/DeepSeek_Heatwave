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
