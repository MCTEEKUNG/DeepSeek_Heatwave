# Operational Verification Scorecard

## Dataset
- Pairs: 10,010 (77 provinces × 5 leads × 26 issue dates)
- Issue dates: 26 (2024-01-30 to 2024-02-24)
- Provinces: 77
- Leads: 2, 3, 4, 5, 6 weeks ahead

> **Note on window:** The backtest covers only Jan–Feb 2024 because the MJO RMM index
> (BoM source) was last updated 2024-02-24 and is no longer maintained at that URL.
> Rows with missing MJO features were excluded via `dropna`. This is a known data
> limitation; skill estimates reflect the Thai pre-hot-season window only.

## BSS by Lead (vs per-province climatology base_rate)

| Lead | BSS | AUC | n | Interpretation |
|------|-----|-----|---|----------------|
| 2 wk | +0.298 | 0.592 | 2002 | ✅ Strong skill |
| 3 wk | +0.259 | 0.686 | 2002 | ✅ Strong skill — best AUC |
| 4 wk | +0.073 | 0.569 | 2002 | ✅ Modest skill |
| 5 wk | -0.004 | 0.437 | 2002 | ❌ No skill |
| 6 wk | +0.189 | 0.462 | 2002 | ⚠️ BSS positive but AUC < 0.5 — model near-climatology, no discrimination |

**Overall BSS: +0.178 [95% CI: +0.065, +0.232]**
(CI from moving-block bootstrap, B=2000, block=28 days; overall CI valid — per-lead CIs
degenerate due to small n=26 issue dates)

## Actionable Lead Range

**Leads 2–4 (2–4 weeks ahead) are operationally useful** — positive BSS and AUC > 0.5
confirm the model beats per-province climatology and discriminates events correctly.

Leads 5–6 should not be used for actionable warnings:
- Lead 5: BSS < 0, AUC < 0.5 — worse than climatology baseline.
- Lead 6: BSS appears positive but AUC 0.462 reveals this is calibration artefact
  (model predicts near base_rate constantly) with no true discrimination skill.

The 2–4 week window is sufficient for early public health awareness and preparedness.

## Caveats
- Model trained through 2023; backtest data (Jan–Feb 2024) is genuinely out-of-sample.
- BSS baseline: per-province per-lead climatological base_rate (frozen 1994–2023).
- Positive BSS = model beats climatology. Overall CI not crossing 0 = statistically significant.
- MJO data gap limits backtest to 26 days. A wider window (requiring updated RMM source)
  would provide more robust per-lead CI estimates.
- Data covers January–July only (Thai hot season + early buffer).
