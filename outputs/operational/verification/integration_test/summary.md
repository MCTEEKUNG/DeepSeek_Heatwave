# Operational Verification Scorecard

## Dataset
- Pairs: 114,730
- Issue dates: 326 (2022-01-30 to 2023-07-11)
- Provinces: 77
- Leads: [np.int64(2), np.int64(3), np.int64(4), np.int64(5), np.int64(6)]

## BSS by Lead (vs per-province climatology base_rate)

| Lead | BSS | 95% CI (lo) | 95% CI (hi) | n |
|------|-----|-------------|-------------|---|
| 2 | +0.096 | +0.048 | +0.159 | 25102 |
| 3 | +0.165 | +0.103 | +0.255 | 24024 |
| 4 | +0.050 | +0.006 | +0.103 | 22946 |
| 5 | +0.121 | +0.073 | +0.184 | 21868 |
| 6 | +0.019 | -0.018 | +0.065 | 20790 |

**Overall BSS: +0.092 [+0.064, +0.124]**

## Caveats
- CIs use moving-block bootstrap (B=2000, block=28 issue-dates ≈ 28×77 rows).
- Data covers January–July only (Thai hot season + buffer). Skill estimates apply to this period.
- Model trained through 2023; these scores are on 2024–2025 (genuinely out-of-sample).
- BSS compared to per-province per-lead climatology base_rate (frozen 1994-2023 baseline).
- Positive BSS = model beats climatology; CI not crossing 0 = statistically significant.