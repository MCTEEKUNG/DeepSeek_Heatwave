# Data Contract: `forecast_provinces.json`

**Version:** schema_version = 1
**Date:** 2026-06-17
**Status:** Authoritative. This document is the single source of truth for the data contract between the Python backend and the frontend. The machine-enforced gate is `scripts/validate_contract.py`; this document describes and explains that gate. If this doc and the validator ever disagree, the validator wins — fix this doc.

`forecast_provinces.json` is a static JSON artifact produced by a batch job (`scripts/predict_provinces.py` → `build_forecast()`) and served to the frontend (e.g. via GitHub Pages). The frontend only reads the file; it never recomputes the forecast.

---

## What this product is (and is NOT)

- The model produces a **sub-seasonal heatwave PROBABILITY forecast** for each of Thailand's **77 provinces**.
- It is a **WEEKLY outlook**: exactly **5 weekly buckets**, at lead **2, 3, 4, 5, and 6 weeks ahead**. It is **NOT** a daily forecast and **NOT** a "next 7 days" forecast.
- The model predicts **PROBABILITY OF OCCURRENCE only**. There is **NO** severity, intensity, temperature, duration, or heat-index output (e.g. no sWBGT, no max-temp, no "number of hot days"). **Any such field a client displays has no backing in this contract** and must not be invented or inferred from these fields.
- Risk is **RELATIVE to local climatology**, not an absolute probability band (see Risk labels below).

---

## Top-level object

| Field | Type | Notes |
|---|---|---|
| `schema_version` | int | Must equal `1`. |
| `model` | string | Non-empty model name/identifier. |
| `generated_at` | string | ISO-8601 datetime (UTC). When this artifact was built/published. |
| `n_provinces` | int | Must equal `77` and must equal `len(provinces)`. |
| `provinces` | array | Exactly 77 province objects (see below). |

> An optional `skill` array may also appear (per-province skill metrics). It is **not** part of the validated contract and frontends should not depend on it.

---

## Province object

Required keys (enforced order in validator: `id, code, name_th, name_en, region, lat, lon, issue_date, forecasts`):

| Field | Type | Notes |
|---|---|---|
| `id` | int | Unique province id (no duplicates across the array). |
| `code` | string | Province code. |
| `name_th` | string | Thai province name. |
| `name_en` | string | English province name. |
| `region` | string | Region name. |
| `lat` | number | Latitude, within `[-90, 90]`. |
| `lon` | number | Longitude, within `[-180, 180]`. |
| `issue_date` | string | `YYYY-MM-DD`. The date of the latest feature row used (the forecast issue/base date). |
| `forecasts` | array | Exactly the 5 lead buckets below. |

---

## Forecast object (one per lead week)

Each province has exactly **5** forecast objects, with `lead_weeks` ∈ {2, 3, 4, 5, 6} (the full set, no duplicates, no extras).

| Field | Type | Notes |
|---|---|---|
| `lead_weeks` | int | One of `2, 3, 4, 5, 6`. |
| `probability` | number | Calibrated probability of heatwave occurrence, in `[0, 1]`. |
| `climatology_base_rate` | number | Local climatological base rate for this lead, in `[0, 1]`. The "normal" against which probability is judged. |
| `ratio_vs_normal` | number | `probability / climatology_base_rate`, `>= 0`. Drives the risk label. |
| `risk_level_th` | string | Thai risk label, non-empty. Render **verbatim**. |
| `risk_level_en` | string | English risk label. One of `Low`, `Normal`, `Elevated`, `High` (**stable contract keys**). |

---

## Risk labels — RELATIVE to local climatology

Risk is derived from **`ratio = probability / climatology_base_rate`**, NOT from absolute probability thresholds. The same probability can be `Normal` in a hot province and `Elevated` in a cool one. A very low absolute probability is always floored to `Low`.

| Condition | `risk_level_en` | `risk_level_th` |
|---|---|---|
| `prob < 0.05` OR `ratio < 0.75` | `Low` | ต่ำ |
| `ratio < 1.5` | `Normal` | ปกติ |
| `ratio < 2.5` | `Elevated` | ค่อนข้างสูง |
| `ratio >= 2.5` | `High` | สูง |

Notes:
- **English keys (`Low`/`Normal`/`Elevated`/`High`) are STABLE contract keys.** They are validated by `ALLOWED_RISK_EN` in `scripts/validate_contract.py` and mapped by the frontend (`RISK_EN_TO_APP`). Do not rename, translate, or reorder them.
- The Thai strings are **display text** and may be revised for wording; the frontend must render `risk_level_th` exactly as delivered.
- **The frontend MUST render `risk_level_th` / `risk_level_en` VERBATIM and MUST NOT re-derive risk from its own probability thresholds.** Re-deriving causes drift and over-escalation versus the backend.

---

## Semantics a client must get right

- **Weekly, lead 2–6.** 5 buckets only; not daily, not a rolling 7-day window.
- **Probability only.** No severity/intensity/temperature/duration/heat-index. Do not surface or compute such fields.
- **Forecast target week.** For lead `L`, the targeted week is `issue_date + L*7 days`. (E.g. issue_date 2026-06-01, lead 3 → week of 2026-06-22.)
- **Freshness = `generated_at − issue_date`** (the data lag). Roughly **6–16 days is normal**, driven by ERA5 input latency. This is **NOT** `today − issue_date`; do not treat the forecast as "stale" just because today is well past `issue_date`.
- **Render labels verbatim** (repeated for emphasis): use `risk_level_th` / `risk_level_en` as given; never recompute risk on the client.

---

## Fields frontends often ignore (but should use)

These are present and validated, yet commonly dropped by clients:

- **`climatology_base_rate`** — needed to communicate that risk is relative to local normal (e.g. "2.7× the seasonal norm"), not an absolute number.
- **`risk_level_th`** — the Thai label; render it directly for Thai-language UI instead of re-mapping from English.

---

## Abbreviated example (one province, trimmed)

```json
{
  "schema_version": 1,
  "model": "hist_gradient_boosting",
  "generated_at": "2026-06-17T03:21:00+00:00",
  "n_provinces": 77,
  "provinces": [
    {
      "id": 1,
      "code": "BKK",
      "name_th": "กรุงเทพมหานคร",
      "name_en": "Bangkok",
      "region": "Central",
      "lat": 13.7563,
      "lon": 100.5018,
      "issue_date": "2026-06-01",
      "forecasts": [
        {
          "lead_weeks": 2,
          "probability": 0.18,
          "climatology_base_rate": 0.11,
          "ratio_vs_normal": 1.64,
          "risk_level_th": "ค่อนข้างสูง",
          "risk_level_en": "Elevated"
        },
        {
          "lead_weeks": 3,
          "probability": 0.09,
          "climatology_base_rate": 0.11,
          "ratio_vs_normal": 0.82,
          "risk_level_th": "ปกติ",
          "risk_level_en": "Normal"
        }
      ]
    }
  ]
}
```

> The example shows 2 of the 5 required forecast buckets for brevity. A valid artifact contains all of `lead_weeks` 2, 3, 4, 5, 6 for every one of the 77 provinces.
