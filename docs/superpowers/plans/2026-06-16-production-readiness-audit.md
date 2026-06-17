# Production-Readiness Audit & Gate Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** ทำให้ per-province forecast แสดงข้อมูลสด ณ ปัจจุบัน (ไม่ใช่ demo 2023-12-31) + สร้างชุดเช็คความพร้อม 5 หมวด ที่ใช้เป็นทั้ง audit (รายงาน go/no-go) และ gate (ด่านบล็อกก่อน publish).

**Architecture:** Phase 1 มิเรอร์ operational mode ของ regional `predict.py` มาที่ per-province (frozen per-cell p90 threshold grid + `data/raw_recent/` + MJO impute). Phase 2 สร้าง `scripts/readiness/` (numpy/pandas-only, ทุกไฟล์มี `_selftest` ตาม pattern `scripts/analysis/`) ที่อ่าน JSON/artifact ที่มีอยู่ — ไม่ retrain. Phase 3 เสียบ gate เข้า `validate_contract.py`/`publish_bridge.py` แล้วรัน audit จริง.

**Tech Stack:** Python 3.12, numpy, pandas, xarray, pickle. ทดสอบด้วย `python scripts/<module>.py test` (selftest pattern เดิม). ไม่เพิ่ม dependency ใหม่.

---

## File Structure

**Phase 1 (per-province operational):**
- Modify `scripts/build_dataset.py` — เซฟ `thr90_cell` grid ลง climatology.pkl (freeze per-cell threshold)
- Modify `scripts/build_provinces_dataset.py` — `build_provinces_features()` รับ `clim`/`tmax_dir`/`soil_dir`
- Modify `scripts/predict_provinces.py` — `build_forecast(operational=...)`, MJO impute, warnings, data_through
- Modify `scripts/publish_bridge.py` — เรียก operational predict

**Phase 2 (readiness checks) — โฟลเดอร์ใหม่ `scripts/readiness/`:**
- Create `scripts/readiness/__init__.py`
- Create `scripts/readiness/checks.py` — `CheckResult` + freshness + plausibility
- Create `scripts/readiness/data_quality.py` — NaN/domain/range/MJO
- Create `scripts/readiness/skill.py` — อ่าน outputs/analysis BSS
- Create `scripts/readiness/comms.py` — เช็คถ้อยคำ/issue_date ใน UI
- Create `scripts/readiness/audit.py` — รันทุกเช็ค → docs/readiness/AUDIT-YYYY-MM-DD.md
- Create `scripts/readiness/gate.py` — รัน blocking subset → exit 0/1

**Phase 3 (integration):**
- Modify `scripts/validate_contract.py` — เรียก readiness gate (superset)
- Create `docs/readiness/` (ออกโดย audit.py)

---

## PHASE 1 — per-province operational mode

> ⚠️ **DEFERRED — เป็นของ branch `feat/operational-province-mode` (งานคู่ขนาน).**
> Phase 1 ถูกลงมือทำบน branch อื่นแล้ว (commit `3be0b4a feat: freeze per-province climatology`,
> `528de6a`) ด้วยวิธี "freeze per-province climatology (thr90 grid + per-province base_rate)"
> — ต่างจากดีไซน์เดิมด้านล่าง (reuse per-cell grid ของ regional) แต่เป้าหมายเดียวกัน: issue_date สด.
> **branch นี้ (`feat/production-readiness-audit`) ทำเฉพาะ Phase 2-3** แล้ว merge รวมตอนท้าย.
> Task 1-4 ด้านล่างคงไว้เป็นบันทึกดีไซน์ — **ไม่ต้อง execute ที่นี่**.

### Task 1: Freeze per-cell p90 threshold grid ลง climatology

**Files:**
- Modify: `scripts/build_dataset.py:283-293` (build()) + `scripts/build_dataset.py:213-272` (build_feature_table return)
- Test: selftest ใน `scripts/build_dataset.py`

บริบท: ตอนนี้ `build()` คำนวณ `thr90_cell = doy_window_percentile(t_grid, q=90, window=PCTL_WINDOW)` (บรรทัด 292) เพื่อทำ area-fraction target แต่**ไม่เซฟ**. per-province ต้องใช้ค่านี้แช่แข็งตอน operational. ย้ายการคำนวณขึ้นก่อน `save_climatology` แล้วเก็บใน clim_out.

- [ ] **Step 1: เพิ่ม assert ใน selftest ว่า climatology มี thr90_cell**

ใน `_selftest()` ของ build_dataset.py (หาเมธอด selftest ที่มีอยู่; ถ้าไม่มี ใส่เช็คใน build path manual) เพิ่ม:

```python
# selftest: หลัง build() climatology.pkl ต้องมี thr90_cell (per-cell grid) สำหรับ per-province operational
from build_dataset import load_climatology
clim = load_climatology()
assert "thr90_cell" in clim, "climatology ขาด thr90_cell (per-cell threshold grid)"
assert "dayofyear" in clim["thr90_cell"].dims, "thr90_cell ต้อง index ด้วย dayofyear"
print("[OK] climatology มี thr90_cell grid")
```

- [ ] **Step 2: รัน selftest เพื่อยืนยันว่า FAIL (ยังไม่เซฟ thr90_cell)**

Run: `PYTHONIOENCODING=utf-8 python scripts/build_dataset.py test`
Expected: AssertionError "climatology ขาด thr90_cell" (ถ้า selftest โหลด clim เดิมที่ยังไม่มี) — ถ้า selftest ไม่ได้ build ใหม่ ให้ข้ามไป Step 3 แล้วยืนยันด้วย Step 4

- [ ] **Step 3: ย้าย thr90_cell ขึ้นก่อน save_climatology และเก็บลง clim_out**

แก้ `build()` ใน `scripts/build_dataset.py` — ย้ายบรรทัด 292 ขึ้นมาก่อน 288 และเพิ่มลง clim_out:

```python
    feat, daily, t_grid, clim_out = build_feature_table(verbose=verbose)
    clim_out["mjo_means"] = {c: float(feat[c].mean())
                             for c in ["mjo_rmm1", "mjo_rmm2", "mjo_amp", "mjo_sin", "mjo_cos"]}
    # per-cell p90 threshold (ราย doy ราย cell) — แช่แข็งไว้สำหรับ per-province operational
    log("[af] เกณฑ์ p90 ราย doy ต่อเซลล์ (ช้าสุดในไฟล์นี้ ~นาที) ...")
    thr90_cell = doy_window_percentile(t_grid, q=90, window=PCTL_WINDOW)
    clim_out["thr90_cell"] = thr90_cell
    save_climatology(clim_out)

    # target รอง (area-fraction รายเซลล์) ใช้ thr90_cell ที่คำนวณแล้ว
    hw_cell = flag_heatwaves(hot_days(t_grid, thr90_cell), min_len=MIN_RUN)
    area_frac = hw_cell.mean(dim=_spatial_dims(hw_cell))
    hw_af = (area_frac >= AF_THRESHOLD)
    daily["area_frac"] = to_series(area_frac, "area_frac")
    daily["hw_af"] = to_series(hw_af.astype(float), "hw_af")
```

ตรวจว่า `save_climatology` (บรรทัด ~118-123) เซฟทั้ง dict — ใช้ `pickle.dump({k: v for k, v in clim.items()}, f)` อยู่แล้ว → xarray DataArray pickle ได้ ✓

- [ ] **Step 4: rebuild dataset (เซฟ climatology ใหม่) แล้วรัน selftest**

Run: `PYTHONIOENCODING=utf-8 python scripts/build_dataset.py` (rebuild — ~ไม่กี่นาที, เซฟ climatology.pkl ใหม่)
จากนั้น: `PYTHONIOENCODING=utf-8 python scripts/build_dataset.py test`
Expected: PASS "[OK] climatology มี thr90_cell grid"

- [ ] **Step 5: Commit**

```bash
git add scripts/build_dataset.py models/climatology.pkl
git commit -m "feat: freeze per-cell p90 threshold grid in climatology (for per-province operational)"
```

---

### Task 2: build_provinces_features รับ clim/tmax_dir/soil_dir (operational)

**Files:**
- Modify: `scripts/build_provinces_dataset.py:36-87`
- Test: selftest ใน `scripts/build_provinces_dataset.py:113-127`

บริบท: ตอนนี้ `build_provinces_features()` recompute `thr90 = doy_window_percentile(t_grid,...)` (บรรทัด 59) จาก TMAX_DIR เต็มเสมอ. operational ต้องใช้ frozen `clim["thr90_cell"]` + recent dirs.

- [ ] **Step 1: เขียน selftest operational parity (failing)**

แก้ `_selftest()` ใน `scripts/build_provinces_dataset.py` เพิ่มบล็อก: build features ปีเดียวจาก clim แช่แข็ง + ข้อมูลย่อย ต้องตรงกับ dataset เต็ม:

```python
    # operational parity: frozen clim + ข้อมูล "ชุดย่อย" -> feature ต้องตรง dataset เต็ม
    from build_dataset import load_climatology, TMAX_DIR as FULL_TMAX, SOIL_DIR as FULL_SOIL
    from build_provinces_dataset import OUT_FILE as DS
    if DS.exists() and (load_climatology().get("thr90_cell") is not None):
        clim = load_climatology()
        feat_op, _ = build_provinces_features(verbose=False, clim=clim,
                                              tmax_dir=FULL_TMAX, soil_dir=FULL_SOIL)
        ds = pd.read_parquet(DS)
        m = feat_op.merge(ds, on=["province_id", "date"], suffixes=("_a", "_b"))
        bad = 0
        for c in ["sm1_mean30", "tmax_rm", "hot_frac7", "in_hw_today"]:
            a = m[f"{c}_a"].to_numpy(float); b = m[f"{c}_b"].to_numpy(float)
            ok = np.isclose(a, b, rtol=1e-6, atol=1e-8) | (np.isnan(a) & np.isnan(b))
            bad += int((~ok).sum())
        assert bad == 0, f"operational parity ไม่ตรง {bad} จุด (frozen thr90_cell != recompute)"
        print("[OK] per-province operational parity (frozen clim) ตรง dataset")
```

- [ ] **Step 2: รัน selftest เพื่อยืนยัน FAIL**

Run: `PYTHONIOENCODING=utf-8 python scripts/build_provinces_dataset.py test`
Expected: FAIL — `TypeError: build_provinces_features() got an unexpected keyword argument 'clim'`

- [ ] **Step 3: เพิ่ม params operational ใน build_provinces_features**

แก้ signature + การเลือก threshold + dirs ใน `scripts/build_provinces_dataset.py`:

```python
def build_provinces_features(verbose: bool = True, clim: dict | None = None,
                             tmax_dir: Path | None = None, soil_dir: Path | None = None
                             ) -> tuple[pd.DataFrame, xr.DataArray]:
    """คืน (feat_all, hw_grid). reuse ตัวเดียวทั้ง train และ serve เพื่อ parity.

    operational: ส่ง clim (มี thr90_cell แช่แข็ง) + tmax_dir/soil_dir = data/raw_recent/
    -> ใช้เกณฑ์ p90 แช่แข็ง (ไม่ recompute จากข้อมูลย่อย ~60 วัน). None = ข้อมูลเทรนเต็ม.
    """
    def log(m):
        if verbose:
            print(m, flush=True)
    tmax_dir = tmax_dir or TMAX_DIR
    log("[prov] โหลด Tmax grid + เกณฑ์ p90 ราย doy ราย cell ...")
    t_grid = load_tmax_celsius(sorted(tmax_dir.glob("era5_tmax_thailand_*.nc")))
    if clim is not None and clim.get("thr90_cell") is not None:
        thr90 = clim["thr90_cell"]                      # แช่แข็ง (operational)
    else:
        thr90 = doy_window_percentile(t_grid, q=90, window=PCTL_WINDOW)
    hot_grid = hot_days(t_grid, thr90)
    hw_grid = flag_heatwaves(hot_grid, min_len=MIN_RUN)
    log("[prov] โหลด soil moisture grid ชั้น 1, 3 ...")
    sm1_grid, sm3_grid = _soil_grid(1, soil_dir), _soil_grid(3, soil_dir)
    ...
```

และแก้ `_soil_grid` ให้รับ soil_dir:

```python
def _soil_grid(layer: int, soil_dir: Path | None = None) -> xr.DataArray:
    soil_dir = soil_dir or SOIL_DIR
    files = sorted(soil_dir.glob(f"era5_sm_l{layer}_thailand_*.nc"))
    if not files:
        raise FileNotFoundError(f"ไม่พบ soil moisture ชั้น {layer} ใน {soil_dir}")
    ...
```

หมายเหตุ: import `SOIL_DIR`, `TMAX_DIR` มีอยู่แล้ว (บรรทัด 20). `doy_window_percentile` ต้อง import เพิ่ม:
แก้บรรทัด 17 `from heatwave_target import load_tmax_celsius, doy_window_percentile, hot_days, flag_heatwaves` (มี hot_days/flag_heatwaves แล้ว เพิ่ม doy_window_percentile) — ตรวจว่ามีแล้วหรือยัง.

- [ ] **Step 4: รัน selftest เพื่อยืนยัน PASS**

Run: `PYTHONIOENCODING=utf-8 python scripts/build_provinces_dataset.py test`
Expected: PASS — "[OK] per-province operational parity (frozen clim) ตรง dataset" + selftest เดิมยังผ่าน

- [ ] **Step 5: Commit**

```bash
git add scripts/build_provinces_dataset.py
git commit -m "feat: per-province feature builder accepts frozen clim + operational dirs"
```

---

### Task 3: predict_provinces operational mode + impute + warnings

**Files:**
- Modify: `scripts/predict_provinces.py:38-71` (build_forecast), `:93-99` (predict), CLI block ท้ายไฟล์
- Test: selftest ใน `scripts/predict_provinces.py:74-90`

บริบท: ปัจจุบัน `build_forecast()` เรียก `build_provinces_features()` (เต็ม) แล้ว `iloc[-1]` = 2023-12-31. ต้องเพิ่ม operational: ใช้ clim แช่แข็ง + raw_recent + impute MJO + ติดธง warnings/in_training_domain + data_through.

- [ ] **Step 1: เขียน selftest ว่า operational ให้ issue_date ใหม่กว่า demo**

เพิ่มใน `_selftest()` ของ `scripts/predict_provinces.py`:

```python
    # operational: ต้องไม่ล็อกที่วันเทรนล่าสุด (2023-12-31) ถ้ามี raw_recent + clim
    from build_dataset import load_climatology
    from build_provinces_dataset import RECENT_TMAX_DIR if False else None  # placeholder ลบออก
```

(แก้เป็นเช็คจริง — ดู Step 3 ว่า RECENT dirs อยู่ที่ไหน) ใช้บล็อกนี้แทน:

```python
    # operational build_forecast: ถ้ามี raw_recent ต้องได้ issue_date != วันเทรนล่าสุด demo
    import os
    from build_dataset import load_climatology, RECENT_TMAX_DIR, RECENT_SOIL_DIR
    clim = load_climatology()
    if RECENT_TMAX_DIR.exists() and list(RECENT_TMAX_DIR.glob("*.nc")) and clim.get("thr90_cell") is not None:
        fc_op = build_forecast(operational=True)
        assert fc_op["provinces"], "operational forecast ว่าง"
        iss = fc_op["provinces"][0]["issue_date"]
        assert iss != "2023-12-31", f"operational ยังล็อกที่ demo date {iss}"
        assert "data_through" in fc_op, "operational forecast ขาด data_through"
        print(f"[OK] per-province operational issue_date = {iss}")
```

- [ ] **Step 2: รัน selftest เพื่อยืนยัน FAIL**

Run: `PYTHONIOENCODING=utf-8 python scripts/predict_provinces.py test`
Expected: FAIL — `TypeError: build_forecast() got an unexpected keyword argument 'operational'`

- [ ] **Step 3a: parametrize impute_neutral_mjo ให้รับ feature_cols (กัน KeyError province)**

บริบท (advisor): `impute_neutral_mjo` ใน `scripts/build_dataset.py:83-101` สร้าง `other = [c for c in FEATURES if c not in MJO_FEATURES]` แล้ว index `feat[other]` — `FEATURES` เป็นชุด regional. province frame มีแค่ `FEATURES_P` → `feat[other]` จะ KeyError. แก้ให้รับ `feature_cols` (DRY: impute เดียว ใช้ได้ทั้งสองฝั่ง):

```python
def impute_neutral_mjo(feat: pd.DataFrame, mjo_means: dict | None = None,
                       feature_cols: list | None = None):
    """... (docstring เดิม) ...
    feature_cols: ชุด feature ที่ต้องครบก่อน impute (regional FEATURES default ; ส่ง FEATURES_P สำหรับ province)."""
    means = mjo_means or {c: 0.0 for c in MJO_FEATURES}
    cols = feature_cols if feature_cols is not None else FEATURES
    other = [c for c in cols if c not in MJO_FEATURES]
    target = feat[MJO_FEATURES].isna().any(axis=1) & feat[other].notna().all(axis=1)
    dates = set(feat.index[target])
    if dates:
        feat = feat.copy()
        for c in MJO_FEATURES:
            feat.loc[target, c] = means.get(c, 0.0)
    return feat, dates
```

regional caller เดิม (`predict.py:128` `impute_neutral_mjo(feat, ...)`) ไม่ต้องแก้ (default = FEATURES). Commit แยก:

```bash
git add scripts/build_dataset.py
git commit -m "refactor: impute_neutral_mjo accepts feature_cols (reuse for per-province)"
```

- [ ] **Step 3b: เพิ่ม operational ใน build_forecast/predict + impute MJO**

แก้ `scripts/predict_provinces.py`. import เพิ่มบนหัวไฟล์:

```python
from build_dataset import load_climatology, RECENT_TMAX_DIR, RECENT_SOIL_DIR, impute_neutral_mjo, MJO_FEATURES
from build_provinces_dataset import FEATURES_P  # ถ้ายังไม่ import ; ใช้ส่ง feature_cols
```

แก้ `build_forecast` ให้รับ operational และส่ง clim/dirs + impute + warnings:

```python
def build_forecast(operational: bool = False) -> dict:
    if operational:
        clim = load_climatology()
        feat_all, _ = build_provinces_features(verbose=False, clim=clim,
                                               tmax_dir=RECENT_TMAX_DIR, soil_dir=RECENT_SOIL_DIR)
        mjo_means = clim.get("mjo_means")
    else:
        feat_all, _ = build_provinces_features(verbose=False)
        mjo_means = None
    arts = _load_arts()
    pv = load_provinces().set_index("id")
    skill = pd.read_csv(SKILL_CSV) if SKILL_CSV.exists() else pd.DataFrame()
    provinces_out = []
    data_through = None
    for pid, g in feat_all.groupby("province_id"):
        g = g.sort_values("date").set_index("date")
        warnings_p = []
        if operational:
            g, imputed = impute_neutral_mjo(g, mjo_means, feature_cols=FEATURES_P)  # MJO impute (FEATURES_P!)
        else:
            imputed = set()
        valid = g.dropna(subset=FEATURES_P)
        if valid.empty:
            continue
        row = valid.iloc[-1]
        row_date = pd.Timestamp(row.name)
        data_through = row_date if data_through is None else max(data_through, row_date)
        if row_date in imputed:
            warnings_p.append("MJO ไม่อัปเดตถึงวันออกพยากรณ์ — ใช้ค่า MJO กลางแทน")
        X = row[FEATURES_P].to_numpy(float).reshape(1, -1)
        info = pv.loc[pid]
        fcs = []
        for L in LEADS:
            a = arts[L]
            p = float(a["calibrator"].transform(a["estimator"].predict_proba(X)[:, 1])[0])
            th, en, ratio = risk_level(p, a["base_rate"])
            fcs.append({"lead_weeks": L, "probability": round(p, 4),
                        "climatology_base_rate": round(a["base_rate"], 4),
                        "ratio_vs_normal": round(ratio, 2),
                        "risk_level_th": th, "risk_level_en": en})
        provinces_out.append({"id": int(pid), "code": info["code"],
                              "name_th": info["name_th"], "name_en": info["name_en"],
                              "region": info["region"], "lat": float(info["lat"]),
                              "lon": float(info["lon"]),
                              "issue_date": str(row_date.date()),
                              "warnings": warnings_p,
                              "forecasts": fcs})
    out = {"schema_version": 1, "model": arts[LEADS[0]]["model_name"],
           "generated_at": datetime.now(timezone.utc).isoformat(),
           "data_through": str(data_through.date()) if data_through is not None else None,
           "n_provinces": len(provinces_out), "provinces": provinces_out}
    if not skill.empty:
        out["skill"] = skill.to_dict(orient="records")
    return out
```

หมายเหตุ: `impute_neutral_mjo(feat, mjo_means)` ทำงานบน DataFrame ที่ index เป็น date และมีคอลัมน์ MJO_FEATURES (ตรงกับ feat_all ราย province หลัง set_index("date")). FEATURES_P มี MJO อยู่แล้ว.

แก้ `predict()` ให้ส่ง operational ต่อ:

```python
def predict(operational: bool = False, verbose: bool = True) -> dict:
    fc = build_forecast(operational=operational)
    OUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    OUT_FILE.write_text(json.dumps(fc, ensure_ascii=False, indent=2), encoding="utf-8")
    if verbose:
        print(f"[OK] {OUT_FILE} | {fc['n_provinces']} จังหวัด | issue {fc['provinces'][0]['issue_date'] if fc['provinces'] else '-'}")
    return fc
```

แก้ CLI block ท้ายไฟล์ ให้รับ `operational`:

```python
if __name__ == "__main__":
    import sys
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass
    if len(sys.argv) > 1 and sys.argv[1] == "test":
        _selftest()
    elif len(sys.argv) > 1 and sys.argv[1] == "operational":
        predict(operational=True)
    else:
        predict()
```

- [ ] **Step 3c: ยืนยันว่า province artifacts เทรนทั้งปี (advisor #4 — ตัดสินว่าต้องมี domain guard ไหม)**

Run: `PYTHONIOENCODING=utf-8 python -c "import pickle,glob; [print(f, {k:pickle.load(open(f,'rb')).get(k) for k in ('train_issue_doy_min','train_issue_doy_max')}) for f in glob.glob('models/heatwave_prov_lead*.pkl')]"`
Expected: ถ้า doy span ~1–365 (เทรนทั้งปี) → **ไม่ต้องมี domain guard** ; เพิ่มคอมเมนต์ใน build_forecast:
```python
        # หมายเหตุ: ข้อมูลเทรนทั้งปี (ม.ค.-ธ.ค.) → issue date ใด ๆ in-domain ; ไม่ต้องมี train_issue_doy guard แบบ regional
```
ถ้า key ไม่มี/span แคบ (เช่น ม.ค.-ก.ค. เท่านั้น) → ต้องเพิ่ม guard + warning แบบ regional `predict.py:159-165` (port มา) ก่อนไปต่อ.

- [ ] **Step 4: รัน selftest เพื่อยืนยัน PASS**

Run: `PYTHONIOENCODING=utf-8 python scripts/predict_provinces.py test`
Expected: PASS — parity เดิมผ่าน + "[OK] per-province operational issue_date = <วันล่าสุด>" (ไม่ใช่ 2023-12-31)

- [ ] **Step 5: Commit**

```bash
git add scripts/predict_provinces.py
git commit -m "feat: per-province operational mode (frozen clim + raw_recent + MJO impute)"
```

---

### Task 4: publish_bridge เรียก operational predict

**Files:**
- Modify: `scripts/publish_bridge.py:102-104`

- [ ] **Step 1: แก้ให้ predict_provinces ใช้ operational**

แก้ `scripts/publish_bridge.py` บรรทัด ~103:

```python
    if not args.no_predict:
        import predict_provinces  # lazy: ดึง deps หนักเฉพาะตอน predict
        predict_provinces.predict(operational=True, verbose=True)
```

- [ ] **Step 2: รัน bridge แบบ no-predict (ไม่ต้องมี CDS) เพื่อยืนยันไม่พัง import**

Run: `PYTHONIOENCODING=utf-8 python scripts/publish_bridge.py --no-predict`
Expected: validate + sync ทำงาน (ใช้ไฟล์เดิม) — ไม่ error

- [ ] **Step 3: Commit**

```bash
git add scripts/publish_bridge.py
git commit -m "feat: publish_bridge runs per-province operational predict"
```

> **หมายเหตุ:** การรัน operational จริง (มี raw_recent สด) เป็น manual step ของผู้ใช้ (ต้องมี CDS key + download recent). Task นี้แค่ wire path ให้ถูก.

---

## PHASE 2 — readiness checks (audit + gate)

### Task 5: CheckResult + freshness checks

**Files:**
- Create: `scripts/readiness/__init__.py` (ว่าง)
- Create: `scripts/readiness/checks.py`

- [ ] **Step 1: เขียน checks.py พร้อม CheckResult + freshness + selftest**

```python
"""ชุดเช็คความพร้อม production: โครงสร้างผลลัพธ์ + freshness + plausibility.
อ่าน contract JSON ที่จะ publish — ไม่ retrain. ใช้ทั้ง audit (ทุกเช็ค) และ gate (blocking).
"""
from __future__ import annotations
import sys
from dataclasses import dataclass
from datetime import date, datetime, timezone

PASS, WARN, FAIL = "PASS", "WARN", "FAIL"
MAX_ISSUE_AGE_DAYS = 10          # issue_date เก่ากว่านี้ = ข้อมูลไม่สด
MAX_HIGH_FRACTION = 0.80         # จังหวัดที่ขึ้น High พร้อมกันเกินสัดส่วนนี้ = น่าสงสัย
MAX_RATIO = 6.0                  # ratio_vs_normal เกินนี้ = ผิดปกติ


@dataclass
class CheckResult:
    name: str
    category: str
    status: str           # PASS | WARN | FAIL
    detail: str
    blocking: bool = False   # True = ถ้า FAIL ห้าม publish

    def failed_block(self) -> bool:
        return self.blocking and self.status == FAIL


def _today() -> date:
    return datetime.now(timezone.utc).date()


def check_issue_date_fresh(obj: dict) -> CheckResult:
    """Freshness: issue_date (ไม่ใช่ generated_at) ต้องไม่เก่าเกิน MAX_ISSUE_AGE_DAYS."""
    cat = "freshness"
    provs = obj.get("provinces") or []
    if not provs:
        return CheckResult("issue_date_fresh", cat, FAIL, "ไม่มี provinces", blocking=True)
    issues = {p.get("issue_date") for p in provs}
    try:
        ages = [( _today() - date.fromisoformat(d)).days for d in issues if d]
    except (ValueError, TypeError):
        return CheckResult("issue_date_fresh", cat, FAIL, f"issue_date parse ไม่ได้: {issues}", blocking=True)
    if not ages:
        return CheckResult("issue_date_fresh", cat, FAIL, "ไม่มี issue_date", blocking=True)
    worst = max(ages)
    if worst > MAX_ISSUE_AGE_DAYS:
        return CheckResult("issue_date_fresh", cat, FAIL,
                           f"issue_date เก่า {worst} วัน (เกิน {MAX_ISSUE_AGE_DAYS}) — อาจเป็นข้อมูล demo/backtest",
                           blocking=True)
    return CheckResult("issue_date_fresh", cat, PASS, f"issue_date เก่าสุด {worst} วัน", blocking=True)


def check_all_high_fraction(obj: dict) -> CheckResult:
    """Plausibility (WARN เท่านั้น — ไม่ blocking): สัดส่วนจังหวัด 'ทุก lead = High'.

    หมายเหตุสำคัญ (advisor): ห้าม blocking. ช่วง El Niño แรงจริง (เช่น 2023→ต้นปี 2024 ไทยร้อนทำลายสถิติ)
    'เกือบทุกจังหวัด High หลายสัปดาห์' = สัญญาณเตือนที่ "ถูกต้อง" และเป็นตอนที่ประชาชนต้องการเตือนที่สุด.
    ถ้า block ไว้ = fail closed ทับสัญญาณจริง (ตรงข้ามเป้าหมาย). กรณี demo 2023 ถูกจับโดย freshness แล้ว.
    ที่นี่แค่ WARN ให้คนดูยืนยันด้วยตา."""
    cat = "plausibility"
    provs = obj.get("provinces") or []
    if not provs:
        return CheckResult("all_high_fraction", cat, WARN, "ไม่มี provinces")
    n_all_high = 0
    for p in provs:
        fcs = p.get("forecasts") or []
        if fcs and all(f.get("risk_level_en") == "High" for f in fcs):
            n_all_high += 1
    frac = n_all_high / len(provs)
    status = WARN if frac > MAX_HIGH_FRACTION else PASS
    note = " — ตรวจด้วยตา (อาจถูกต้องถ้า El Niño แรง / อาจผิดปกติ)" if status == WARN else ""
    return CheckResult("all_high_fraction", cat, status,
                       f"{n_all_high}/{len(provs)} จังหวัด High ทุก lead ({frac:.0%}){note}")


def check_ratio_bounds(obj: dict) -> CheckResult:
    """Plausibility: ratio_vs_normal ทุกค่าต้อง <= MAX_RATIO."""
    cat = "plausibility"
    worst = 0.0
    for p in obj.get("provinces") or []:
        for f in p.get("forecasts") or []:
            r = f.get("ratio_vs_normal")
            if isinstance(r, (int, float)) and not isinstance(r, bool):
                worst = max(worst, float(r))
    if worst > MAX_RATIO:
        return CheckResult("ratio_bounds", cat, WARN,
                           f"ratio_vs_normal สูงสุด {worst} (เกิน {MAX_RATIO}) — ตรวจสอบความสมเหตุสมผล")
    return CheckResult("ratio_bounds", cat, PASS, f"ratio_vs_normal สูงสุด {worst}")


FRESHNESS_PLAUSIBILITY = [check_issue_date_fresh, check_all_high_fraction, check_ratio_bounds]


def _selftest() -> None:
    today = _today().isoformat()
    good = {"provinces": [
        {"issue_date": today, "forecasts": [{"risk_level_en": "Normal", "ratio_vs_normal": 1.2}]},
        {"issue_date": today, "forecasts": [{"risk_level_en": "Low", "ratio_vs_normal": 0.8}]},
    ]}
    assert check_issue_date_fresh(good).status == PASS
    assert check_all_high_fraction(good).status == PASS
    # negative: ข้อมูลเก่า 2023 ต้อง FAIL + blocking
    stale = {"provinces": [{"issue_date": "2023-12-31",
                            "forecasts": [{"risk_level_en": "High", "ratio_vs_normal": 4.0}]}]}
    r = check_issue_date_fresh(stale)
    assert r.status == FAIL and r.blocking, "ข้อมูลเก่าต้อง FAIL+blocking"
    # ทุกจังหวัด High ทุก lead = WARN (ไม่ใช่ FAIL/blocking — อาจเป็นสัญญาณจริงช่วง El Niño)
    allhigh = {"provinces": [{"issue_date": today, "forecasts": [{"risk_level_en": "High", "ratio_vs_normal": 5.0}]}
                             for _ in range(10)]}
    r2 = check_all_high_fraction(allhigh)
    assert r2.status == WARN and not r2.blocking, "all-High ต้อง WARN ไม่ block (กันทับสัญญาณจริง)"
    print("[OK] checks.py self-test ผ่าน (freshness + plausibility + negative cases)")


if __name__ == "__main__":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass
    _selftest()
```

- [ ] **Step 2: รัน selftest**

Run: `PYTHONIOENCODING=utf-8 python scripts/readiness/checks.py`
Expected: PASS "[OK] checks.py self-test ผ่าน ..."

- [ ] **Step 3: Commit**

```bash
git add scripts/readiness/__init__.py scripts/readiness/checks.py
git commit -m "feat: readiness checks - freshness + plausibility (CheckResult)"
```

---

### Task 6: data_quality checks

**Files:**
- Create: `scripts/readiness/data_quality.py`

- [ ] **Step 1: เขียน data_quality.py**

```python
"""เช็คคุณภาพ contract ที่จะ publish: schema completeness, prob/base_rate ในช่วง, ไม่มี NaN/None."""
from __future__ import annotations
import math
import sys
from scripts_readiness_import import *  # ลบบรรทัดนี้ — ดู import จริงด้านล่าง
```

แก้ import ให้ถูก (อยู่ใน package เดียวกัน):

```python
"""เช็คคุณภาพ contract ที่จะ publish: prob/base_rate ในช่วง, ไม่มี NaN/None, leads ครบ."""
from __future__ import annotations
import math
import sys

from checks import CheckResult, PASS, WARN, FAIL  # รันแบบ script จาก scripts/readiness/

EXPECTED_LEADS = {2, 3, 4, 5, 6}


def _num_ok(x) -> bool:
    return isinstance(x, (int, float)) and not isinstance(x, bool) and not math.isnan(x)


def check_no_nan_probs(obj: dict) -> CheckResult:
    cat = "data_quality"
    bad = []
    for p in obj.get("provinces") or []:
        for f in p.get("forecasts") or []:
            prob = f.get("probability")
            if not (_num_ok(prob) and 0 <= prob <= 1):
                bad.append(f"{p.get('code')} lead{f.get('lead_weeks')}={prob!r}")
    if bad:
        return CheckResult("no_nan_probs", cat, FAIL,
                           f"probability เสีย/นอกช่วง {len(bad)} จุด: {bad[:5]}", blocking=True)
    return CheckResult("no_nan_probs", cat, PASS, "probability ครบและอยู่ใน [0,1]", blocking=True)


def check_leads_complete(obj: dict) -> CheckResult:
    cat = "data_quality"
    bad = []
    for p in obj.get("provinces") or []:
        leads = {f.get("lead_weeks") for f in (p.get("forecasts") or [])}
        if leads != EXPECTED_LEADS:
            bad.append(f"{p.get('code')}={sorted(x for x in leads if x is not None)}")
    if bad:
        return CheckResult("leads_complete", cat, FAIL,
                           f"leads ไม่ครบ {sorted(EXPECTED_LEADS)} ที่ {len(bad)} จังหวัด: {bad[:5]}", blocking=True)
    return CheckResult("leads_complete", cat, PASS, f"ทุกจังหวัดมี leads ครบ {sorted(EXPECTED_LEADS)}", blocking=True)


def check_mjo_warning(obj: dict) -> CheckResult:
    """ถ้ามี warning MJO impute -> WARN (ไม่บล็อก แต่ต้องรู้)."""
    cat = "data_quality"
    n = sum(1 for p in (obj.get("provinces") or [])
            if any("MJO" in w for w in (p.get("warnings") or [])))
    if n:
        return CheckResult("mjo_warning", cat, WARN, f"{n} จังหวัดใช้ค่า MJO กลาง (impute) — แหล่ง MJO อาจล่าช้า")
    return CheckResult("mjo_warning", cat, PASS, "ไม่มี MJO impute")


DATA_QUALITY = [check_no_nan_probs, check_leads_complete, check_mjo_warning]


def _selftest() -> None:
    good = {"provinces": [{"code": "BKK", "warnings": [],
            "forecasts": [{"lead_weeks": L, "probability": 0.3} for L in (2, 3, 4, 5, 6)]}]}
    assert check_no_nan_probs(good).status == PASS
    assert check_leads_complete(good).status == PASS
    bad = {"provinces": [{"code": "BKK", "warnings": ["MJO ..."],
           "forecasts": [{"lead_weeks": 2, "probability": float("nan")}]}]}
    assert check_no_nan_probs(bad).status == FAIL
    assert check_leads_complete(bad).status == FAIL
    assert check_mjo_warning(bad).status == WARN
    print("[OK] data_quality.py self-test ผ่าน")


if __name__ == "__main__":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass
    _selftest()
```

> **หมายเหตุ import:** modules ใน `scripts/readiness/` รันแบบ standalone (`python scripts/readiness/X.py`). ใช้ `from checks import ...` ได้เพราะ Python ใส่ directory ของ script ลง sys.path. audit.py/gate.py จะ import แบบเดียวกัน.

- [ ] **Step 2: รัน selftest**

Run: `PYTHONIOENCODING=utf-8 python scripts/readiness/data_quality.py`
Expected: PASS "[OK] data_quality.py self-test ผ่าน"

- [ ] **Step 3: Commit**

```bash
git add scripts/readiness/data_quality.py
git commit -m "feat: readiness data_quality checks (nan/leads/mjo)"
```

---

### Task 7: skill check (อ่าน outputs/analysis BSS — ไม่ retrain)

**Files:**
- Create: `scripts/readiness/skill.py`

- [ ] **Step 1: ค้นหาไฟล์ BSS ที่มีอยู่**

Run: `ls outputs/analysis/ && ls outputs/analysis/skill_by_season/ 2>/dev/null`
Expected: เห็นไฟล์ผล (เช่น results_master.md, bss_*.csv) — ใช้กำหนด path จริงใน Step 2

- [ ] **Step 2: เขียน skill.py (อ่านไฟล์ผลที่เจอใน Step 1)**

```python
"""เช็คว่าโมเดลยังมี skill: อ่าน BSS ที่ analysis รันไว้ (ไม่ retrain). WARN ถ้าหาไฟล์ไม่เจอ."""
from __future__ import annotations
import csv
import sys
from pathlib import Path

from checks import CheckResult, PASS, WARN, FAIL

ROOT = Path(__file__).resolve().parent.parent.parent
# ปรับ path ตามที่เจอใน Step 1:
BSS_CSV = ROOT / "outputs" / "analysis" / "skill_by_season" / "bss_by_season.csv"


def check_bss_positive() -> CheckResult:
    cat = "skill"
    if not BSS_CSV.exists():
        return CheckResult("bss_positive", cat, WARN,
                           f"ไม่พบไฟล์ BSS ({BSS_CSV.name}) — รัน analysis ก่อนเพื่อยืนยัน skill")
    vals = []
    with open(BSS_CSV, encoding="utf-8") as f:
        for row in csv.DictReader(f):
            for k, v in row.items():
                if "bss" in k.lower():
                    try:
                        vals.append(float(v))
                    except (ValueError, TypeError):
                        pass
    if not vals:
        return CheckResult("bss_positive", cat, WARN, f"อ่านค่า BSS จาก {BSS_CSV.name} ไม่ได้")
    neg = [v for v in vals if v <= 0]
    if neg:
        return CheckResult("bss_positive", cat, WARN,
                           f"พบ BSS <= 0 จำนวน {len(neg)}/{len(vals)} ค่า — skill อ่อนบางฤดู/lead")
    return CheckResult("bss_positive", cat, PASS,
                       f"BSS เป็นบวกทุกค่า ({len(vals)} ค่า, ต่ำสุด {min(vals):.3f})")


SKILL = [check_bss_positive]


def _selftest() -> None:
    r = check_bss_positive()
    assert r.status in (PASS, WARN, FAIL)
    print(f"[OK] skill.py self-test ผ่าน (status={r.status}: {r.detail})")


if __name__ == "__main__":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass
    _selftest()
```

> ถ้า Step 1 พบชื่อไฟล์/คอลัมน์ต่างจากนี้ ให้แก้ `BSS_CSV` และ logic การอ่านให้ตรง.

- [ ] **Step 3: รัน selftest**

Run: `PYTHONIOENCODING=utf-8 python scripts/readiness/skill.py`
Expected: PASS (status เป็น PASS หรือ WARN ตามไฟล์ที่มี)

- [ ] **Step 4: Commit**

```bash
git add scripts/readiness/skill.py
git commit -m "feat: readiness skill check (reads existing BSS, no retrain)"
```

---

### Task 8: risk communication check (UI)

**Files:**
- Create: `scripts/readiness/comms.py`

- [ ] **Step 1: ตรวจว่า UI พูดอะไรอยู่**

Run: `grep -l "ความน่าจะเป็น\|issue_date\|probability" docs/index.html docs/*.html`
Expected: เห็นว่าไฟล์ UI ไหนมีคำเหล่านี้ — ยืนยัน path ที่จะเช็ค

- [ ] **Step 2: เขียน comms.py**

```python
"""เช็คการสื่อสารบนเว็บ (กันตื่นตระหนก): UI ต้องสื่อ 'โอกาสเกิด ไม่ใช่ความรุนแรง' + โชว์ issue_date."""
from __future__ import annotations
import sys
from pathlib import Path

from checks import CheckResult, PASS, WARN, FAIL

ROOT = Path(__file__).resolve().parent.parent.parent
UI_FILES = [ROOT / "docs" / "index.html"]
# คำที่สื่อ "ความน่าจะเป็นการเกิด" (อย่างน้อยหนึ่งคำ)
PROB_PHRASES = ["ความน่าจะเป็น", "โอกาสเกิด", "probability", "โอกาส"]
ISSUE_PHRASES = ["issue_date", "วันออกพยากรณ์", "ออกพยากรณ์", "ข้อมูล ณ", "data_through"]


def check_ui_communication() -> CheckResult:
    cat = "communication"
    missing = []
    for f in UI_FILES:
        if not f.exists():
            missing.append(f"{f.name} ไม่พบ")
            continue
        text = f.read_text(encoding="utf-8", errors="ignore")
        if not any(ph in text for ph in PROB_PHRASES):
            missing.append(f"{f.name}: ไม่พบคำสื่อ 'ความน่าจะเป็น/โอกาสเกิด'")
        if not any(ph in text for ph in ISSUE_PHRASES):
            missing.append(f"{f.name}: ไม่พบการโชว์วันออกพยากรณ์ (issue_date)")
    if missing:
        return CheckResult("ui_communication", cat, WARN,
                           "เว็บอาจสื่อสารไม่ชัด: " + " ; ".join(missing))
    return CheckResult("ui_communication", cat, PASS,
                       "UI สื่อ 'ความน่าจะเป็น' + โชว์วันออกพยากรณ์")


COMMS = [check_ui_communication]


def _selftest() -> None:
    r = check_ui_communication()
    assert r.status in (PASS, WARN, FAIL)
    print(f"[OK] comms.py self-test ผ่าน (status={r.status}: {r.detail})")


if __name__ == "__main__":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass
    _selftest()
```

- [ ] **Step 3: รัน selftest**

Run: `PYTHONIOENCODING=utf-8 python scripts/readiness/comms.py`
Expected: PASS (status PASS หรือ WARN)

- [ ] **Step 4: Commit**

```bash
git add scripts/readiness/comms.py
git commit -m "feat: readiness risk-communication check (UI wording + issue_date)"
```

---

### Task 9: audit.py (รวมทุกเช็ค → รายงาน go/no-go)

**Files:**
- Create: `scripts/readiness/audit.py`

- [ ] **Step 1: เขียน audit.py**

```python
"""Audit: รันทุกเช็ค (5 หมวด) บน contract ที่จะ publish -> รายงาน go/no-go (markdown)."""
from __future__ import annotations
import json
import sys
from datetime import date
from pathlib import Path

from checks import FRESHNESS_PLAUSIBILITY, PASS, WARN, FAIL, CheckResult
from data_quality import DATA_QUALITY
from skill import SKILL
from comms import COMMS

ROOT = Path(__file__).resolve().parent.parent.parent
CONTRACT = ROOT / "docs" / "forecast_provinces.json"
OUT_DIR = ROOT / "docs" / "readiness"


def run_all(obj: dict) -> list[CheckResult]:
    results = []
    for fn in FRESHNESS_PLAUSIBILITY + DATA_QUALITY:
        results.append(fn(obj))          # เช็คที่รับ contract
    for fn in SKILL + COMMS:
        results.append(fn())             # เช็คที่อ่านไฟล์ระบบ (ไม่รับ contract)
    return results


def render_report(results: list[CheckResult]) -> tuple[str, bool]:
    blockers = [r for r in results if r.failed_block()]
    go = not blockers
    lines = [f"# Production Readiness Audit — {date.today().isoformat()}", ""]
    lines.append(f"**ผล: {'✅ GO (พร้อม)' if go else '🔴 NO-GO (ยังไม่พร้อม)'}**")
    if blockers:
        lines.append(f"\nblocker {len(blockers)} ข้อ (ต้องแก้ก่อน publish):")
        for r in blockers:
            lines.append(f"- 🔴 [{r.category}] {r.name}: {r.detail}")
    lines += ["", "| หมวด | เช็ค | สถานะ | รายละเอียด |", "| --- | --- | --- | --- |"]
    icon = {PASS: "✅", WARN: "⚠️", FAIL: "🔴"}
    for r in results:
        block = " (blocking)" if r.blocking else ""
        lines.append(f"| {r.category} | {r.name}{block} | {icon[r.status]} {r.status} | {r.detail} |")
    return "\n".join(lines) + "\n", go


def main(argv) -> int:
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass
    path = Path(argv[1]) if len(argv) > 1 else CONTRACT
    obj = json.loads(path.read_text(encoding="utf-8"))
    results = run_all(obj)
    report, go = render_report(results)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out = OUT_DIR / f"AUDIT-{date.today().isoformat()}.md"
    out.write_text(report, encoding="utf-8")
    print(report)
    print(f"\n[เขียนรายงาน] {out}")
    return 0 if go else 1


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
```

- [ ] **Step 2: รัน audit บน contract ปัจจุบัน (คาดว่า NO-GO เพราะ 2023)**

Run: `PYTHONIOENCODING=utf-8 python scripts/readiness/audit.py`
Expected: พิมพ์รายงาน + exit 1 (NO-GO) เพราะ issue_date 2023 = freshness FAIL blocking ; เขียนไฟล์ `docs/readiness/AUDIT-2026-06-16.md`

- [ ] **Step 3: Commit**

```bash
git add scripts/readiness/audit.py docs/readiness/
git commit -m "feat: readiness audit report (go/no-go across 5 categories)"
```

---

### Task 10: gate.py (blocking subset → exit code + negative selftest)

**Files:**
- Create: `scripts/readiness/gate.py`

- [ ] **Step 1: เขียน gate.py พร้อม negative selftest**

```python
"""Gate: รันเฉพาะเช็ค blocking บน contract -> exit 1 ถ้ามี blocker (ใช้ก่อน publish)."""
from __future__ import annotations
import json
import sys
from pathlib import Path

from checks import FRESHNESS_PLAUSIBILITY, FAIL, PASS
from data_quality import DATA_QUALITY

ROOT = Path(__file__).resolve().parent.parent.parent
CONTRACT = ROOT / "docs" / "forecast_provinces.json"


def gate_results(obj: dict):
    return [fn(obj) for fn in FRESHNESS_PLAUSIBILITY + DATA_QUALITY]


def run_gate(obj: dict) -> tuple[bool, list]:
    results = gate_results(obj)
    blockers = [r for r in results if r.failed_block()]
    return (not blockers), blockers


def _selftest() -> None:
    from datetime import datetime, timezone
    today = datetime.now(timezone.utc).date().isoformat()
    good = {"provinces": [{"code": "BKK", "issue_date": today, "warnings": [],
            "forecasts": [{"lead_weeks": L, "probability": 0.3, "risk_level_en": "Normal",
                           "ratio_vs_normal": 1.1} for L in (2, 3, 4, 5, 6)]}]}
    ok, blk = run_gate(good)
    assert ok and not blk, f"contract ดีต้องผ่าน gate: {[b.detail for b in blk]}"
    # negative: ข้อมูล 2023 ต้องโดนบล็อก
    stale = json.loads(json.dumps(good))
    stale["provinces"][0]["issue_date"] = "2023-12-31"
    ok2, blk2 = run_gate(stale)
    assert not ok2 and blk2, "ข้อมูล 2023 ต้องโดน gate บล็อก"
    print(f"[OK] gate.py self-test ผ่าน (good=GO, stale=NO-GO, blocker={blk2[0].name})")


def main(argv) -> int:
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass
    if len(argv) > 1 and argv[1] == "test":
        _selftest()
        return 0
    path = Path(argv[1]) if len(argv) > 1 else CONTRACT
    obj = json.loads(path.read_text(encoding="utf-8"))
    ok, blockers = run_gate(obj)
    if not ok:
        print(f"[GATE FAIL] blocker {len(blockers)} ข้อ — ห้าม publish:")
        for b in blockers:
            print(f"  - [{b.category}] {b.name}: {b.detail}")
        return 1
    print("[GATE OK] ผ่านเช็ค blocking ทั้งหมด")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
```

- [ ] **Step 2: รัน negative selftest**

Run: `PYTHONIOENCODING=utf-8 python scripts/readiness/gate.py test`
Expected: PASS "[OK] gate.py self-test ผ่าน (good=GO, stale=NO-GO ...)"

- [ ] **Step 3: รัน gate บน contract ปัจจุบัน (คาด FAIL เพราะ 2023)**

Run: `PYTHONIOENCODING=utf-8 python scripts/readiness/gate.py`
Expected: exit 1, "[GATE FAIL] ... issue_date เก่า"

- [ ] **Step 4: Commit**

```bash
git add scripts/readiness/gate.py
git commit -m "feat: readiness gate (blocking subset, exit code, negative selftest)"
```

---

## PHASE 3 — integration

### Task 11: เสียบ gate เข้า "ประตูจริง" (publish_bridge.validate_file) + standalone CLI

**Files:**
- Modify: `scripts/publish_bridge.py:31-50` (validate_file — **ประตูที่ `--publish` ใช้จริง**)
- Modify: `scripts/validate_contract.py:118-143` (main — standalone CLI ให้สอดคล้อง)

บริบท (advisor #2): `publish_bridge.main()` เรียก `validate_file()` ซึ่งเรียก **ฟังก์ชัน** `vc.validate_contract(obj)` (schema-only, บรรทัด 43) — **ไม่ใช่** `vc.main()`. ถ้าเสียบ gate แค่ใน `main()` จะไม่กั้น `--publish`. ต้องเสียบที่ `validate_file()` ด้วย.

- [ ] **Step 1: เสียบ readiness gate ใน publish_bridge.validate_file (หลัง schema ผ่าน)**

แก้ `scripts/publish_bridge.py` — เพิ่มหลังบรรทัด 48 (`raise SystemExit(1)` ของ schema errs) ก่อน `print("[OK] validate ผ่าน...")`:

```python
    # readiness gate (freshness/plausibility/data-quality blocking) — ชั้นที่สอง ก่อน distribute
    sys.path.insert(0, str(Path(__file__).resolve().parent / "readiness"))
    from gate import run_gate
    ok, blockers = run_gate(obj)
    if not ok:
        print(f"[FAIL] readiness gate ไม่ผ่าน {len(blockers)} ข้อ — ยกเลิก distribute:")
        for b in blockers:
            print(f"  - [{b.category}] {b.name}: {b.detail}")
        raise SystemExit(1)
    print(f"[OK] validate ผ่าน: {len(obj['provinces'])} จังหวัด + readiness gate")
    return obj
```

(ลบบรรทัด `print(f"[OK] validate ผ่าน: {len(obj['provinces'])} จังหวัด")` + `return obj` เดิมที่ 49-50 ออก — แทนที่ด้วยบล็อกข้างบน)

- [ ] **Step 2: เสียบ gate ใน validate_contract.main ด้วย (standalone CLI สอดคล้อง)**

แก้ส่วนท้ายของ `main()` ใน `scripts/validate_contract.py` หลัง schema OK ~142:

```python
    if errs:
        print(f"[FAIL] contract ไม่ผ่าน {len(errs)} ข้อ:")
        for e in errs:
            print(f"  - {e}")
        return 1
    sys.path.insert(0, str(Path(__file__).resolve().parent / "readiness"))
    from gate import run_gate
    ok, blockers = run_gate(obj)
    if not ok:
        print(f"[FAIL] readiness gate ไม่ผ่าน {len(blockers)} ข้อ:")
        for b in blockers:
            print(f"  - [{b.category}] {b.name}: {b.detail}")
        return 1
    print(f"[OK] contract ผ่าน: {len(obj['provinces'])} จังหวัด, schema v{obj['schema_version']} + readiness gate")
    return 0
```

- [ ] **Step 3: รันทั้งสองประตูบน contract ปัจจุบัน (คาด FAIL จาก readiness)**

Run:
```
PYTHONIOENCODING=utf-8 python scripts/validate_contract.py
PYTHONIOENCODING=utf-8 python scripts/publish_bridge.py --no-predict
```
Expected: ทั้งคู่ schema OK แต่ readiness gate FAIL (issue_date 2023) → exit 1 / SystemExit(1) ; publish_bridge ต้อง **ไม่** sync/distribute

- [ ] **Step 4: Commit**

```bash
git add scripts/publish_bridge.py scripts/validate_contract.py
git commit -m "feat: readiness gate on real publish door (validate_file) + standalone CLI"
```

---

### Task 12: รัน audit จริง + ตรวจ end-to-end + แก้จนเขียว

**Files:** (อาจแก้ตามผล audit — docs/index.html ถ้า comms WARN)

- [ ] **Step 1: รัน selftest ทุกโมดูลที่แตะ (ยืนยันไม่พังของเดิม)**

Run:
```
PYTHONIOENCODING=utf-8 python scripts/build_dataset.py test
PYTHONIOENCODING=utf-8 python scripts/build_provinces_dataset.py test
PYTHONIOENCODING=utf-8 python scripts/predict_provinces.py test
PYTHONIOENCODING=utf-8 python scripts/readiness/checks.py
PYTHONIOENCODING=utf-8 python scripts/readiness/data_quality.py
PYTHONIOENCODING=utf-8 python scripts/readiness/gate.py test
```
Expected: ทุกตัว PASS

- [ ] **Step 2: รัน app tests เดิม (ไม่ทำของเดิมพัง)**

Run: `PYTHONIOENCODING=utf-8 python -m pytest app/ -q` (ถ้ามี) หรือ test runner เดิมของโปรเจกต์
Expected: ผ่านเท่าเดิม

- [ ] **Step 3: รัน audit + review รายงาน**

Run: `PYTHONIOENCODING=utf-8 python scripts/readiness/audit.py`
Expected: รายงาน NO-GO (เพราะยังไม่ได้รัน operational จริง — ต้องมี CDS). ถ้า comms WARN → แก้ถ้อยคำ docs/index.html ให้ชัด (เช่น เพิ่มประโยค "ค่านี้คือความน่าจะเป็นการเกิด ไม่ใช่ความรุนแรง" + โชว์ issue_date) แล้วรัน audit ซ้ำจน comms = PASS

- [ ] **Step 4: เขียนสรุปผล audit ลง docs/INTEGRITY.md**

เพิ่มหัวข้อใน `docs/INTEGRITY.md`:
```markdown
## Production-Readiness Audit (2026-06-16)
- Freshness gate: validate_contract ตรวจ issue_date (≤10 วัน) ไม่ใช่แค่ generated_at
- Plausibility: บล็อก all-High > 80% จังหวัด, ratio cap
- gate เสียบใน validate_contract (hard gate ก่อน distribute)
- per-province มี operational mode (frozen clim + raw_recent) แล้ว
- run audit: `python scripts/readiness/audit.py` -> docs/readiness/AUDIT-*.md
```

- [ ] **Step 5: Commit**

```bash
git add docs/INTEGRITY.md docs/index.html docs/readiness/
git commit -m "docs: production-readiness audit results + integrity note"
```

- [ ] **Step 6: อัปเดต RUNBOOK ด้วยคำสั่ง operational per-province + audit/gate**

เพิ่มใน `docs/RUNBOOK.md` (section C): `python scripts/predict_provinces.py operational`, `python scripts/readiness/audit.py`, `python scripts/readiness/gate.py`. Commit:

```bash
git add docs/RUNBOOK.md
git commit -m "docs: RUNBOOK - per-province operational + readiness audit/gate commands"
```

---

## Self-Review (ผู้เขียนแผนตรวจเอง)

**Spec coverage:** ทุกหมวดในสเปก mapped → Phase1=freshness root-cause; Task5=freshness+plausibility; Task6=data quality; Task7=skill; Task8=comms; Task9=audit; Task10/11=gate; Task12=success criteria #6 (negative test) + run audit. ✓ ครบ 6 success criteria.

**Placeholder scan:** มีจุดหนึ่งใน Task 7/8 ที่ path ขึ้นกับไฟล์จริง (`outputs/analysis/...`, UI file) — กำหนด Step 1 ให้ค้นหา path จริงก่อนเขียน เป็น discovery ที่จำเป็น ไม่ใช่ placeholder ของ logic. Task 6 Step 1 มีบรรทัด import ผิดโดยตั้งใจ (`scripts_readiness_import`) พร้อมคำสั่งให้ลบ — แก้แล้วใน code block จริงถัดมา.

**Type consistency:** `CheckResult(name, category, status, detail, blocking)` + `failed_block()` ใช้สม่ำเสมอทุก task. รายการเช็ค `FRESHNESS_PLAUSIBILITY`/`DATA_QUALITY`/`SKILL`/`COMMS` ชื่อตรงกันระหว่าง checks.py/audit.py/gate.py. ✓

**ข้อควรระวังตอน execute:** Task 1 rebuild dataset เขียนทับ climatology.pkl ที่ commit ไว้ — ค่า threshold เดิมต้องไม่เปลี่ยน (แค่เพิ่ม key thr90_cell) ; ตรวจว่า thr90_rm/thr95_rm/mjo_means เดิมยังอยู่.

---

## ⚠️ Post-merge integration (ต้องทำตอนรวม 2 branch — advisor จับ)

Phase 2-3 (branch นี้) เสร็จแล้ว แต่มี **seam ระหว่าง branch** ที่ selftest มองไม่เห็น ต้องจัดการตอน merge:

### M1 (BLOCKER — ถ้าไม่ทำ pipeline จะ deadlock): flip publish_bridge เป็น operational
ทั้ง `feat/operational-province-mode` และ branch นี้ ที่ `scripts/publish_bridge.py:~104` ยังเรียก
`predict_provinces.predict(verbose=True)` (operational ดีฟอลต์ **False** → สร้าง demo 2023).
หลัง merge: publish_bridge สร้าง demo → readiness gate (freshness) บล็อก → **publish ไม่ได้เลย**.
**แก้ 1 บรรทัด (ทำหลัง merge เมื่อ predict_provinces มี param operational แล้ว):**
```python
        predict_provinces.predict(operational=True, verbose=True)
```
(ทำบน branch นี้ตอนนี้ไม่ได้ เพราะ predict_provinces เวอร์ชัน branch นี้ยังไม่มี param `operational`
— เป็นของ feat/operational-province-mode. เจ้าของ merge ต้อง flip บรรทัดนี้ + รัน parity).

### M2 (gap — ความเสี่ยงต่ำ): cron forecast.yml ไม่ผ่าน gate
`.github/workflows/forecast.yml` publish **regional** `docs/forecast.json` อัตโนมัติ (predict.py operational
→ issue_date สด) ด้วย inline check ของตัวเอง (เช็คแค่ 5 lead + prob range) **ไม่ผ่าน readiness gate**.
ความเสี่ยงต่ำ (operational = สด, regional ไม่ใช่ per-province ที่เคยหลุด) แต่ควร: ทำ gate รองรับ schema
regional (top-level `forecasts` ไม่มี `provinces`) แล้วให้ cron เรียกก่อน commit. = งานต่อยอด.
หมายเหตุ: per-province contract (ตัวที่เคยหลุด demo) **ไม่มี automation** — publish ผ่าน manual
`publish_bridge --publish` ที่ gate แล้ว ✓.

### M3 (แก้แล้ว ✅): freshness redesign — วัด gen→issue gap แทน today→issue
**เดิม** วัด `วันนี้ - issue_date ≤ 10 วัน` → บล็อกพยากรณ์สดของ Phase 1 ผิดๆ (issue 2026-05-31, gen วันนี้
= เก่า 16 วันเทียบวันนี้ แต่เป็นพยากรณ์ที่ถูกต้อง! ERA5 ล่าช้า). **แก้เป็น** `check_data_lag` วัด
`generated_at - issue_date ≤ 30 วัน` (ข้อมูลล้าหลังตอนสร้างแค่ไหน) — ตัวแยก demo(gap ~898) จากสด(gap 16).
+ `check_generated_recent` (WARN) วัด today→generated_at ≤ 14. **ยืนยัน integration:** audit กับพยากรณ์สด
Phase 1 จริง → ✅ GO (data_lag 16 ผ่าน, all-High 4% สมจริง). ค่าอยู่ `checks.py:MAX_DATA_LAG_DAYS`/`MAX_GENERATED_AGE_DAYS`.

### M4 (merge): publish_bridge.py แก้คนละจุดทั้งสอง branch → จะ conflict
ทั้งสอง branch แก้ `scripts/publish_bridge.py`: branch operational เพิ่ม arg `operational` (publish operational
by default) ที่ predict call ; branch นี้เพิ่ม readiness gate ใน `validate_file`. merge ต้องรวมทั้งคู่
(คนละบรรทัด ไม่ขัดเชิงตรรกะ — predict operational แล้ว validate ผ่าน gate). M1 เดิม (deadlock) = **หมดแล้ว**
เพราะ branch operational flip เป็น operational-by-default ไปแล้ว (commit 1eaba23).
