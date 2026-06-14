# Pipeline Bridge Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a reproducible, validated pipeline that takes DeepSeek's real per-province forecast and delivers it to the HeatMAP_Frontend — locally (frontend `public/`) and in production (public contract repo → GitHub Pages → Vercel frontend).

**Architecture:** Producer-owned orchestrator in `DeepSeek_Heatwave/scripts/` runs `predict → validate (hard gate) → distribute` to configurable sinks. Validation is a separate importable module so it is unit-testable. Frontend gains a schema-version guard. Prod publish uses a dedicated public `heatwave-contract` repo (DeepSeek stays private/local).

**Tech Stack:** Python 3.12 (stdlib only for the bridge: json/shutil/subprocess/argparse), pytest; TypeScript/Vitest (frontend); `gh` CLI + GitHub Pages; Vercel.

**Spec:** `docs/superpowers/specs/2026-06-14-pipeline-bridge-design.md`

**Conventions observed:**
- DeepSeek tests use `sys.path.insert(0, ...)` then bare import (no conftest). Run with `python -m pytest <file> -v`.
- `predict_provinces.predict(verbose=True)` regenerates `docs/forecast_provinces.json` and returns the dict. `risk_level(p, base_rate)` lives in `scripts/predict.py`.
- Frontend adapter `services/deepseekContract.ts`; tests `services/deepseekContract.test.ts` (vitest, `bun run test:unit`).

---

## Task 1: Contract validator (the gate)

**Files:**
- Create: `scripts/validate_contract.py`
- Test: `scripts/test_validate_contract.py`

- [ ] **Step 1: Write the failing test**

Create `scripts/test_validate_contract.py`:

```python
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import validate_contract as vc


def _good_province(pid=1):
    return {
        "id": pid, "code": "BKK", "name_th": "กรุงเทพมหานคร", "name_en": "Bangkok",
        "region": "Central", "lat": 13.75, "lon": 100.5, "issue_date": "2023-12-31",
        "forecasts": [
            {"lead_weeks": L, "probability": 0.3, "climatology_base_rate": 0.11,
             "ratio_vs_normal": 2.7, "risk_level_th": "สูง", "risk_level_en": "Elevated"}
            for L in (2, 3, 4, 5, 6)
        ],
    }


def _good_contract(n=77):
    return {
        "schema_version": 1, "model": "logistic_balanced_cal",
        "generated_at": "2026-06-14T10:00:00+00:00",
        "n_provinces": n, "provinces": [_good_province(i + 1) for i in range(n)],
    }


def test_good_contract_passes():
    assert vc.validate_contract(_good_contract()) == []


def test_wrong_schema_version():
    c = _good_contract(); c["schema_version"] = 2
    assert any("schema_version" in e for e in vc.validate_contract(c))


def test_n_provinces_mismatch():
    c = _good_contract(); c["n_provinces"] = 70
    assert any("n_provinces" in e for e in vc.validate_contract(c))


def test_wrong_count():
    assert any("77" in e for e in vc.validate_contract(_good_contract(n=76)))


def test_missing_lead():
    c = _good_contract(); c["provinces"][0]["forecasts"].pop()  # remove lead 6
    assert any("leads" in e for e in vc.validate_contract(c))


def test_probability_out_of_range():
    c = _good_contract(); c["provinces"][0]["forecasts"][0]["probability"] = 1.5
    assert any("probability" in e for e in vc.validate_contract(c))


def test_bad_risk_level():
    c = _good_contract(); c["provinces"][0]["forecasts"][0]["risk_level_en"] = "Extreme"
    assert any("risk_level_en" in e for e in vc.validate_contract(c))


def test_lat_out_of_range():
    c = _good_contract(); c["provinces"][0]["lat"] = 999
    assert any("lat" in e for e in vc.validate_contract(c))


def test_duplicate_id():
    c = _good_contract(); c["provinces"][1]["id"] = 1
    assert any("ซ้ำ" in e for e in vc.validate_contract(c))


def test_staleness_warns_when_old():
    c = _good_contract(); c["generated_at"] = "2020-01-01T00:00:00+00:00"
    assert vc.check_staleness(c) is not None


def test_staleness_quiet_when_fresh():
    from datetime import datetime, timezone
    c = _good_contract(); c["generated_at"] = datetime.now(timezone.utc).isoformat()
    assert vc.check_staleness(c) is None
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest scripts/test_validate_contract.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'validate_contract'`

- [ ] **Step 3: Write minimal implementation**

Create `scripts/validate_contract.py`:

```python
"""ตรวจ contract forecast_provinces.json ก่อน distribute (hard gate)."""
from __future__ import annotations

import json
import sys
from datetime import date, datetime, timezone
from pathlib import Path

ALLOWED_RISK_EN = {"Low", "Normal", "Elevated", "High"}
EXPECTED_LEADS = {2, 3, 4, 5, 6}
EXPECTED_N = 77
PROVINCE_KEYS = ("id", "code", "name_th", "name_en", "region",
                 "lat", "lon", "issue_date", "forecasts")


def _is_num(x) -> bool:
    return isinstance(x, (int, float)) and not isinstance(x, bool)


def validate_contract(obj) -> list[str]:
    errs: list[str] = []
    if not isinstance(obj, dict):
        return ["contract ต้องเป็น object/dict"]
    if obj.get("schema_version") != 1:
        errs.append(f"schema_version ต้อง == 1 (เจอ {obj.get('schema_version')!r})")
    model = obj.get("model")
    if not isinstance(model, str) or not model.strip():
        errs.append("model ต้องเป็น str ไม่ว่าง")
    ga = obj.get("generated_at")
    if not isinstance(ga, str):
        errs.append("generated_at ต้องเป็น str (ISO datetime)")
    else:
        try:
            datetime.fromisoformat(ga)
        except ValueError:
            errs.append(f"generated_at parse ไม่ได้: {ga!r}")

    provinces = obj.get("provinces")
    if not isinstance(provinces, list):
        errs.append("provinces ต้องเป็น list")
        return errs
    if obj.get("n_provinces") != len(provinces):
        errs.append(f"n_provinces ({obj.get('n_provinces')}) != len(provinces) ({len(provinces)})")
    if len(provinces) != EXPECTED_N:
        errs.append(f"จำนวนจังหวัดต้อง == {EXPECTED_N} (เจอ {len(provinces)})")

    seen_ids: set = set()
    for i, p in enumerate(provinces):
        tag = f"province[{i}]"
        if not isinstance(p, dict):
            errs.append(f"{tag} ต้องเป็น object")
            continue
        for k in PROVINCE_KEYS:
            if k not in p:
                errs.append(f"{tag} ขาด key '{k}'")
        pid = p.get("id")
        if isinstance(pid, int) and not isinstance(pid, bool):
            if pid in seen_ids:
                errs.append(f"{tag} id ซ้ำ: {pid}")
            seen_ids.add(pid)
            tag = f"province id={pid}"
        else:
            errs.append(f"{tag} id ต้องเป็น int")
        lat, lon = p.get("lat"), p.get("lon")
        if not (_is_num(lat) and -90 <= lat <= 90):
            errs.append(f"{tag} lat นอกช่วง [-90,90]: {lat!r}")
        if not (_is_num(lon) and -180 <= lon <= 180):
            errs.append(f"{tag} lon นอกช่วง [-180,180]: {lon!r}")
        try:
            date.fromisoformat(p.get("issue_date"))
        except (ValueError, TypeError):
            errs.append(f"{tag} issue_date parse ไม่ได้: {p.get('issue_date')!r}")

        fcs = p.get("forecasts")
        if not isinstance(fcs, list):
            errs.append(f"{tag} forecasts ต้องเป็น list")
            continue
        leads = []
        for f in fcs:
            if not isinstance(f, dict):
                errs.append(f"{tag} forecast ต้องเป็น object")
                continue
            L = f.get("lead_weeks")
            leads.append(L)
            prob = f.get("probability")
            if not (_is_num(prob) and 0 <= prob <= 1):
                errs.append(f"{tag} lead {L} probability นอกช่วง [0,1]: {prob!r}")
            br = f.get("climatology_base_rate")
            if not (_is_num(br) and 0 <= br <= 1):
                errs.append(f"{tag} lead {L} climatology_base_rate นอกช่วง [0,1]: {br!r}")
            ratio = f.get("ratio_vs_normal")
            if not (_is_num(ratio) and ratio >= 0):
                errs.append(f"{tag} lead {L} ratio_vs_normal ต้อง >= 0: {ratio!r}")
            if f.get("risk_level_en") not in ALLOWED_RISK_EN:
                errs.append(f"{tag} lead {L} risk_level_en ไม่ถูก: {f.get('risk_level_en')!r}")
            th = f.get("risk_level_th")
            if not isinstance(th, str) or not th.strip():
                errs.append(f"{tag} lead {L} risk_level_th ว่าง")
        if set(leads) != EXPECTED_LEADS:
            got = sorted(x for x in leads if x is not None)
            errs.append(f"{tag} leads ต้องเป็น {sorted(EXPECTED_LEADS)} (เจอ {got})")
    return errs


def check_staleness(obj, max_age_days: int = 7) -> str | None:
    try:
        ga_dt = datetime.fromisoformat(obj.get("generated_at"))
    except (ValueError, TypeError):
        return None
    if ga_dt.tzinfo is None:
        ga_dt = ga_dt.replace(tzinfo=timezone.utc)
    age = (datetime.now(timezone.utc) - ga_dt).days
    return f"[เตือน] contract เก่า {age} วัน (generated_at {obj.get('generated_at')})" if age > max_age_days else None


def main(argv) -> int:
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass
    path = Path(argv[1]) if len(argv) > 1 else \
        Path(__file__).resolve().parent.parent / "docs" / "forecast_provinces.json"
    obj = json.loads(path.read_text(encoding="utf-8"))
    warn = check_staleness(obj)
    if warn:
        print(warn)
    errs = validate_contract(obj)
    if errs:
        print(f"[FAIL] contract ไม่ผ่าน {len(errs)} ข้อ:")
        for e in errs:
            print(f"  - {e}")
        return 1
    print(f"[OK] contract ผ่าน: {len(obj['provinces'])} จังหวัด, schema v{obj['schema_version']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest scripts/test_validate_contract.py -v`
Expected: PASS (11 passed)

- [ ] **Step 5: Validate the REAL current contract (catch over-strict gate)**

Run: `python scripts/validate_contract.py`
Expected: `[OK] contract ผ่าน: 77 จังหวัด, schema v1` and exit 0.
If it FAILS, the validator is stricter than real output — fix the validator to match the real schema, not the other way around.

- [ ] **Step 6: Commit**

```bash
git add scripts/validate_contract.py scripts/test_validate_contract.py
git commit -m "feat: contract validator (hard gate before distribute)"
```

---

## Task 2: Bridge orchestrator (predict → validate → distribute)

**Files:**
- Create: `scripts/publish_bridge.py`
- Test: `scripts/test_publish_bridge.py`

- [ ] **Step 1: Write the failing test**

Create `scripts/test_publish_bridge.py`:

```python
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import publish_bridge as pb
from test_validate_contract import _good_contract


def test_validate_file_ok(tmp_path):
    p = tmp_path / "c.json"
    p.write_text(json.dumps(_good_contract()), encoding="utf-8")
    obj = pb.validate_file(p)
    assert obj["n_provinces"] == 77


def test_validate_file_aborts_on_bad(tmp_path):
    bad = _good_contract(); bad["schema_version"] = 9
    p = tmp_path / "c.json"
    p.write_text(json.dumps(bad), encoding="utf-8")
    try:
        pb.validate_file(p)
        assert False, "ควร raise SystemExit"
    except SystemExit as e:
        assert e.code == 1


def test_sync_copies_bytes(tmp_path):
    src = tmp_path / "src.json"
    src.write_text("hello", encoding="utf-8")
    dst = tmp_path / "nested" / "dst.json"
    pb.sync(src, dst)
    assert dst.read_text(encoding="utf-8") == "hello"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest scripts/test_publish_bridge.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'publish_bridge'`

- [ ] **Step 3: Write minimal implementation**

Create `scripts/publish_bridge.py`:

```python
"""Bridge: predict -> validate (gate) -> distribute ไป dev/prod sink."""
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))
import validate_contract as vc  # noqa: E402

DOCS_JSON = ROOT / "docs" / "forecast_provinces.json"


def default_frontend() -> Path:
    base = Path(os.environ["BRIDGE_FRONTEND_DIR"]) if os.environ.get("BRIDGE_FRONTEND_DIR") \
        else ROOT.parent / "HeatMAP_Frontend" / "public"
    return base / "forecast_provinces.json"


def default_contract() -> Path:
    base = Path(os.environ["BRIDGE_CONTRACT_DIR"]) if os.environ.get("BRIDGE_CONTRACT_DIR") \
        else ROOT.parent / "heatwave-contract"
    return base / "forecast_provinces.json"


def validate_file(path: Path) -> dict:
    obj = json.loads(path.read_text(encoding="utf-8"))
    warn = vc.check_staleness(obj)
    if warn:
        print(warn)
    errs = vc.validate_contract(obj)
    if errs:
        print(f"[FAIL] validate ไม่ผ่าน {len(errs)} ข้อ — ยกเลิก distribute:")
        for e in errs:
            print(f"  - {e}")
        raise SystemExit(1)
    print(f"[OK] validate ผ่าน: {len(obj['provinces'])} จังหวัด")
    return obj


def sync(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(src, dst)
    print(f"[OK] sync -> {dst}")


def publish_contract(contract_json: Path) -> None:
    repo = contract_json.parent
    if not (repo / ".git").exists():
        print(f"[FAIL] ไม่พบ git repo ที่ {repo} — สร้าง/clone contract repo ก่อน")
        raise SystemExit(1)
    sync(DOCS_JSON, contract_json)
    subprocess.run(["git", "-C", str(repo), "add", "forecast_provinces.json"], check=True)
    unchanged = subprocess.run(["git", "-C", str(repo), "diff", "--cached", "--quiet"]).returncode == 0
    if unchanged:
        print("[ข้าม] contract ไม่เปลี่ยน — ไม่ commit/push")
        return
    subprocess.run(["git", "-C", str(repo), "commit", "-m", "data: update forecast_provinces.json"], check=True)
    subprocess.run(["git", "-C", str(repo), "push"], check=True)
    print("[OK] push contract -> Pages")


def main() -> int:
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass
    ap = argparse.ArgumentParser(description="Pipeline bridge: predict -> validate -> distribute")
    ap.add_argument("--no-predict", action="store_true", help="ข้าม predict, ใช้ docs/forecast_provinces.json เดิม")
    ap.add_argument("--publish", action="store_true", help="sync เข้า contract repo แล้ว git push (prod)")
    ap.add_argument("--frontend", type=Path, default=None, help="path ปลายทาง dev sink (override)")
    args = ap.parse_args()

    if not args.no_predict:
        sys.path.insert(0, str(ROOT / "scripts"))
        import predict_provinces
        predict_provinces.predict(verbose=True)
    elif not DOCS_JSON.exists():
        print(f"[FAIL] ไม่มี {DOCS_JSON} แต่ใช้ --no-predict")
        return 1

    validate_file(DOCS_JSON)
    sync(DOCS_JSON, args.frontend or default_frontend())

    if args.publish:
        publish_contract(default_contract())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest scripts/test_publish_bridge.py -v`
Expected: PASS (3 passed)

- [ ] **Step 5: Run the real dev sync end-to-end**

Run: `python scripts/publish_bridge.py`
Expected output includes `[OK] ...forecast_provinces.json | 77 จังหวัด`, `[OK] validate ผ่าน: 77 จังหวัด`, and `[OK] sync -> ...HeatMAP_Frontend/public/forecast_provinces.json`.

- [ ] **Step 6: Verify dev sink is byte-identical to docs**

Run: `python -c "import filecmp; print(filecmp.cmp(r'docs/forecast_provinces.json', r'../HeatMAP_Frontend/public/forecast_provinces.json', shallow=False))"`
Expected: `True`

- [ ] **Step 7: Verify `--no-predict` does not regenerate**

Run:
```bash
python -c "import json;print(json.load(open('docs/forecast_provinces.json',encoding='utf-8'))['generated_at'])"
python scripts/publish_bridge.py --no-predict
python -c "import json;print(json.load(open('docs/forecast_provinces.json',encoding='utf-8'))['generated_at'])"
```
Expected: the two `generated_at` values are identical (no new prediction run).

- [ ] **Step 8: Commit**

```bash
git add scripts/publish_bridge.py scripts/test_publish_bridge.py
git commit -m "feat: pipeline bridge orchestrator (predict->validate->sync)"
```

---

## Task 3: Frontend schema-version guard

**Files:**
- Modify: `services/deepseekContract.ts` (in `C:\Users\ASUS\HeatMAP_Frontend`)
- Modify: `services/deepseekContract.test.ts`

> All commands in this task run in `C:\Users\ASUS\HeatMAP_Frontend`.

- [ ] **Step 1: Write the failing test**

Append to `services/deepseekContract.test.ts` (add `assertContract` to the import on line 2: `import { mapPoints, provinceDays, RISK_EN_TO_APP, assertContract } from './deepseekContract';`), then add before the final closing `});` of the describe block:

```ts
  it('assertContract: passes schema_version 1', () => {
    expect(assertContract(sample as any).schema_version).toBe(1);
  });
  it('assertContract: throws on unsupported schema_version', () => {
    expect(() => assertContract({ ...sample, schema_version: 2 } as any)).toThrow(/schema_version/);
  });
```

- [ ] **Step 2: Run test to verify it fails**

Run: `bun run test:unit`
Expected: FAIL — `assertContract is not a function` / import has no exported member `assertContract`.

- [ ] **Step 3: Write minimal implementation**

In `services/deepseekContract.ts`, add after the `Contract` interface (line 9):

```ts
export const SUPPORTED_SCHEMA = 1;
export function assertContract(c: Contract): Contract {
  if (c.schema_version !== SUPPORTED_SCHEMA) {
    throw new Error(`schema_version ไม่รองรับ: ${c.schema_version} (รองรับ ${SUPPORTED_SCHEMA})`);
  }
  return c;
}
```

Then wire it into `loadContract` — change the `.then((r) => {...})` body so the parsed JSON passes through `assertContract`:

```ts
    _cache = fetch(FORECAST_URL).then(async (r) => {
      if (!r.ok) throw new Error(`โหลด forecast ไม่สำเร็จ (${r.status})`);
      return assertContract((await r.json()) as Contract);
    }).catch((e) => { _cache = null; throw e; });
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `bun run test:unit`
Expected: PASS (all deepseekContract tests, including the 2 new ones).

- [ ] **Step 5: Verify the web build still compiles**

Run: `bunx expo export -p web`
Expected: completes without error; `dist/` produced.

- [ ] **Step 6: Commit**

```bash
git add services/deepseekContract.ts services/deepseekContract.test.ts
git commit -m "feat: guard unsupported forecast schema_version in adapter"
```

---

## Task 4: Local bridge verified in browser (dev end-to-end)

**Files:** none (verification + the stale-date go/no-go).

> Runs in `C:\Users\ASUS\HeatMAP_Frontend`. This proves the LOCAL half of the bridge before any prod work.

- [ ] **Step 1: Ensure fresh synced data**

In `C:\Users\ASUS\DeepSeek_Heatwave`: `python scripts/publish_bridge.py` (regenerate + validate + sync to frontend `public/`).

- [ ] **Step 2: Build and serve the frontend statically**

```bash
cd /c/Users/ASUS/HeatMAP_Frontend
bunx expo export -p web
bunx serve dist -l 5055
```
(Leave serving; or use the existing port from prior QA.)

- [ ] **Step 3: Browser QA via chrome-devtools MCP**

- `navigate_page` → `http://localhost:5055`
- `list_console_messages` → expect ZERO errors.
- `list_network_requests` → confirm `forecast_provinces.json` returned 200.
- `take_screenshot` → confirm 77-province map renders with colors; tap a province → weekly 2–6wk panel.

- [ ] **Step 4: STALE-DATE GO/NO-GO (user decision — do NOT skip)**

In the province panel and a map card, read the rendered `target_date`s — they will be **Jan–Feb 2024** (issue_date 2023-12-31 + lead×7). Report exactly what renders to the user and ask them to choose ONE before any public deploy:
- (a) Ship as-is (tech demo)
- (b) Add a "historical model run" banner near the date
- (c) Switch UI to relative "+2…+6 สัปดาห์" labels and hide absolute dates

If (b) or (c): stop and brainstorm that UI change as a small follow-up before continuing to Task 5–6. If (a): proceed.

- [ ] **Step 5: Commit (only if any verification artifact/doc was added)**

No code change expected here. If QA notes are worth keeping, add them to `README.md` in Task 7 instead.

---

## Task 5: Production contract repo + GitHub Pages (OUTWARD-FACING — confirm with user)

**Files:**
- Create new repo at `C:\Users\ASUS\heatwave-contract`: `forecast_provinces.json`, `CONTRACT.md`, `README.md`, `.gitignore`

> Creates a PUBLIC GitHub repo and publishes the contract JSON. Confirm with the user before pushing. `gh` is authed as `MCTEEKUNG`.

- [ ] **Step 1: Scaffold the repo locally**

```bash
mkdir -p /c/Users/ASUS/heatwave-contract
cd /c/Users/ASUS/heatwave-contract
git init -b main
cp /c/Users/ASUS/DeepSeek_Heatwave/docs/forecast_provinces.json ./forecast_provinces.json
printf "node_modules/\n.DS_Store\n" > .gitignore
```

- [ ] **Step 2: Write `CONTRACT.md`**

```markdown
# Heatwave Forecast Contract (schema_version 1)

Static JSON published for the HeatMAP frontend. No API server.

`forecast_provinces.json`:
- `schema_version` (int) = 1
- `model` (str), `generated_at` (ISO datetime, UTC), `n_provinces` (int, = 77)
- `provinces[]`: `id`, `code`, `name_th`, `name_en`, `region`, `lat`, `lon`, `issue_date` (YYYY-MM-DD)
  - `forecasts[]` (one per lead 2,3,4,5,6 weeks): `lead_weeks`, `probability` (0–1),
    `climatology_base_rate` (0–1), `ratio_vs_normal` (≥0), `risk_level_th`, `risk_level_en`
- Risk levels (`risk_level_en`): `Low | Normal | Elevated | High`
  - app mapping: Low→low, Normal→moderate, Elevated→high, High→extreme

Consumers MUST reject any object whose `schema_version` != 1.
Produced by `DeepSeek_Heatwave/scripts/publish_bridge.py --publish`.
```

- [ ] **Step 3: Write `README.md`**

```markdown
# heatwave-contract

Published forecast contract for the Thailand sub-seasonal heatwave app.
Served via GitHub Pages: `https://mcteekung.github.io/heatwave-contract/forecast_provinces.json`
See `CONTRACT.md` for the schema. Data is regenerated/validated/pushed by the DeepSeek bridge.
```

- [ ] **Step 4: Commit and create the public repo**

```bash
git add .
git commit -m "init: forecast contract (schema v1) + Pages"
gh repo create heatwave-contract --public --source=. --remote=origin --push
```
Expected: repo created under `MCTEEKUNG`, `main` pushed.

- [ ] **Step 5: Enable GitHub Pages (branch main, root)**

```bash
gh api -X POST repos/MCTEEKUNG/heatwave-contract/pages -f "source[branch]=main" -f "source[path]=/" || \
gh api -X PUT repos/MCTEEKUNG/heatwave-contract/pages -f "source[branch]=main" -f "source[path]=/"
```
(If the API path errors, enable Pages in the repo Settings → Pages → Source: `main` / root.)

- [ ] **Step 6: Verify the published URL is live with CORS**

Wait ~1–2 min for first publish, then:
```bash
curl -I https://mcteekung.github.io/heatwave-contract/forecast_provinces.json
```
Expected: `HTTP/2 200` and header `access-control-allow-origin: *`. Re-run until 200 (Pages first build can lag).

- [ ] **Step 7: Confirm `--publish` round-trips through the bridge**

In `C:\Users\ASUS\DeepSeek_Heatwave`:
```bash
python scripts/publish_bridge.py --no-predict --publish
```
Expected: `[ข้าม] contract ไม่เปลี่ยน — ไม่ commit/push` (data already current) OR a commit+push log. No errors.

---

## Task 6: Frontend prod config + Vercel deploy (OUTWARD-FACING — confirm with user)

**Files:**
- Modify: `.env.example` (HeatMAP_Frontend)
- Create: `.env.production` (HeatMAP_Frontend) — git-ignored if it contains anything secret; here it is just a public URL, safe to commit

> Deploys the static site publicly. Confirm with the user. Gated by Task 4 Step 4 (stale-date go/no-go) being resolved.

- [ ] **Step 1: Point prod at the Pages URL**

Ensure `.env.example` documents:
```
EXPO_PUBLIC_FORECAST_URL=https://mcteekung.github.io/heatwave-contract/forecast_provinces.json
```
Create `.env.production` with the same line.

- [ ] **Step 2: Build prod bundle with the env var**

```bash
cd /c/Users/ASUS/HeatMAP_Frontend
EXPO_PUBLIC_FORECAST_URL=https://mcteekung.github.io/heatwave-contract/forecast_provinces.json bunx expo export -p web
```
Expected: `dist/` built. The fetch URL is inlined at build time (Expo `EXPO_PUBLIC_*`).

- [ ] **Step 3: Sanity-check the bundle references the prod URL**

Run: `grep -rl "mcteekung.github.io/heatwave-contract" dist/ | head`
Expected: at least one bundle file matches.

- [ ] **Step 4: Deploy `dist/` to Vercel**

Use the deploy-to-vercel skill (or `vercel deploy --prebuilt`/static). Deploy `dist/` as a static site.

- [ ] **Step 5: Verify the deployed site uses the Pages contract**

Via chrome-devtools MCP on the Vercel URL:
- `list_network_requests` → confirm a request to `https://mcteekung.github.io/heatwave-contract/forecast_provinces.json` returned 200 (cross-origin OK).
- `list_console_messages` → ZERO errors.
- `take_screenshot` → 77-province map renders.

---

## Task 7: Documentation

**Files:**
- Modify: `docs/RUNBOOK.md` (DeepSeek_Heatwave)
- Modify: `README.md` (HeatMAP_Frontend)

- [ ] **Step 1: Add bridge section to DeepSeek `docs/RUNBOOK.md`**

Append:
```markdown
## Pipeline bridge (forecast → frontend)

- Dev sync (regenerate + validate + copy to frontend public/):
  `python scripts/publish_bridge.py`
- Validate only the current contract: `python scripts/validate_contract.py`
- Use existing forecast (no re-predict), still validate + sync: `python scripts/publish_bridge.py --no-predict`
- Publish to prod (also push contract repo → Pages): `python scripts/publish_bridge.py --publish`
- Sink overrides: env `BRIDGE_FRONTEND_DIR`, `BRIDGE_CONTRACT_DIR`.
- Validation is a HARD gate — distribute aborts if the contract is invalid.
```

- [ ] **Step 2: Add data-source section to frontend `README.md`**

Append:
```markdown
## Forecast data source (bridge)

The app reads a static contract `forecast_provinces.json` (schema v1; see `services/deepseekContract.ts`).
- Dev: same-origin `public/forecast_provinces.json` (synced by the DeepSeek bridge `publish_bridge.py`).
- Prod: set `EXPO_PUBLIC_FORECAST_URL` to the published Pages URL
  (`https://mcteekung.github.io/heatwave-contract/forecast_provinces.json`) before `bunx expo export -p web`.
Unsupported `schema_version` is rejected at load (`assertContract`).
```

- [ ] **Step 3: Commit (DeepSeek)**

```bash
cd /c/Users/ASUS/DeepSeek_Heatwave
git add docs/RUNBOOK.md
git commit -m "docs: document pipeline bridge commands"
```

- [ ] **Step 4: Commit (frontend)**

```bash
cd /c/Users/ASUS/HeatMAP_Frontend
git add README.md .env.example .env.production
git commit -m "docs: document forecast bridge data source + prod URL"
```

---

## Done criteria
- `python -m pytest scripts/test_validate_contract.py scripts/test_publish_bridge.py -v` → all pass.
- `python scripts/publish_bridge.py` regenerates, validates, and syncs to the frontend `public/` (byte-identical).
- Frontend `bun run test:unit` passes incl. the schema guard; local browser QA clean.
- (Prod, if user proceeds) Pages URL serves the contract with `access-control-allow-origin: *`; Vercel site fetches it cross-origin with zero console errors.
- Stale-date go/no-go resolved by the user before public deploy.
```
