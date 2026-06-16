import json
import subprocess
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


def test_validate_file_missing(tmp_path):
    try:
        pb.validate_file(tmp_path / "nope.json")
        assert False, "ควร raise SystemExit"
    except SystemExit as e:
        assert e.code == 1


def test_default_frontend_env_override(monkeypatch, tmp_path):
    monkeypatch.setenv("BRIDGE_FRONTEND_DIR", str(tmp_path))
    assert pb.default_frontend() == tmp_path / "forecast_provinces.json"


def test_default_frontend_fallback(monkeypatch):
    monkeypatch.delenv("BRIDGE_FRONTEND_DIR", raising=False)
    p = pb.default_frontend()
    assert p.name == "forecast_provinces.json"
    assert "HeatMAP_Frontend" in str(p)


def test_default_contract_env_override(monkeypatch, tmp_path):
    monkeypatch.setenv("BRIDGE_CONTRACT_DIR", str(tmp_path))
    assert pb.default_contract() == tmp_path / "forecast_provinces.json"


def test_publish_contract_aborts_without_git(tmp_path):
    try:
        pb.publish_contract(tmp_path / "forecast_provinces.json")
        assert False, "ควร raise SystemExit"
    except SystemExit as e:
        assert e.code == 1


def test_has_unpushed_no_commits_then_after_commit(tmp_path):
    repo = tmp_path / "repo"
    repo.mkdir()
    subprocess.run(["git", "-C", str(repo), "init", "-q"], check=True)
    subprocess.run(["git", "-C", str(repo), "config", "user.email", "t@t"], check=True)
    subprocess.run(["git", "-C", str(repo), "config", "user.name", "t"], check=True)
    assert pb._has_unpushed(repo) is False  # ยังไม่มี commit
    (repo / "f.txt").write_text("x", encoding="utf-8")
    subprocess.run(["git", "-C", str(repo), "add", "f.txt"], check=True)
    subprocess.run(["git", "-C", str(repo), "commit", "-q", "-m", "c"], check=True)
    assert pb._has_unpushed(repo) is True  # มี commit แต่ยังไม่มี upstream


def test_dest_issue_date_reads_and_missing(tmp_path):
    import json as _json
    p = tmp_path / "c.json"
    p.write_text(_json.dumps({"provinces": [{"issue_date": "2026-05-31"}]}), encoding="utf-8")
    assert pb._dest_issue_date(p) == "2026-05-31"
    assert pb._dest_issue_date(tmp_path / "missing.json") is None


def test_staleness_guard_blocks_older_over_newer(tmp_path, monkeypatch, capsys):
    """ถ้าปลายทางมี issue_date ใหม่กว่า contract ใหม่ -> main() ต้อง return 1 (กันของเก่าทับของใหม่)."""
    import json as _json
    # docs (ของใหม่ที่จะ publish) = เก่ากว่าปลายทาง
    docs = tmp_path / "docs" / "forecast_provinces.json"
    docs.parent.mkdir(parents=True)
    old_contract = {"schema_version": 1, "model": "m", "generated_at": "2023-12-31T00:00:00Z",
                    "n_provinces": 1, "provinces": [{"issue_date": "2023-12-31"}]}
    docs.write_text(_json.dumps(old_contract), encoding="utf-8")
    front = tmp_path / "front" / "forecast_provinces.json"
    front.parent.mkdir(parents=True)
    front.write_text(_json.dumps({"provinces": [{"issue_date": "2026-05-31"}]}), encoding="utf-8")
    monkeypatch.setattr(pb, "DOCS_JSON", docs)
    # validate ผ่าน (อย่าให้ validate บล็อกก่อนถึง guard) — stub validate_file ให้คืน obj ของ docs
    monkeypatch.setattr(pb, "validate_file", lambda path: _json.loads(docs.read_text(encoding="utf-8")))
    monkeypatch.setattr(sys, "argv", ["publish_bridge.py", "--no-predict", "--frontend", str(front)])
    rc = pb.main()
    assert rc == 1
    assert "issue_date ใหม่กว่า" in capsys.readouterr().out
