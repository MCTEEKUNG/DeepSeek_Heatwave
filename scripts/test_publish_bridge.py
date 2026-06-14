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
