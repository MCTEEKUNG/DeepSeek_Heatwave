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
