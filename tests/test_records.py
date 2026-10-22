"""Permanent records: the database refuses deletion/rewrites; exports and publishing keep a copy."""
import hashlib
import json
import sqlite3
import subprocess

import pytest

from judgelab import records, runner
from judgelab.cases import ROOT
from judgelab.db import connect

SPEC = ROOT / "experiments" / "ping-01.yaml"


@pytest.fixture
def con(tmp_path, monkeypatch):
    monkeypatch.setattr(runner.time, "sleep", lambda s: None)
    c = connect(tmp_path / "r.sqlite")
    runner.plan(c, SPEC)
    runner.run(c, "ping-01", dry_run=True, concurrency=1)
    return c


def test_records_cannot_be_deleted_or_rewritten(con):
    for sql in ("DELETE FROM attempts", "DELETE FROM trials", "DELETE FROM experiments", "DELETE FROM cases",
                "UPDATE attempts SET probability = 99", "UPDATE trials SET user_prompt = 'edited'",
                "UPDATE experiments SET spec_yaml = 'x'", "UPDATE cases SET yaml = '{}'", "DELETE FROM runs"):
        with pytest.raises(sqlite3.DatabaseError, match="permanent"):
            con.execute(sql)


def test_completed_analysis_can_only_be_excluded_with_a_reason(con):
    tid = con.execute("SELECT trial_id FROM trials WHERE status='done' LIMIT 1").fetchone()[0]
    with pytest.raises(sqlite3.DatabaseError, match="permanent"):
        con.execute("UPDATE trials SET status='pending' WHERE trial_id=?", (tid,))
    with pytest.raises(sqlite3.DatabaseError, match="permanent"):
        con.execute("UPDATE trials SET status='excluded' WHERE trial_id=?", (tid,))
    con.execute("UPDATE trials SET status='excluded', exclusion_reason='protocol deviation' WHERE trial_id=?", (tid,))
    assert con.execute("SELECT status FROM trials WHERE trial_id=?", (tid,)).fetchone()[0] == "excluded"


def test_run_sessions_and_instruction_texts_are_recorded(con):
    run = con.execute("SELECT finished_at, options_json, environment_json, summary_json FROM runs").fetchone()
    assert run[0] and json.loads(run[1])["dry_run"] is True
    assert json.loads(run[2])["packages"]["pandas"] and json.loads(run[3])["done"] == 3
    names = {(r[0], r[1]) for r in con.execute("SELECT name, version FROM prompts")}
    assert names == {("system", "v2"), ("user", "v2")}


def test_export_writes_checksummed_files_and_redacts_keys(con, tmp_path):
    tid = con.execute("SELECT trial_id FROM trials LIMIT 1").fetchone()[0]
    fake_key = "sk-" + "proj-" + "ABCDEFGHIJKLMNOPQRSTUV"  # assembled so the repo's key scanner never sees a key
    con.execute("INSERT INTO attempts (trial_id, attempt_no, started_at, finished_at, error, parse_status) "
                "VALUES (?, 9, 'now', 'now', ?, 'api_error')", (tid, f"bad key {fake_key}"))
    out = records.export(con, "ping-01", tmp_path / "records")
    names = {p.relative_to(out).as_posix() for p in out.rglob("*") if p.is_file()}
    assert {"experiment.yaml", "trials.csv", "attempts.csv", "observations.csv", "runs.csv", "conditions.csv",
            "manifest.json"} <= names
    assert any(n.startswith("cases/IN-THEFT-001-") for n in names)
    assert {n.split("-")[0] for n in names if n.startswith("prompts/")} == {"prompts/system_v2", "prompts/user_v2"}
    attempts = (out / "attempts.csv").read_text(encoding="utf-8")
    assert "[REDACTED-KEY]" in attempts and "sk-proj-ABC" not in attempts
    manifest = json.loads((out / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["counts"]["attempts"] == 4 and not manifest["system_prompt_text_missing_for_hashes"]
    for name, digest in manifest["sha256"].items():
        assert hashlib.sha256((out / name).read_bytes()).hexdigest() == digest


def test_publish_commits_only_the_records_folder(tmp_path, monkeypatch):
    git = lambda *a: subprocess.run(["git", *a], cwd=tmp_path, capture_output=True, text=True)  # noqa: E731
    git("init", "-q")
    git("config", "user.name", "Test")
    git("config", "user.email", "test@example.com")
    (tmp_path / "records" / "exp-1").mkdir(parents=True)
    (tmp_path / "records" / "exp-1" / "trials.csv").write_text("a,b\n1,2\n")
    (tmp_path / "unrelated.txt").write_text("work in progress")
    git("add", "unrelated.txt")  # staged, but must not be swept into the records commit
    monkeypatch.setattr(records, "ROOT", tmp_path)
    monkeypatch.setattr(records, "RECORDS_DIR", tmp_path / "records")
    msg = records.publish(["exp-1"])
    assert "committed" in msg and "push failed" in msg  # no remote in this test repo
    assert git("show", "--name-only", "--format=", "HEAD").stdout.split() == ["records/exp-1/trials.csv"]
    assert records.publish(["exp-1"]) == "records already up to date in git"
    monkeypatch.setenv("JUDGELAB_PUBLISH_RECORDS", "0")
    assert "turned off" in records.publish(["exp-1"])
