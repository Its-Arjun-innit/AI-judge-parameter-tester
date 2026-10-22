from judgelab import adapters, runner
from judgelab.adapters import ProviderError
from judgelab.cases import ROOT, load_case, render
from judgelab.db import connect
from judgelab.schema import response_schema

SPEC = ROOT / "experiments" / "ping-01.yaml"


def test_lean_prompt_v2_keeps_all_measures():
    case = load_case(ROOT / "cases" / "IN-THEFT-001.yaml")
    v1 = render(case, [("E01", "base"), ("E06", "base")], "v1")
    v2 = render(case, [("E01", "base"), ("E06", "base")], "v2")
    assert len(v2.system) < len(v1.system) and "PW-1:" in v2.user and "DW-1:" in v2.user
    for field in response_schema()["properties"]:
        assert field in v2.system  # every response field is still requested
    assert len(response_schema()["properties"]) == 7


def test_quota_errors_leave_trials_pending_and_block_model(tmp_path, monkeypatch):
    monkeypatch.setattr(runner.time, "sleep", lambda s: None)  # skip pacing and backoff waits
    con = connect(tmp_path / "t.sqlite")
    runner.plan(con, SPEC)

    def no_credit(cfg, prompt, schema):
        raise ProviderError("RateLimitError: 429 insufficient_quota", status=429, retryable=True)
    monkeypatch.setitem(adapters._CALL, "fake", no_credit)
    out = runner.run(con, "ping-01", dry_run=True, concurrency=1)
    assert out["done"] == 0 and out["paused_models"] == ["claude", "gemini", "gpt"]
    assert {r[0] for r in con.execute("SELECT status FROM trials")} == {"pending"}
    # billing errors are not retried: exactly one logged attempt per trial
    assert [r[0] for r in con.execute("SELECT COUNT(*) FROM attempts GROUP BY trial_id")] == [1, 1, 1]

    monkeypatch.setitem(adapters._CALL, "fake", adapters._fake)  # credit restored: resume completes
    assert runner.run(con, "ping-01", dry_run=True, concurrency=1)["done"] == 3
    assert runner.run(con, "ping-01", dry_run=True, concurrency=1)["executed"] == 0
