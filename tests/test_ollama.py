"""Ollama provider against a local fake Ollama server (stdlib http.server; no network, no cloud)."""
import json
import socket
import sqlite3
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest

from judgelab import adapters, runner
from judgelab.adapters import ProviderError, call_model
from judgelab.cases import ROOT, load_case, render
from judgelab.db import DDL, connect
from judgelab.parse import parse
from judgelab.schema import ModelConfig, response_schema

GOOD = {"reasoning": "r", "legal_standard_applied": "BRD", "key_evidence": [{"label": "PW-1", "bearing": "supports_conviction"}],
        "missing_information": [], "probability_committed": 60, "verdict": "acquit", "confidence": 4}


class FakeOllama(BaseHTTPRequestHandler):
    models = ["qwen3:4b-instruct"]
    mode = "valid"          # valid | malformed | length | overflow | slow
    received = []

    def log_message(self, *a):
        pass

    def _send(self, code, obj):
        body = json.dumps(obj).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        if self.path == "/api/version":
            return self._send(200, {"version": "9.9.9"})
        if self.path == "/api/tags":
            return self._send(200, {"models": [{"name": n, "model": n, "size": 2_500_000_000, "digest": "abcdef1234567890",
                                                "details": {"parameter_size": "4.0B", "quantization_level": "Q4_K_M",
                                                            "family": "qwen3"}} for n in self.models]})
        self._send(404, {"error": "not found"})

    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        if self.path == "/api/show":
            return self._send(200, {"capabilities": ["completion"], "model_info": {"qwen3.context_length": 40960}})
        FakeOllama.received.append(body)
        if body["model"] not in self.models:
            return self._send(404, {"error": f"model '{body['model']}' not found"})
        if self.mode == "slow":
            threading.Event().wait(2)
        content = {"malformed": "Sure! The accused is guilty.", "length": '{"reasoning": "cut'}.get(self.mode, json.dumps(GOOD))
        self._send(200, {"model": body["model"], "message": {"role": "assistant", "content": content}, "done": True,
                         "done_reason": "length" if self.mode == "length" else "stop",
                         "prompt_eval_count": 990 if self.mode == "overflow" else 900, "eval_count": 120,
                         "total_duration": 1_000_000})


@pytest.fixture
def ollama(monkeypatch):
    FakeOllama.models, FakeOllama.mode, FakeOllama.received = ["qwen3:4b-instruct"], "valid", []
    srv = ThreadingHTTPServer(("127.0.0.1", 0), FakeOllama)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    monkeypatch.setenv("OLLAMA_HOST", f"127.0.0.1:{srv.server_address[1]}")
    for p in ("anthropic", "openai", "gemini"):  # any cloud call fails the test: no silent fallback
        monkeypatch.setitem(adapters._CALL, p, lambda *a: pytest.fail("unexpected cloud request"))
    monkeypatch.setattr(runner.time, "sleep", lambda s: None)  # skip retry backoff
    yield FakeOllama
    srv.shutdown()


def spec(tmp_path, runs=3, conditions=("baseline", "add_bank"), model="qwen3:4b-instruct", **cfg):
    ops = {"baseline": "{id: baseline}", "add_bank": "{id: add_bank, factor: presence, level: added, ops: [{add: E03}]}"}
    m = {"key": "local", "provider": "ollama", "model": model, "num_ctx": 4096, "max_tokens": 600} | cfg
    p = tmp_path / "spec.yaml"
    p.write_text(f"experiment_id: local-test\ncases: [IN-THEFT-001]\nprompt_version: v2\nmodels:\n  - {json.dumps(m)}\n"
                 f"runs: {runs}\nseed: 5\nbudget: {{max_usd: 0, max_calls: 100}}\nconditions:\n"  # $0: local is free
                 + "".join(f"  - {ops[c]}\n" for c in conditions), encoding="utf-8")
    return p


def planned(tmp_path, name="a.sqlite", **kw):
    con = connect(tmp_path / name)
    runner.plan(con, spec(tmp_path, **kw))
    return con


def test_status_models_and_show(ollama):
    assert adapters.ollama_status() == (True, "9.9.9")
    m = adapters.ollama_models()[0]
    assert m["name"] == "qwen3:4b-instruct" and m["size_gb"] == 2.5 and m["parameters"] == "4.0B"
    assert adapters.ollama_show("qwen3:4b-instruct")["context_length"] == 40960


def test_service_down_is_clear_and_runs_nothing(tmp_path, monkeypatch):
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    port = s.getsockname()[1]
    s.close()  # nothing listens here now
    monkeypatch.setenv("OLLAMA_HOST", f"127.0.0.1:{port}")
    ok, msg = adapters.ollama_status()
    assert not ok and "not running" in msg
    con = planned(tmp_path)
    with pytest.raises(ProviderError, match="not running"):
        runner.run(con, "local-test")
    assert con.execute("SELECT COUNT(*) FROM attempts").fetchone()[0] == 0


def test_missing_model_stops_before_any_analysis(ollama, tmp_path):
    con = planned(tmp_path, model="llama-not-installed")
    with pytest.raises(ProviderError, match="Not installed"):
        runner.run(con, "local-test")
    assert not ollama.received


def test_ollama_cloud_models_are_refused(ollama, tmp_path):
    ollama.models = ["qwen3:4b-instruct", "gpt-oss:20b-cloud"]  # '-cloud' runs on Ollama's servers
    assert [m["name"] for m in adapters.ollama_models()] == ["qwen3:4b-instruct"]
    con = planned(tmp_path, model="gpt-oss:20b-cloud")
    with pytest.raises(ProviderError, match="not on this computer"):
        runner.run(con, "local-test")
    assert not ollama.received


def test_reasoning_level_is_sent_for_gpt_oss(ollama, tmp_path):
    ollama.models = ["gpt-oss:20b"]
    runner.run(planned(tmp_path, runs=1, conditions=("baseline",), model="gpt-oss:20b", thinking="low"), "local-test")
    assert ollama.received[0]["think"] == "low"


def test_remote_address_refused_unless_allowed(monkeypatch):
    monkeypatch.setenv("OLLAMA_HOST", "http://example.com:11434")
    with pytest.raises(ProviderError, match="not on this computer"):
        adapters.ollama_url()
    monkeypatch.setenv("OLLAMA_HOST", "0.0.0.0")  # server-side bind-all setting -> connect locally
    assert adapters.ollama_url() == "http://127.0.0.1:11434"


def test_run_valid_isolated_reproducible(ollama, tmp_path):
    con = planned(tmp_path, thinking=False)
    out = runner.run(con, "local-test")
    assert out["done"] == 6 and out["failed"] == 0
    bodies = ollama.received
    assert all([m["role"] for m in b["messages"]] == ["system", "user"] for b in bodies)  # no carried-over context
    assert all(b["format"] == response_schema() and b["think"] is False and not b["stream"] for b in bodies)
    seeds = [b["options"]["seed"] for b in bodies]
    assert len(set(seeds)) == 6  # every analysis has its own seed
    rows = con.execute("SELECT model_returned, request_json, parse_status FROM attempts").fetchall()
    assert all(r[2] == "valid" and "@abcdef123456 (ollama 9.9.9)" in r[0] for r in rows)
    assert {json.loads(r[1])["options"]["seed"] for r in rows} == set(seeds)

    ollama.received = []  # a fresh database with the same spec reproduces the same seeds
    runner.run(planned(tmp_path, name="b.sqlite"), "local-test")
    assert sorted(b["options"]["seed"] for b in ollama.received) == sorted(seeds)


def test_malformed_truncated_and_context_overflow_are_recorded(ollama):
    case = load_case(ROOT / "cases" / "IN-THEFT-001.yaml")
    p = render(case, [("E01", "base"), ("E06", "base")], "v2")
    cfg = ModelConfig(key="l", provider="ollama", model="qwen3:4b-instruct", num_ctx=1100, max_tokens=600)
    adapters.ollama_preflight(["qwen3:4b-instruct"])
    for mode, status in [("malformed", "invalid_json"), ("length", "truncated"), ("overflow", "context_overflow")]:
        ollama.mode = mode
        assert parse(call_model(cfg, p, response_schema()), p.label_map)["parse_status"] == status


def test_one_model_at_a_time(ollama, tmp_path):
    con = planned(tmp_path)
    with pytest.raises(ValueError, match="no model 'other'"):
        runner.run(con, "local-test", model_key="other")
    assert runner.run(con, "local-test", model_key="local")["done"] == 6


def test_timeout_is_retried_then_recorded(ollama, tmp_path):
    ollama.mode = "slow"
    con = planned(tmp_path, runs=1, conditions=("baseline",), timeout_s=0.3)
    out = runner.run(con, "local-test")
    assert out["failed"] == 1
    errs = [r[0] for r in con.execute("SELECT error FROM attempts")]
    assert len(errs) == runner.MAX_ATTEMPTS and all("did not answer" in e for e in errs)


def test_second_run_blocked_by_lock(ollama, tmp_path):
    con = planned(tmp_path)
    lock, _ = runner.run_files(tmp_path / "a.sqlite", "local-test")
    lock.write_text(str(__import__("os").getpid()))  # this test process is alive
    with pytest.raises(RuntimeError, match="already running"):
        runner.run(con, "local-test")
    lock.unlink()


def test_stop_request_leaves_rest_pending(ollama, tmp_path, monkeypatch):
    con = planned(tmp_path)
    real = adapters._CALL["ollama"]

    def stop_after_first(cfg, prompt, schema):
        runner.request_stop(tmp_path / "a.sqlite", "local-test")
        return real(cfg, prompt, schema)
    monkeypatch.setitem(adapters._CALL, "ollama", stop_after_first)
    out = runner.run(con, "local-test")
    assert out["stopped"] == "stopped by the researcher" and out["done"] == 1 and out["remaining"] == 5
    assert not runner.run_files(tmp_path / "a.sqlite", "local-test")[0].exists()  # lock released


def test_migration_adds_columns_and_keeps_old_data(tmp_path):
    old = sqlite3.connect(tmp_path / "old.sqlite")
    old.executescript(DDL)  # database as created by the previous version
    old.execute("INSERT INTO experiments VALUES ('x', 'h', 'y', 'now', 0)")
    old.commit()
    old.close()
    for _ in range(2):  # idempotent
        con = connect(tmp_path / "old.sqlite")
    cols = [r[1] for r in con.execute("PRAGMA table_info(attempts)")]
    assert cols.count("request_json") == 1 and cols.count("response_meta_json") == 1
    assert con.execute("SELECT experiment_id FROM experiments").fetchone()[0] == "x"
