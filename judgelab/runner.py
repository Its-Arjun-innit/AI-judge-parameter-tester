"""Experiment Runner: plan trials into the DB, then execute pending ones.

- Trials are shuffled once with the experiment seed, so conditions and models interleave.
- Only transport errors / 429 / 5xx are retried. Invalid output and refusals are outcomes
  and are never re-asked (re-asking would resample exactly the failing conditions).
- Resume = run again: only 'pending' trials are executed; UNIQUE keys stop duplicates.
"""
import functools
import json
import os
import random
import threading
import time
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from datetime import datetime, timezone
from pathlib import Path

import yaml

from .adapters import OLLAMA_INFO, ProviderError, call_model, ollama_preflight
from .records import ensure_prompts, environment
from .cases import ROOT, Prompt, case_hash, load_case, render, sha256, template
from .conditions import expand, presentation, trial_rng
from .parse import parse
from .schema import Case, Experiment, response_schema

MAX_ATTEMPTS = 3
EST_OUTPUT_TOKENS = 400  # ponytail: flat guess for estimates (lean v2 prompt measured ~270); use pilot means


def now():
    return datetime.now(timezone.utc).isoformat()


def load_experiment(path) -> tuple[Experiment, list[Case], str]:
    text = Path(path).read_text(encoding="utf-8")
    exp = Experiment.model_validate(yaml.safe_load(text))
    cases = [load_case(ROOT / "cases" / f"{cid}.yaml") for cid in exp.cases]
    return exp, cases, text


def plan(con, path) -> dict:
    exp, cases, spec_text = load_experiment(path)
    states = expand(exp, cases)  # raises if any condition is not a single controlled change
    row = con.execute("SELECT spec_hash, frozen FROM experiments WHERE experiment_id=?",
                      (exp.experiment_id,)).fetchone()
    if row and row["spec_hash"] != sha256(spec_text):
        raise ValueError(f"{exp.experiment_id} already planned with a different spec; "
                         "use a new experiment_id (frozen data must not change)")
    con.execute("INSERT OR IGNORE INTO experiments VALUES (?,?,?,?,0)",
                (exp.experiment_id, sha256(spec_text), spec_text, now()))
    hashes = {}
    for c in cases:
        hashes[c.case_id] = case_hash(c)
        con.execute("INSERT OR IGNORE INTO cases VALUES (?,?,?,?)",
                    (hashes[c.case_id], c.case_id, json.dumps(c.model_dump(mode="json")), now()))
    for c in exp.conditions:
        con.execute("INSERT OR IGNORE INTO conditions VALUES (?,?,?,?,?,?,?)",
                    (exp.experiment_id, c.id, c.ref, c.factor, c.level, int(c.exploratory),
                     json.dumps([o.model_dump(exclude_none=True) for o in c.ops])))
    keys = [(case, c.id, m, r) for case in cases for c in exp.conditions for m in exp.models
            for r in range(exp.runs)]
    random.Random(exp.seed).shuffle(keys)
    ensure_prompts(con, exp.prompt_version)  # full instruction texts, kept even if the files change later
    system_hash = sha256(template("system", exp.prompt_version))
    new = est_usd = 0
    for pos, (case, cid, m, r) in enumerate(keys):
        st = states[case.case_id, cid]
        order = presentation(case, st, exp.order == "randomize_within_side",
                             trial_rng(exp.seed, case.case_id, cid, r))
        p = render(case, order, exp.prompt_version)
        cur = con.execute(
            "INSERT OR IGNORE INTO trials (experiment_id, case_id, case_hash, condition_id, model_key, run_index,"
            " schedule_pos, evidence_order_json, label_map_json, prompt_version, system_hash, user_prompt,"
            " prompt_hash, params_json) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
            (exp.experiment_id, case.case_id, hashes[case.case_id], cid, m.key, r, pos, json.dumps(order),
             json.dumps(p.label_map), exp.prompt_version, system_hash, p.user, sha256(p.user),
             m.model_dump_json()))
        new += cur.rowcount
        est_usd += ((len(p.system) + len(p.user)) / 4 * m.price_in + EST_OUTPUT_TOKENS * m.price_out) / 1e6
    con.commit()
    return {"experiment_id": exp.experiment_id, "trials": len(keys), "new_trials": new,
            "estimated_usd_all_trials": round(est_usd, 2),
            "unpriced_models": [m.key for m in exp.models if not (m.price_in and m.price_out)]}


def _prompt(con, trial) -> Prompt:
    """Re-render from stored case content and verify it matches what was planned."""
    case = Case.model_validate(json.loads(
        con.execute("SELECT yaml FROM cases WHERE case_hash=?", (trial["case_hash"],)).fetchone()[0]))
    p = render(case, [tuple(x) for x in json.loads(trial["evidence_order_json"])], trial["prompt_version"])
    if sha256(p.user) != trial["prompt_hash"] or sha256(p.system) != trial["system_hash"]:
        raise RuntimeError(f"trial {trial['trial_id']}: prompt template changed since planning")
    return p


_pace_lock, _next_slot = threading.Lock(), {}
QUOTA_WORDS = ("quota", "resource_exhausted")
BILLING_WORDS = ("credit balance", "credits remaining", "insufficient_quota", "billing")
KEY_WORDS = ("api_key", "api key", "authentication method", "authentication_error", "invalid x-api-key")


def _pace(cfg):
    """Reserve the model's next call slot so requests stay under cfg.rpm."""
    if not cfg.rpm:
        return
    with _pace_lock:
        slot = max(time.monotonic(), _next_slot.get(cfg.key, 0.0))
        _next_slot[cfg.key] = slot + 60 / cfg.rpm
    time.sleep(max(0.0, slot - time.monotonic()))


def _work(cfg, prompt, schema):
    """Returns (attempts, status): done | failed | paused (trial goes back to pending: rate/quota
    limit, no credit, or the local Ollama service/model is unavailable)."""
    attempts = []
    for n in range(1, MAX_ATTEMPTS + 1):
        _pace(cfg)
        started = now()
        try:
            raw = call_model(cfg, prompt, schema)
        except ProviderError as e:
            attempts.append({"attempt_no": n, "started_at": started, "finished_at": now(), "error": str(e)[:2000],
                             "http_status": e.status, "parse_status": "api_error", "cost_usd": 0.0})
            msg = str(e).lower()
            billing = any(w in msg for w in BILLING_WORDS)  # no credit: retrying cannot help
            local_down = "ollama" in msg and ("not running" in msg or "not installed" in msg)
            no_key = e.status == 401 or any(w in msg for w in KEY_WORDS)  # key removed or invalid: fixable
            quota = billing or local_down or no_key or e.status == 429 or any(w in msg for w in QUOTA_WORDS)
            if quota and (billing or local_down or no_key or not e.retryable or n == MAX_ATTEMPTS):
                return attempts, "paused"
            if not e.retryable or n == MAX_ATTEMPTS:
                return attempts, "failed"
            time.sleep(2 ** n + random.random())
            continue
        cost = (raw.tokens_in * cfg.price_in + raw.tokens_out * cfg.price_out) / 1e6
        attempts.append({"attempt_no": n, "started_at": started, "finished_at": now(),
                         "model_returned": raw.model_returned, "request_id": raw.request_id,
                         "raw_text": raw.text, "input_tokens": raw.tokens_in, "output_tokens": raw.tokens_out,
                         "stop_reason": raw.stop_reason, "cost_usd": cost,
                         "request_json": json.dumps(raw.request), "response_meta_json": json.dumps(raw.meta),
                         **parse(raw, prompt.label_map)})
        return attempts, "done"


def run_files(db_path, experiment_id) -> tuple[Path, Path]:
    """(lock file holding the runner's PID, stop-request file) next to the database."""
    p = Path(db_path)
    return p.with_name(f"{p.stem}-run-{experiment_id}.lock"), p.with_name(f"{p.stem}-run-{experiment_id}.stop")


def _pid_alive(pid: int) -> bool:
    if os.name == "nt":  # os.kill(pid, 0) would terminate the process on Windows
        import ctypes
        k32 = ctypes.windll.kernel32
        h = k32.OpenProcess(0x1000, False, pid)  # PROCESS_QUERY_LIMITED_INFORMATION
        if not h:
            return False
        code = ctypes.c_ulong()
        k32.GetExitCodeProcess(h, ctypes.byref(code))
        k32.CloseHandle(h)
        return code.value == 259  # STILL_ACTIVE
    try:
        os.kill(pid, 0)
        return True
    except OSError:
        return False


def running_pid(db_path, experiment_id) -> int | None:
    """PID of a live runner for this experiment, or None (stale locks are ignored)."""
    try:
        pid = int(run_files(db_path, experiment_id)[0].read_text())
    except (OSError, ValueError):
        return None
    return pid if _pid_alive(pid) else None


def request_stop(db_path, experiment_id) -> None:
    """Ask a running experiment to stop after the analyses already in progress."""
    run_files(db_path, experiment_id)[1].write_text("stop")


def run(con, experiment_id, condition=None, max_usd=None, dry_run=False, concurrency=None,
        log=functools.partial(print, flush=True), model_key=None) -> dict:
    """model_key: run only that model's analyses - on one PC this avoids reloading large local
    models between analyses; the order of scenarios within the model stays randomised.
    Every call is recorded in the runs table: options, software versions and outcome."""
    cur = con.execute("INSERT INTO runs (experiment_id, started_at, pid, options_json) VALUES (?,?,?,?)",
                      (experiment_id, now(), os.getpid(), json.dumps(
                          {"condition": condition, "model": model_key, "max_usd": max_usd, "dry_run": dry_run,
                           "concurrency": concurrency})))
    con.commit()
    summary = {"error": "interrupted"}
    try:
        summary = _run_locked(con, experiment_id, condition, max_usd, dry_run, concurrency, log, model_key)
        return summary
    except Exception as e:
        summary = {"error": f"{type(e).__name__}: {e}"}
        raise
    finally:
        con.execute("UPDATE runs SET finished_at=?, environment_json=?, summary_json=? WHERE run_id=?",
                    (now(), json.dumps(environment(OLLAMA_INFO["version"])), json.dumps(summary), cur.lastrowid))
        con.commit()


def _run_locked(con, experiment_id, condition, max_usd, dry_run, concurrency, log, model_key) -> dict:
    db_path = con.execute("PRAGMA database_list").fetchone()[2]
    if not db_path:  # in-memory database: no other process can share it, no lock needed
        return _run(con, experiment_id, condition, max_usd, dry_run, concurrency, log, None, model_key)
    lock, stop_file = run_files(db_path, experiment_id)
    try:  # atomic: two runners started at the same moment cannot both get the lock
        fd = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    except FileExistsError:
        pid = running_pid(db_path, experiment_id)
        if pid:
            raise RuntimeError(f"{experiment_id} is already running (process {pid}). Stop it before starting again.")
        lock.unlink(missing_ok=True)  # stale lock left by a crashed run
        fd = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    os.write(fd, str(os.getpid()).encode())
    os.close(fd)
    stop_file.unlink(missing_ok=True)
    try:
        return _run(con, experiment_id, condition, max_usd, dry_run, concurrency, log, stop_file, model_key)
    finally:
        lock.unlink(missing_ok=True)
        stop_file.unlink(missing_ok=True)


def _run(con, experiment_id, condition, max_usd, dry_run, concurrency, log, stop_file, model_key=None) -> dict:
    row = con.execute("SELECT spec_yaml FROM experiments WHERE experiment_id=?", (experiment_id,)).fetchone()
    if not row:
        raise ValueError(f"unknown experiment {experiment_id}; run `plan` first")
    exp = Experiment.model_validate(yaml.safe_load(row[0]))
    if model_key and model_key not in {m.key for m in exp.models}:
        raise ValueError(f"no model '{model_key}' in {experiment_id}; choose one of {[m.key for m in exp.models]}")
    used = [m for m in exp.models if not model_key or m.key == model_key]
    models = {m.key: (m.model_copy(update={"provider": "fake"}) if dry_run else m) for m in exp.models}
    local = sorted({m.model for m in used if m.provider == "ollama"})
    if not dry_run:
        bad = [m.key for m in used if "TODO" in m.model]
        if bad:
            raise ValueError(f"set real model ids for {bad} in the experiment spec before a live run")
        if local:
            ollama_preflight(local)  # Ollama running + every model installed, before any analysis
    if concurrency is None:  # one analysis at a time on a personal computer
        concurrency = 1 if local and not dry_run else 4
    cap_usd = min(max_usd if max_usd is not None else float("inf"), exp.budget.max_usd)
    spent, calls = con.execute(
        "SELECT COALESCE(SUM(a.cost_usd),0), COUNT(*) FROM attempts a JOIN trials t USING (trial_id)"
        " WHERE t.experiment_id=?", (experiment_id,)).fetchone()
    q = "SELECT * FROM trials WHERE experiment_id=? AND status='pending'"
    args = [experiment_id]
    if condition:
        q += " AND condition_id=?"
        args.append(condition)
    if model_key:
        q += " AND model_key=?"
        args.append(model_key)
    pending = con.execute(q + " ORDER BY schedule_pos", args).fetchall()
    schema = response_schema()
    done = failed = processed = 0
    stopped = None
    blocked = set()  # models paused this run (limit, no credit, Ollama unavailable); their trials stay pending
    cols = ["trial_id", "attempt_no", "started_at", "finished_at", "model_returned", "request_id", "http_status",
            "error", "raw_text", "input_tokens", "output_tokens", "stop_reason", "cost_usd", "parse_status",
            "probability", "verdict", "confidence", "key_evidence_json", "legal_standard", "reasoning",
            "missing_info_json", "parse_error", "request_json", "response_meta_json"]
    t0 = time.monotonic()
    # ponytail: budget is checked before each submit, so it can overshoot by up to `concurrency` calls
    with ThreadPoolExecutor(concurrency) as pool:
        futures, it, models_by_trial = {}, iter(pending), {}
        while True:
            while len(futures) < concurrency and stopped is None:
                if stop_file and stop_file.exists():
                    stopped = "stopped by the researcher"
                    break
                if calls >= exp.budget.max_calls:
                    stopped = f"max_calls {exp.budget.max_calls} reached"
                    break
                t = next(it, None)
                if t is None:
                    break
                if t["model_key"] in blocked:
                    continue
                m = models[t["model_key"]]
                if spent >= cap_usd and (m.price_in or m.price_out):  # free local models never hit the cap
                    stopped = f"budget cap ${cap_usd:.2f} reached"
                    break
                # a fixed seed per analysis: repeats differ, yet each one can be reproduced exactly
                seed = int(sha256(f"{exp.seed}|{t['case_id']}|{t['condition_id']}|{t['model_key']}|"
                                  f"{t['run_index']}")[:8], 16)
                cfg = models[t["model_key"]].model_copy(update={"seed": seed})
                futures[pool.submit(_work, cfg, _prompt(con, t), schema)] = t["trial_id"]
                models_by_trial[t["trial_id"]] = t["model_key"]
                calls += 1
            if not futures:
                break
            try:
                finished, _ = wait(futures, return_when=FIRST_COMPLETED)
            except KeyboardInterrupt:  # Ctrl+C: submit nothing new; record analyses already running
                stopped = "cancelled (Ctrl+C)"
                log("  stopping after the analyses already in progress...")
                continue
            for f in finished:
                trial_id = futures.pop(f)
                attempts, status = f.result()
                for a in attempts:
                    a["trial_id"] = trial_id
                    con.execute(f"INSERT INTO attempts ({','.join(cols)}) VALUES ({','.join('?' * len(cols))})",
                                [a.get(c) for c in cols])
                    spent += a.get("cost_usd") or 0
                calls += len(attempts) - 1
                if status == "paused":
                    key = models_by_trial[trial_id]
                    if key not in blocked:
                        log(f"  paused {key}: {attempts[-1]['error'][:300]}")
                    blocked.add(key)
                    status = "pending"  # retry on a later run; the failed attempts stay logged
                models_by_trial.pop(trial_id, None)
                con.execute("UPDATE trials SET status=? WHERE trial_id=?", (status, trial_id))
                con.commit()
                done += status == "done"
                failed += status == "failed"
                processed += 1
                el = time.monotonic() - t0
                if el / processed > 1 or processed % 25 == 0:  # slow local models: report every analysis
                    log(f"  {processed}/{len(pending)} analyses processed, {el:.0f}s elapsed, "
                        f"about {el / processed * (len(pending) - processed):.0f}s left, ${spent:.2f} spent")
    return {"executed": done + failed, "done": done, "failed": failed,
            "remaining": len(pending) - done - failed, "spent_usd": round(spent, 4), "stopped": stopped,
            "paused_models": sorted(blocked)}
