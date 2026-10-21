"""Experiment Runner: plan trials into the DB, then execute pending ones.

- Trials are shuffled once with the experiment seed, so conditions and models interleave.
- Only transport errors / 429 / 5xx are retried. Invalid output and refusals are outcomes
  and are never re-asked (re-asking would resample exactly the failing conditions).
- Resume = run again: only 'pending' trials are executed; UNIQUE keys stop duplicates.
"""
import json
import random
import time
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from datetime import datetime, timezone
from pathlib import Path

import yaml

from .adapters import ProviderError, call_model
from .cases import ROOT, Prompt, case_hash, load_case, render, sha256, template
from .conditions import expand, presentation, trial_rng
from .parse import parse
from .schema import Case, Experiment, response_schema

MAX_ATTEMPTS = 3
EST_OUTPUT_TOKENS = 1500  # ponytail: flat guess for estimates; replace with pilot means


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


def _work(cfg, prompt, schema):
    attempts = []
    for n in range(1, MAX_ATTEMPTS + 1):
        started = now()
        try:
            raw = call_model(cfg, prompt, schema)
        except ProviderError as e:
            attempts.append({"attempt_no": n, "started_at": started, "finished_at": now(), "error": str(e)[:2000],
                             "http_status": e.status, "parse_status": "api_error"})
            if not e.retryable or n == MAX_ATTEMPTS:
                return attempts, "failed"
            time.sleep(2 ** n + random.random())
            continue
        cost = (raw.tokens_in * cfg.price_in + raw.tokens_out * cfg.price_out) / 1e6
        attempts.append({"attempt_no": n, "started_at": started, "finished_at": now(),
                         "model_returned": raw.model_returned, "request_id": raw.request_id,
                         "raw_text": raw.text, "input_tokens": raw.tokens_in, "output_tokens": raw.tokens_out,
                         "stop_reason": raw.stop_reason, "cost_usd": cost,
                         **parse(raw, prompt.label_map)})
        return attempts, "done"


def run(con, experiment_id, condition=None, max_usd=None, dry_run=False, concurrency=4, log=print) -> dict:
    row = con.execute("SELECT spec_yaml FROM experiments WHERE experiment_id=?", (experiment_id,)).fetchone()
    if not row:
        raise ValueError(f"unknown experiment {experiment_id}; run `plan` first")
    exp = Experiment.model_validate(yaml.safe_load(row[0]))
    models = {m.key: (m.model_copy(update={"provider": "fake"}) if dry_run else m) for m in exp.models}
    if not dry_run:
        bad = [m.key for m in exp.models if "TODO" in m.model]
        if bad:
            raise ValueError(f"set real model ids for {bad} in the experiment spec before a live run")
    cap_usd = min(max_usd if max_usd is not None else float("inf"), exp.budget.max_usd)
    spent, calls = con.execute(
        "SELECT COALESCE(SUM(a.cost_usd),0), COUNT(*) FROM attempts a JOIN trials t USING (trial_id)"
        " WHERE t.experiment_id=?", (experiment_id,)).fetchone()
    q = "SELECT * FROM trials WHERE experiment_id=? AND status='pending'"
    args = [experiment_id]
    if condition:
        q += " AND condition_id=?"
        args.append(condition)
    pending = con.execute(q + " ORDER BY schedule_pos", args).fetchall()
    schema = response_schema()
    done = failed = 0
    stopped = None
    cols = ["trial_id", "attempt_no", "started_at", "finished_at", "model_returned", "request_id", "http_status",
            "error", "raw_text", "input_tokens", "output_tokens", "stop_reason", "cost_usd", "parse_status",
            "probability", "verdict", "confidence", "key_evidence_json", "legal_standard", "reasoning",
            "missing_info_json", "parse_error"]
    # ponytail: budget is checked before each submit, so it can overshoot by up to `concurrency` calls
    with ThreadPoolExecutor(concurrency) as pool:
        futures, it = {}, iter(pending)
        while True:
            while len(futures) < concurrency and stopped is None:
                if spent >= cap_usd:
                    stopped = f"budget cap ${cap_usd:.2f} reached"
                    break
                if calls >= exp.budget.max_calls:
                    stopped = f"max_calls {exp.budget.max_calls} reached"
                    break
                t = next(it, None)
                if t is None:
                    break
                futures[pool.submit(_work, models[t["model_key"]], _prompt(con, t), schema)] = t["trial_id"]
                calls += 1
            if not futures:
                break
            finished, _ = wait(futures, return_when=FIRST_COMPLETED)
            for f in finished:
                trial_id = futures.pop(f)
                attempts, status = f.result()
                for a in attempts:
                    a["trial_id"] = trial_id
                    con.execute(f"INSERT INTO attempts ({','.join(cols)}) VALUES ({','.join('?' * len(cols))})",
                                [a.get(c) for c in cols])
                    spent += a.get("cost_usd") or 0
                calls += len(attempts) - 1
                con.execute("UPDATE trials SET status=? WHERE trial_id=?", (status, trial_id))
                con.commit()
                done += status == "done"
                failed += status == "failed"
                if (done + failed) % 25 == 0:
                    log(f"  {done + failed}/{len(pending)} trials, ${spent:.2f} spent")
    return {"executed": done + failed, "done": done, "failed": failed,
            "remaining": len(pending) - done - failed, "spent_usd": round(spent, 4), "stopped": stopped}
