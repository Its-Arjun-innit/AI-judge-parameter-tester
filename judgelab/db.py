"""Results Database (SQLite). Duplicate trials are prevented by a UNIQUE constraint."""
import sqlite3
from pathlib import Path

DDL = """
CREATE TABLE IF NOT EXISTS cases (
  case_hash TEXT PRIMARY KEY, case_id TEXT NOT NULL, yaml TEXT NOT NULL, loaded_at TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS experiments (
  experiment_id TEXT PRIMARY KEY, spec_hash TEXT NOT NULL, spec_yaml TEXT NOT NULL,
  created_at TEXT NOT NULL, frozen INTEGER NOT NULL DEFAULT 0);
CREATE TABLE IF NOT EXISTS conditions (
  experiment_id TEXT NOT NULL, condition_id TEXT NOT NULL, ref TEXT NOT NULL,
  factor TEXT NOT NULL, level TEXT NOT NULL, exploratory INTEGER NOT NULL, ops_json TEXT NOT NULL,
  PRIMARY KEY (experiment_id, condition_id));
CREATE TABLE IF NOT EXISTS trials (
  trial_id INTEGER PRIMARY KEY, experiment_id TEXT NOT NULL, case_id TEXT NOT NULL, case_hash TEXT NOT NULL,
  condition_id TEXT NOT NULL, model_key TEXT NOT NULL, run_index INTEGER NOT NULL, schedule_pos INTEGER NOT NULL,
  evidence_order_json TEXT NOT NULL, label_map_json TEXT NOT NULL, prompt_version TEXT NOT NULL,
  system_hash TEXT NOT NULL, user_prompt TEXT NOT NULL, prompt_hash TEXT NOT NULL, params_json TEXT NOT NULL,
  status TEXT NOT NULL DEFAULT 'pending' CHECK (status IN ('pending','done','failed','excluded')),
  exclusion_reason TEXT,
  UNIQUE (experiment_id, case_id, condition_id, model_key, run_index));
CREATE TABLE IF NOT EXISTS attempts (
  attempt_id INTEGER PRIMARY KEY, trial_id INTEGER NOT NULL REFERENCES trials(trial_id), attempt_no INTEGER NOT NULL,
  started_at TEXT NOT NULL, finished_at TEXT NOT NULL, model_returned TEXT, request_id TEXT,
  http_status INTEGER, error TEXT, raw_text TEXT, input_tokens INTEGER, output_tokens INTEGER,
  stop_reason TEXT, cost_usd REAL NOT NULL DEFAULT 0, parse_status TEXT NOT NULL,
  probability INTEGER, verdict TEXT, confidence INTEGER, key_evidence_json TEXT,
  legal_standard TEXT, reasoning TEXT, missing_info_json TEXT, parse_error TEXT);
CREATE TABLE IF NOT EXISTS prompts (  -- full text of every instruction template ever used, by hash
  prompt_hash TEXT PRIMARY KEY, name TEXT NOT NULL, version TEXT NOT NULL, text TEXT NOT NULL, first_seen TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS runs (     -- one row per run session: options, software versions, outcome
  run_id INTEGER PRIMARY KEY, experiment_id TEXT NOT NULL, started_at TEXT NOT NULL, finished_at TEXT,
  pid INTEGER, options_json TEXT, environment_json TEXT, summary_json TEXT);
"""

# Research records are permanent: the database itself refuses to delete or rewrite them.
# The only allowed changes: a trial's status moving forward (a completed analysis can only be
# marked 'excluded', with a reason), and a run session being finished once.
_PERMANENT = "research records are permanent"
PERMANENT = "".join(
    f"CREATE TRIGGER IF NOT EXISTS keep_{t}_delete BEFORE DELETE ON {t} "
    f"BEGIN SELECT RAISE(ABORT, '{_PERMANENT}: rows in {t} cannot be deleted'); END;\n"
    for t in ("cases", "experiments", "conditions", "trials", "attempts", "prompts", "runs")) + "".join(
    f"CREATE TRIGGER IF NOT EXISTS keep_{t}_update BEFORE UPDATE ON {t} "
    f"BEGIN SELECT RAISE(ABORT, '{_PERMANENT}: rows in {t} cannot be changed'); END;\n"
    for t in ("cases", "conditions", "attempts", "prompts")) + f"""
CREATE TRIGGER IF NOT EXISTS keep_experiments_update BEFORE UPDATE ON experiments
WHEN NEW.experiment_id IS NOT OLD.experiment_id OR NEW.spec_yaml IS NOT OLD.spec_yaml
  OR NEW.spec_hash IS NOT OLD.spec_hash OR NEW.created_at IS NOT OLD.created_at
BEGIN SELECT RAISE(ABORT, '{_PERMANENT}: an experiment design cannot be changed'); END;
CREATE TRIGGER IF NOT EXISTS keep_trials_update BEFORE UPDATE ON trials
WHEN NEW.trial_id IS NOT OLD.trial_id OR NEW.experiment_id IS NOT OLD.experiment_id
  OR NEW.case_id IS NOT OLD.case_id OR NEW.case_hash IS NOT OLD.case_hash OR NEW.condition_id IS NOT OLD.condition_id
  OR NEW.model_key IS NOT OLD.model_key OR NEW.run_index IS NOT OLD.run_index OR NEW.schedule_pos IS NOT OLD.schedule_pos
  OR NEW.evidence_order_json IS NOT OLD.evidence_order_json OR NEW.label_map_json IS NOT OLD.label_map_json
  OR NEW.prompt_version IS NOT OLD.prompt_version OR NEW.system_hash IS NOT OLD.system_hash
  OR NEW.user_prompt IS NOT OLD.user_prompt OR NEW.prompt_hash IS NOT OLD.prompt_hash
  OR NEW.params_json IS NOT OLD.params_json
  OR (OLD.status = 'done' AND NEW.status NOT IN ('done', 'excluded'))
  OR (OLD.status = 'excluded' AND NEW.status != 'excluded')
  OR (NEW.status = 'excluded' AND COALESCE(TRIM(NEW.exclusion_reason), '') = '')
BEGIN SELECT RAISE(ABORT, '{_PERMANENT}: only a trial''s status can change; completed analyses can only be excluded, with a reason'); END;
CREATE TRIGGER IF NOT EXISTS keep_runs_update BEFORE UPDATE ON runs
WHEN OLD.finished_at IS NOT NULL OR NEW.run_id IS NOT OLD.run_id OR NEW.experiment_id IS NOT OLD.experiment_id
  OR NEW.started_at IS NOT OLD.started_at
BEGIN SELECT RAISE(ABORT, '{_PERMANENT}: a finished run session cannot be changed'); END;
"""

# Additive, idempotent migrations: existing rows keep NULL, nothing is rewritten.
ADDED_COLUMNS = {"attempts": {"request_json": "TEXT", "response_meta_json": "TEXT"}}


def connect(path, check_same_thread=True) -> sqlite3.Connection:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    con = sqlite3.connect(path, check_same_thread=check_same_thread)
    con.row_factory = sqlite3.Row
    con.execute("PRAGMA journal_mode=WAL")  # crash-safe, and the dashboard can read while a run writes
    con.execute("PRAGMA busy_timeout=10000")
    con.executescript(DDL)
    for table, cols in ADDED_COLUMNS.items():
        have = {r[1] for r in con.execute(f"PRAGMA table_info({table})")}
        for col, typ in cols.items():
            if col not in have:
                con.execute(f"ALTER TABLE {table} ADD COLUMN {col} {typ}")
    con.executescript(PERMANENT)
    con.commit()
    return con
