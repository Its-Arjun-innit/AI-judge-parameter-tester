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
"""


# Additive, idempotent migrations: existing rows keep NULL, nothing is rewritten.
ADDED_COLUMNS = {"attempts": {"request_json": "TEXT", "response_meta_json": "TEXT"}}


def connect(path, check_same_thread=True) -> sqlite3.Connection:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    con = sqlite3.connect(path, check_same_thread=check_same_thread)
    con.row_factory = sqlite3.Row
    con.executescript(DDL)
    for table, cols in ADDED_COLUMNS.items():
        have = {r[1] for r in con.execute(f"PRAGMA table_info({table})")}
        for col, typ in cols.items():
            if col not in have:
                con.execute(f"ALTER TABLE {table} ADD COLUMN {col} {typ}")
    con.commit()
    return con
