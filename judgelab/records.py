"""Permanent records outside the database: a dated database backup, a plain-file copy of each
experiment in records/<experiment_id>/ (readable without this software), and publishing that
folder to the project's git repository. Anything that looks like an API key is redacted first."""
import hashlib
import importlib.metadata
import json
import os
import platform
import re
import sqlite3
import subprocess
import sys
from datetime import datetime
from pathlib import Path

import pandas as pd

from . import analysis
from .cases import ROOT, sha256, template

RECORDS_DIR = ROOT / "records"
BACKUP_DIR = ROOT / "data" / "backups"
SECRET = re.compile(r"sk-(?:ant|proj)-[A-Za-z0-9_-]{8,}|sk-[A-Za-z0-9]{20,}|AIza[0-9A-Za-z_-]{20,}|AQ\.[A-Za-z0-9_-]{20,}")
PACKAGES = ("pydantic", "pandas", "numpy", "scipy", "statsmodels", "anthropic", "openai", "google-genai", "streamlit")


def _git(*args, timeout=60) -> str | None:
    try:
        r = subprocess.run(["git", *args], cwd=ROOT, capture_output=True, text=True, timeout=timeout)
    except (OSError, subprocess.SubprocessError):
        return None
    return r.stdout.strip() if r.returncode == 0 else None


def environment(ollama_version=None) -> dict:
    """Software used for a run, so results can be traced to exact code and package versions."""
    def ver(p):
        try:
            return importlib.metadata.version(p)
        except importlib.metadata.PackageNotFoundError:
            return None
    return {"python": sys.version.split()[0], "platform": platform.platform(),
            "judgelab_commit": _git("rev-parse", "HEAD"),
            "uncommitted_code_changes": bool(_git("status", "--porcelain", "--", "judgelab", "prompts")),
            "packages": {p: ver(p) for p in PACKAGES}, "ollama": ollama_version}


def ensure_prompts(con, version: str) -> None:
    """Store the full text of the instruction templates (by hash) so it survives later edits."""
    for name in ("system", "user"):
        text = template(name, version)
        con.execute("INSERT OR IGNORE INTO prompts VALUES (?,?,?,?,?)",
                    (sha256(text), name, version, text, datetime.now().astimezone().isoformat()))
    con.commit()


def backup(con, db_path) -> Path:
    """A complete, consistent copy of the database (safe while it is in use)."""
    BACKUP_DIR.mkdir(parents=True, exist_ok=True)
    dest = BACKUP_DIR / f"{Path(db_path).stem}-{datetime.now():%Y%m%d-%H%M%S}.sqlite"
    out = sqlite3.connect(dest)
    try:
        con.backup(out)
    finally:
        out.close()
    return dest


def _scrub(df: pd.DataFrame) -> pd.DataFrame:
    return df.apply(lambda col: col.map(lambda v: SECRET.sub("[REDACTED-KEY]", v) if isinstance(v, str) else v))


def _write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(SECRET.sub("[REDACTED-KEY]", text), encoding="utf-8", newline="\n")


def export(con, experiment_id: str, out_root: Path | None = None) -> Path:
    """Write everything recorded for one experiment as plain files, plus a manifest of checksums."""
    out = (out_root or RECORDS_DIR) / experiment_id
    q = lambda sql: _scrub(pd.read_sql_query(sql, con, params=(experiment_id,)))  # noqa: E731
    spec = con.execute("SELECT spec_yaml, prompt_version FROM experiments e JOIN trials t USING (experiment_id) "
                       "WHERE experiment_id=? LIMIT 1", (experiment_id,)).fetchone()
    if spec is None:
        raise ValueError(f"no planned analyses for {experiment_id}")
    ensure_prompts(con, spec[1])  # records older experiments' templates too (hash-checked below)
    _write(out / "experiment.yaml", spec[0])
    for h, cid, text in con.execute("SELECT DISTINCT c.case_hash, c.case_id, c.yaml FROM cases c JOIN trials t "
                                    "USING (case_hash) WHERE t.experiment_id=?", (experiment_id,)):
        _write(out / "cases" / f"{cid}-{h[:12]}.json", json.dumps(json.loads(text), indent=2, ensure_ascii=False))
    system_hashes = {r[0] for r in con.execute("SELECT DISTINCT system_hash FROM trials WHERE experiment_id=?",
                                               (experiment_id,))}
    missing = set(system_hashes)
    for h, name, version, text in con.execute("SELECT prompt_hash, name, version, text FROM prompts WHERE version IN "
                                              "(SELECT DISTINCT prompt_version FROM trials WHERE experiment_id=?)",
                                              (experiment_id,)):
        if name == "user" or h in system_hashes:
            _write(out / "prompts" / f"{name}_{version}-{h[:12]}.md", text)
            missing.discard(h)
    tables = {
        "conditions.csv": q("SELECT * FROM conditions WHERE experiment_id=? ORDER BY condition_id"),
        "trials.csv": q("SELECT * FROM trials WHERE experiment_id=? ORDER BY trial_id"),
        "attempts.csv": q("SELECT a.* FROM attempts a JOIN trials t USING (trial_id) WHERE t.experiment_id=? "
                          "ORDER BY a.attempt_id"),
        "runs.csv": q("SELECT * FROM runs WHERE experiment_id=? ORDER BY run_id"),
        "observations.csv": _scrub(analysis.observations(con, experiment_id)),
    }
    for name, df in tables.items():
        _write(out / name, df.to_csv(index=False, lineterminator="\n"))
    files = sorted(p for p in out.rglob("*") if p.is_file() and p.name != "manifest.json")
    last = con.execute("SELECT MAX(a.finished_at) FROM attempts a JOIN trials t USING (trial_id) "
                       "WHERE t.experiment_id=?", (experiment_id,)).fetchone()[0]
    manifest = {"experiment_id": experiment_id, "records_up_to": last,
                "counts": {k[:-4]: len(v) for k, v in tables.items()},
                "system_prompt_text_missing_for_hashes": sorted(missing),
                "sha256": {p.relative_to(out).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest() for p in files}}
    _write(out / "manifest.json", json.dumps(manifest, indent=2))
    return out


def publish(experiment_ids: list[str], log=print) -> str:
    """Commit records/<id>/ for these experiments (only those paths) and push. Never blocks a run:
    any problem is reported, and the records stay safely on disk and in the database."""
    if os.environ.get("JUDGELAB_PUBLISH_RECORDS", "1") == "0":
        return "publishing is turned off (JUDGELAB_PUBLISH_RECORDS=0)"
    if _git("rev-parse", "--is-inside-work-tree") != "true":
        return "not a git repository; records kept on disk only"
    paths = [(RECORDS_DIR / e).relative_to(ROOT).as_posix() for e in experiment_ids]
    _git("add", "--", *paths)
    if not _git("diff", "--cached", "--name-only", "--", *paths):
        return "records already up to date in git"
    msg = f"Record results: {', '.join(experiment_ids)}"
    r = subprocess.run(["git", "commit", "-m", msg, "--", *paths], cwd=ROOT, capture_output=True, text=True)
    if r.returncode != 0:
        return f"could not commit records: {(r.stderr or r.stdout).strip()[:300]}"
    p = subprocess.run(["git", "push"], cwd=ROOT, capture_output=True, text=True, timeout=180)
    if p.returncode != 0:
        return f"records committed locally; push failed (will be included in the next push): {p.stderr.strip()[:300]}"
    return f"records committed and pushed ({msg})"


def after_run(con, db_path, experiment_ids: list[str], log=print) -> None:
    """Backup, plain-file export and publication after real runs."""
    log(f"  backup: {backup(con, db_path).relative_to(ROOT)}")
    for e in experiment_ids:
        log(f"  records: {export(con, e).relative_to(ROOT)}")
    log(f"  {publish(experiment_ids, log)}")
