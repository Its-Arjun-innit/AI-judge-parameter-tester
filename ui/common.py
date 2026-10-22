"""Shared helpers for the researcher interface: data access, plain-language labels, help text."""
import ctypes
import os
import shutil
import subprocess
from pathlib import Path

import pandas as pd
import streamlit as st
import yaml

from judgelab.cases import ROOT, load_case
from judgelab.db import connect
from judgelab.schema import Case, Condition

DATA_DIR = ROOT / "data"
CASES_DIR = ROOT / "cases"
EXPERIMENTS_DIR = ROOT / "experiments"
REAL_DB, PRACTICE_DB = "lab.sqlite", "dryrun.sqlite"

# ---- plain-language labels (technical value -> what a legal researcher sees) ----
KIND = {"witness": "Witness testimony", "document": "Document or exhibit", "object": "Physical object"}
CATEGORY = {"eyewitness": "Eyewitness testimony", "electronic": "Electronic record (e.g. CCTV)",
            "physical": "Physical / material object", "forensic": "Forensic evidence",
            "expert": "Expert opinion", "confession": "Confession or admission", "documentary": "Documentary evidence",
            "circumstantial": "Circumstantial evidence", "character": "Character evidence",
            "procedural": "Procedural / formal evidence"}
DIRECTION = {"inculpatory": "Points towards guilt", "exculpatory": "Points towards innocence", "neutral": "Neutral"}
ATTRIBUTE = {"reliability": "Reliability", "independence": "Independence", "specificity": "Specificity",
             "source_credibility": "Source credibility", "admissibility": "Admissibility",
             "potential_bias": "Potential bias"}
LEVEL = {"high": "high", "low": "low", "independent": "independent", "specific": "specific", "vague": "vague",
         "neutral_official": "neutral / official", "interested": "interested party", "related": "related to a party",
         "hostile": "hostile witness", "admitted": "admitted", "contested_admitted": "admitted but contested",
         "inadmissible_heard": "inadmissible (but heard)", "none": "none", "enmity": "enmity",
         "motive_to_implicate": "motive to implicate"}
HELP = {
    "reliability": "How far the evidence can be trusted to be accurate, e.g. lighting and distance for an eyewitness.",
    "independence": "Whether the item has its own source, or merely repeats another item (derived evidence).",
    "specificity": "How precisely it identifies a person, time or act.",
    "source_credibility": "Whether the source is neutral, related to a party, interested in the outcome, or hostile.",
    "admissibility": "Whether the law allows the court to rely on it, e.g. BSA s.23 bars confessions to police.",
    "potential_bias": "Any reason the source may slant the account, such as enmity with the accused.",
    "baseline": "The original evidence set is what the AI sees before any change is made.",
    "variant": "An alternative version of the same item. It may change one attribute (e.g. lower reliability) "
               "or only the wording while keeping the content the same.",
    "one_change": "Changing one piece of evidence at a time means any difference in the AI's judgment can be "
                  "attributed to that change. Several changes at once make the result impossible to interpret.",
    "repeats": "The same AI can answer the same question differently. Repeating each analysis shows how "
               "consistent it is and makes the before-and-after comparison more trustworthy.",
    "probability": "The AI's own estimate (0-100) that the accused committed the offence. It is a reported "
                   "number, not a calibrated probability, and not a measure of legal probative value.",
}
OUTCOME = {"valid": "Interpreted", "invalid_json": "Could not be interpreted automatically",
           "schema_violation": "Could not be interpreted automatically", "refusal": "The AI declined to decide",
           "truncated": "The response was cut off", "context_overflow": "The case was too long for this model",
           "api_error": "The AI could not be reached"}
MEASUREMENT_NOTE = ("Observed change in the AI's stated likelihood of guilt under controlled conditions. "
                    "It is not a measure of the AI's internal reasoning, nor of the legal probative value of the evidence.")


def label(d: dict, v) -> str:
    return d.get(v, str(v).replace("_", " "))


def defaults() -> dict:
    """Default settings for new experiments (kept in session state, outside any widget)."""
    return st.session_state.setdefault("defaults", {"max_tokens": 600, "num_ctx": 4096, "timeout_s": 300,
                                                    "temperature": "", "parallel": 1})


# ---- data access ----
def db_path() -> Path:
    return DATA_DIR / st.session_state.get("db_name", REAL_DB)


def practice() -> bool:
    return db_path().name == PRACTICE_DB


def con():
    return connect(db_path(), check_same_thread=False)


def case_files() -> dict[str, Path]:
    return {p.stem: p for p in sorted(CASES_DIR.glob("*.yaml"))}


def load_cases() -> dict[str, Case]:
    out = {}
    for cid, p in case_files().items():
        try:
            out[cid] = load_case(p)
        except Exception:  # a broken file is shown on the Cases screen, not here
            pass
    return out


def experiments(c) -> pd.DataFrame:
    """One row per experiment with progress counts."""
    return pd.read_sql_query("""
        SELECT e.experiment_id, e.created_at,
               COUNT(t.trial_id) AS total,
               SUM(t.status = 'done') AS done, SUM(t.status = 'pending') AS pending,
               SUM(t.status = 'failed') AS failed,
               GROUP_CONCAT(DISTINCT t.case_id) AS cases, GROUP_CONCAT(DISTINCT t.model_key) AS models
        FROM experiments e LEFT JOIN trials t USING (experiment_id)
        GROUP BY e.experiment_id ORDER BY e.created_at DESC""", c)


def experiment_bundle(c, exp_id: str):
    """(experiment design, observations, {case_id: the exact case version used}) for one experiment."""
    import json

    from judgelab import analysis
    from judgelab.schema import Experiment
    exp = Experiment.model_validate(yaml.safe_load(
        c.execute("SELECT spec_yaml FROM experiments WHERE experiment_id=?", (exp_id,)).fetchone()[0]))
    obs = analysis.observations(c, exp_id)
    cases = {cid: Case.model_validate(json.loads(c.execute("SELECT yaml FROM cases WHERE case_hash=?", (h,)).fetchone()[0]))
             for cid, h in obs[["case_id", "case_hash"]].drop_duplicates().itertuples(index=False)}
    return exp, obs, cases


def item_name(case: Case, item_id: str, n: int = 70) -> str:
    it = case.item(item_id)
    text = it.text.strip().replace("\n", " ")
    return f"{label(CATEGORY, it.category)} ({it.side}): “{text[:n]}{'...' if len(text) > n else ''}”"


def describe_change(case: Case, cond: Condition) -> str:
    """A plain sentence for what a scenario changes relative to its comparison point."""
    if cond.id == "baseline":
        return "Original evidence set"
    parts = []
    for op in cond.ops:
        if op.add:
            parts.append(f"Added {item_name(case, op.add)}")
        elif op.remove:
            parts.append(f"Removed {item_name(case, op.remove)}")
        elif op.variant:
            for x, v in op.variant.items():
                var = case.item(x).variants.get(v)
                if var and var.changes:
                    (a, lvl), = var.changes.items()
                    parts.append(f"Changed the {label(ATTRIBUTE, a).lower()} of {item_name(case, x, 50)} "
                                 f"to {label(LEVEL, lvl)}")
                else:
                    parts.append(f"Reworded {item_name(case, x, 50)} "
                                 f"({(var.presentation or 'new wording').replace('_', ' ') if var else v}; same content)")
        elif op.replace:
            for x, y in op.replace.items():
                parts.append(f"Replaced {item_name(case, x, 40)} with {item_name(case, y, 50)}")
        elif op.position:
            for x, where in op.position.items():
                parts.append(f"Moved {item_name(case, x, 50)} to the {'start' if where == 'first' else 'end'} "
                             f"of the {case.item(x).side} evidence")
    return "; ".join(parts)


# ---- this computer ----
@st.cache_data(ttl=3600)
def hardware() -> dict:
    """Graphics memory (NVIDIA) and RAM in GB, when detectable; None otherwise."""
    vram = ram = None
    if shutil.which("nvidia-smi"):
        try:
            out = subprocess.run(["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader,nounits"],
                                 capture_output=True, text=True, timeout=5).stdout.strip().splitlines()
            if out:
                name, mb = out[0].rsplit(",", 1)
                vram = {"gpu": name.strip(), "vram_gb": round(int(mb) / 1024, 1)}
        except (OSError, ValueError, subprocess.SubprocessError):
            pass
    if os.name == "nt":
        class MS(ctypes.Structure):
            _fields_ = [("dwLength", ctypes.c_ulong), ("dwMemoryLoad", ctypes.c_ulong),
                        ("ullTotalPhys", ctypes.c_ulonglong), ("ullAvailPhys", ctypes.c_ulonglong),
                        ("ullTotalPageFile", ctypes.c_ulonglong), ("ullAvailPageFile", ctypes.c_ulonglong),
                        ("ullTotalVirtual", ctypes.c_ulonglong), ("ullAvailVirtual", ctypes.c_ulonglong),
                        ("ullAvailExtendedVirtual", ctypes.c_ulonglong)]
        m = MS()
        m.dwLength = ctypes.sizeof(MS)
        if ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(m)):
            ram = round(m.ullTotalPhys / 1024 ** 3, 1)
    elif hasattr(os, "sysconf"):
        ram = round(os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES") / 1024 ** 3, 1)
    return {"gpu": vram["gpu"] if vram else None, "vram_gb": vram["vram_gb"] if vram else None, "ram_gb": ram}


def model_fit(size_gb: float) -> tuple[str, str]:
    """(icon, plain description) of how a model of this size will run here."""
    hw = hardware()
    vram, ram = hw["vram_gb"] or 0, hw["ram_gb"] or 16
    if vram and size_gb <= vram * 0.9:
        return "🟢", "Fits in graphics memory: the fastest option on this computer"
    if size_gb <= ram * 0.6:
        return "🟡", "Partly runs on the processor: expect each analysis to be several times slower"
    return "🔴", "Probably too large for this computer: may be very slow or fail"


def write_yaml(path: Path, data: dict) -> None:
    path.write_text(yaml.safe_dump(data, sort_keys=False, allow_unicode=True, width=110), encoding="utf-8")
