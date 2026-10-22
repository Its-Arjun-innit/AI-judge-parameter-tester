"""Experiments: design a controlled evidence comparison, choose a local AI, and run it."""
import os
import random
import re
import subprocess
import sys
import time
from datetime import datetime

import streamlit as st

from judgelab import runner
from judgelab.adapters import ollama_models, ollama_status
from judgelab.conditions import ConditionError, apply, expand
from judgelab.schema import Budget, Condition, Experiment, ModelConfig
from ui import common

st.title("Experiments")
c = common.con()
db = common.db_path()


def start(exp_id: str, parallel: int = 1):
    """Run pending analyses in a background process (same CLI as the terminal); log to data/logs."""
    logs = common.DATA_DIR / "logs"
    logs.mkdir(parents=True, exist_ok=True)
    args = [sys.executable, "-m", "judgelab", "--db", str(db)] + (["--dry-run"] if common.practice() else [])
    args += ["run", exp_id] + (["--concurrency", str(parallel)] if parallel > 1 else [])
    with open(logs / f"{exp_id}.log", "a", encoding="utf-8") as f:
        f.write(f"\n--- started {datetime.now():%Y-%m-%d %H:%M:%S} ---\n")
        f.flush()
        subprocess.Popen(args, cwd=common.ROOT, stdout=f, stderr=subprocess.STDOUT,
                         creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0)
    st.session_state.setdefault("starting", {})[exp_id] = time.time()


@st.fragment(run_every=3)
def progress(exp_id: str):
    r = common.experiments(common.con()).set_index("experiment_id").loc[exp_id]
    total, done, failed = int(r.total or 0), int(r.done or 0), int(r.failed or 0)
    running = runner.running_pid(db, exp_id) or time.time() - st.session_state.get("starting", {}).get(exp_id, 0) < 8
    note = f"{done} of {total} analyses complete" + (f", {failed} failed" if failed else "")
    st.progress(done / total if total else 0.0, text=note + (" - running..." if running else ""))
    was = st.session_state.setdefault("was_running", {})
    if was.get(exp_id) and not running:  # just finished: refresh the buttons
        was[exp_id] = False
        st.rerun()
    was[exp_id] = bool(running)


# ---- existing experiments ----
st.subheader("Your experiments")
exps = common.experiments(c)
if exps.empty:
    st.caption("None yet. Design your first one below.")
for r in exps.itertuples():
    with st.container(border=True):
        left, right = st.columns([5, 2])
        left.markdown(f"**{r.experiment_id}**  \nCase {r.cases or '-'} · AI: {r.models or '-'}")
        with left:
            progress(r.experiment_id)
        running = runner.running_pid(db, r.experiment_id)
        if running:
            if right.button("Stop", key=f"stop-{r.experiment_id}",
                            help="Finishes the analysis in progress, then stops. Nothing is lost; you can continue later."):
                runner.request_stop(db, r.experiment_id)
                st.toast("Stopping after the current analysis...")
        elif r.pending:
            if right.button("Run remaining analyses", key=f"run-{r.experiment_id}"):
                start(r.experiment_id, common.defaults()["parallel"])
                st.rerun()
        if r.done and right.button("Open results", key=f"res-{r.experiment_id}"):
            st.session_state["results_exp"] = r.experiment_id
            st.switch_page("ui/results.py")

# ---- guided workflow ----
st.divider()
st.subheader("Design a new experiment")
cases = common.load_cases()
if not cases:
    st.info("There are no cases yet. Create one first.")
    st.page_link("ui/cases.py", label="Go to Cases", icon=":material/gavel:")
    st.stop()

st.markdown("#### Step 1 - Choose a case")
case_id = st.selectbox("Case", list(cases), key="w_case",
                       format_func=lambda k: f"{k}: {cases[k].law.offence[:90]}")
case = cases[case_id]
if st.session_state.get("w_changes_for") != case_id:  # new case: start a fresh comparison list
    st.session_state["w_changes_for"], st.session_state["w_changes"] = case_id, []
changes: list[dict] = st.session_state["w_changes"]
with st.expander("What the AI is told about the law (the same in every scenario)"):
    st.markdown(f"**Charge:** {case.law.offence}  \n**Standard of proof:** {case.law.standard_of_proof}")
    st.text(case.law.provisions_text)
    for p in case.law.principles:
        st.markdown(f"- {p}")

st.markdown("#### Step 2 - Review the evidence")
st.caption(common.HELP["baseline"])
pros, defs = st.columns(2)
for col, side in ((pros, "prosecution"), (defs, "defence")):
    col.markdown(f"**{side.capitalize()} evidence in the original set**")
    for i in case.baseline:
        if case.item(i).side == side:
            col.markdown(f"- {common.item_name(case, i, 110)}")
spare = [e.id for e in case.evidence if e.id not in case.baseline]
if spare:
    with st.expander(f"{len(spare)} more items available for comparisons (not in the original set)"):
        for i in spare:
            st.markdown(f"- {common.item_name(case, i, 110)}")
st.page_link("ui/cases.py", label="Edit this case's evidence", icon=":material/edit:")

st.markdown("#### Step 3 - Choose what to compare")
st.caption(common.HELP["one_change"])


def draft_conditions(extra: dict | None = None) -> list[Condition]:
    out = [Condition(id="baseline")]
    for ch in changes + ([extra] if extra else []):
        out.append(Condition(**{k: v for k, v in ch.items() if k != "label"}))
    return out


def check(conds: list[Condition]):
    """Raises ConditionError/ValueError unless every scenario is one controlled change from its comparison."""
    expand(Experiment(experiment_id="draft", cases=[case_id], prompt_version="v2", runs=1, seed=1,
                      models=[ModelConfig(key="x", provider="fake", model="fake")],
                      budget=Budget(max_usd=0, max_calls=1), conditions=conds), [case])


def factor_level(kind: str, item: str, value: str | None) -> tuple[str, str]:
    it = case.item(item)
    if kind == "add":
        if it.relations.contradicts:
            return "contradiction", "added"
        if it.relations.corroborates:
            return "corroboration", "dependent" if it.attributes.independence.startswith("derived") else "independent"
        return ("control", "neutral_item") if it.direction == "neutral" else ("presence", "added")
    if kind == "remove":
        return "presence", "removed"
    if kind == "variant":
        v = it.variants[value]
        return next(iter(v.changes.items())) if v.changes else ("wording", v.presentation)
    if kind == "replace":
        return "category", case.item(value).category
    return "order", value


conds_now = {k.id: k for k in draft_conditions()}
refs = ["baseline"]
with st.expander("More complex comparisons (optional)"):
    st.caption("By default each change is compared with the original evidence set. You can instead build on a "
               "scenario you already added, e.g. to test two pieces of evidence together.")
    refs += [ch["id"] for ch in changes]
    ref = st.selectbox("Compare with", refs, key="w_ref",
                       format_func=lambda k: "The original evidence set" if k == "baseline" else f"Scenario '{k}'")
present = [i for i, _ in apply(case, conds_now, ref).items]
absent = [e.id for e in case.evidence if e.id not in present]
KINDS = {"add": "Add an item", "remove": "Remove an item", "variant": "Use an alternative version of an item",
         "replace": "Replace an item with a different type of evidence", "position": "Move an item to the start or end"}
kind = st.selectbox("Type of change", list(KINDS), format_func=KINDS.get, key="w_kind")
name = lambda i: common.item_name(case, i, 90)  # noqa: E731
item = value = None
if kind == "add":
    item = st.selectbox("Item to add", absent, format_func=name, key="w_item_add") if absent else None
elif kind == "remove":
    item = st.selectbox("Item to remove", present, format_func=name, key="w_item_rm")
elif kind == "variant":
    with_variants = [i for i in present if case.item(i).variants]
    item = st.selectbox("Item", with_variants, format_func=name, key="w_item_var") if with_variants else None
    if item:
        vs = case.item(item).variants
        value = st.selectbox("Alternative version", list(vs), key="w_variant", help=common.HELP["variant"],
                             format_func=lambda v: (f"{v}: changes {common.label(common.ATTRIBUTE, next(iter(vs[v].changes)))}"
                                                    f" to {common.label(common.LEVEL, next(iter(vs[v].changes.values())))}"
                                                    if vs[v].changes else f"{v}: same content, different wording"))
        st.caption(f"New text: {vs[value].text.strip()}")
elif kind == "replace":
    item = st.selectbox("Item to replace", present, format_func=name, key="w_item_rep")
    value = st.selectbox("Replacement", absent, format_func=name, key="w_item_with") if absent else None
else:
    item = st.selectbox("Item to move", present, format_func=name, key="w_item_mv")
    value = st.radio("Move it to", ["first", "last"], horizontal=True, key="w_where",
                     format_func={"first": "the start of its side's evidence", "last": "the end"}.get)
if (kind == "add" and not absent) or (kind == "variant" and not item) or (kind == "replace" and not absent):
    st.caption("No suitable items for this type of change in the current case. Add them on the Cases screen.")
elif st.button("Add this comparison", key="w_add"):
    op = {"add": item, "remove": item, "variant": {item: value}, "replace": {item: value},
          "position": {item: value}}[kind]
    factor, level = factor_level(kind, item, value)
    base_id = re.sub(r"[^A-Za-z0-9]+", "-", f"{kind}-{item}-{value or ''}").strip("-").lower()
    cid, n = base_id, 2
    while cid in conds_now:
        cid, n = f"{base_id}-{n}", n + 1
    new = {"id": cid, "factor": factor, "level": str(level), "ref": ref, "ops": [{kind: op}],
           "exploratory": kind == "add" and case.item(item).attributes.admissibility == "inadmissible_heard"}
    try:
        check(draft_conditions(new))
        changes.append(new)
        st.rerun()
    except (ConditionError, ValueError) as e:
        st.error(f"This comparison can't be added: {e}")

for k, ch in enumerate(changes):
    cols = st.columns([8, 1])
    cond = Condition(**ch)
    cols[0].markdown(f"{k + 1}. {common.describe_change(case, cond)}"
                     + ("" if cond.ref == "baseline" else f" *(compared with '{cond.ref}')*"))
    if cols[1].button("Remove", key=f"w_del_{ch['id']}"):
        changes.pop(k)
        dependants = [x for x in changes if x["ref"] == ch["id"]]
        for x in dependants:
            changes.remove(x)
        st.rerun()
if not changes:
    st.caption("No comparisons yet. Add at least one change above.")

st.markdown("#### Step 4 - Choose the AI")
ok, info = ollama_status() if not common.practice() else (True, "practice")
model_cfg = None
if common.practice():
    st.info("Practice data is selected: a simulated AI answers instantly. Switch to real research data in "
            "Models & Settings to use a local model.")
    model_cfg = {"key": "simulated", "provider": "fake", "model": "simulated", "max_tokens": 600}
elif not ok:
    st.error(f"{info}")
    st.caption("Start the Ollama app, then reload this page. A cloud service is never used instead.")
else:
    models = ollama_models()
    if not models:
        st.warning("No local models are installed. Install one from a terminal, e.g. `ollama pull qwen3:4b-instruct`.")
    else:
        names = [m["name"] for m in models]
        default = names.index("qwen3:4b-instruct") if "qwen3:4b-instruct" in names else 0
        fits = [common.model_fit(m["size_gb"]) for m in models]
        pick = st.radio("Local AI model", names, index=default, key="w_model",
                        captions=[f"{m['parameters'] or '?'} parameters · {m['size_gb']} GB · {f[0]} {f[1]}"
                                  for m, f in zip(models, fits)])
        st.caption("Models differ in speed and in how carefully they analyse a case; results are reported per model.")
        d = common.defaults()
        model_cfg = {"key": re.sub(r"[^A-Za-z0-9]+", "-", pick).strip("-"), "provider": "ollama", "model": pick,
                     "num_ctx": d["num_ctx"], "max_tokens": d["max_tokens"], "timeout_s": d["timeout_s"]}
        if d["temperature"]:
            model_cfg["temperature"] = float(d["temperature"])
        with st.expander("Generation settings"):
            st.write(f"Maximum response length {d['max_tokens']} tokens · context size {d['num_ctx']} tokens · "
                     f"time limit {d['timeout_s']} s per analysis · temperature "
                     f"{d['temperature'] or 'model default'} · {d['parallel']} analysis at a time.")
            st.page_link("ui/settings.py", label="Change these in Models & Settings", icon=":material/tune:")

st.markdown("#### Step 5 - Run the experiment")
repeats = st.number_input("Repeated analyses per scenario", 1, 50, 5, key="w_repeats", help=common.HELP["repeats"])
st.caption(common.HELP["repeats"])
n_total = (len(changes) + 1) * repeats
secs = 0.1 if common.practice() else 20
if model_cfg and model_cfg["provider"] == "ollama":
    past = c.execute("SELECT AVG((julianday(a.finished_at) - julianday(a.started_at)) * 86400) FROM attempts a "
                     "WHERE a.model_returned LIKE ? AND a.parse_status != 'api_error'",
                     (model_cfg["model"] + "%",)).fetchone()[0]
    secs = past or secs
st.write(f"This will run **{n_total} analyses** ({len(changes) + 1} scenarios × {repeats} repeats), "
         f"taking roughly **{max(1, round(n_total * secs / 60))} minutes** on this computer.")
exp_name = st.text_input("Experiment name", f"{case_id}-{datetime.now():%Y%m%d-%H%M}", key="w_name",
                         help="Letters, numbers, dots, dashes and underscores.")
if st.button("Start analyses", type="primary", disabled=not (changes and model_cfg), key="w_start"):
    taken = set(common.experiments(c).experiment_id) | {p.stem for p in common.EXPERIMENTS_DIR.glob("*.yaml")}
    if not re.fullmatch(r"[A-Za-z0-9_.-]+", exp_name or ""):
        st.error("Please use only letters, numbers, dots, dashes and underscores in the name.")
    elif exp_name in taken:
        st.error("An experiment with this name already exists. Choose another name.")
    else:
        spec = {"experiment_id": exp_name, "cases": [case_id], "prompt_version": "v2", "models": [model_cfg],
                "runs": int(repeats), "seed": random.randrange(1, 10 ** 9), "order": "randomize_within_side",
                "budget": {"max_usd": 0, "max_calls": n_total * 3},
                "conditions": [k.model_dump(exclude_defaults=True) | {"id": k.id} for k in draft_conditions()]}
        try:
            Experiment.model_validate(spec)
            path = common.EXPERIMENTS_DIR / f"{exp_name}.yaml"
            common.write_yaml(path, spec)
            runner.plan(c, path)
            start(exp_name, common.defaults()["parallel"])
            st.session_state["w_changes"] = []
            st.session_state["results_exp"] = exp_name
            st.success(f"Started '{exp_name}'. Progress is shown under 'Your experiments' above; results appear "
                       "on the Results screen as analyses finish.")
            st.rerun()
        except (ConditionError, ValueError, RuntimeError) as e:
            st.error(f"The experiment could not be started: {e}")

st.markdown("#### Step 6 - Review the results")
st.page_link("ui/results.py", label="Open Results", icon=":material/fact_check:")
