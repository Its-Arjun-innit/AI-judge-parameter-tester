"""Overview: what has been studied, which evidence changes moved the AI most, consistency, what needs review."""
import altair as alt
import numpy as np
import pandas as pd
import streamlit as st

from judgelab import analysis, runner
from ui import common

st.title("Research overview")
c = common.con()
exps = common.experiments(c)
if exps.empty or not exps.done.fillna(0).sum():
    st.info("Nothing has been analysed yet. A typical study: write a fictional case, choose one piece of evidence "
            "to change, and let a local AI analyse both versions several times.")
    st.page_link("ui/cases.py", label="Step 1: create or review a case", icon=":material/gavel:")
    st.page_link("ui/experiments.py", label="Step 2: design and run an experiment", icon=":material/science:")
    st.stop()

done_cases = c.execute("SELECT COUNT(DISTINCT case_id) FROM trials WHERE status='done'").fetchone()[0]
complete = exps[(exps.pending.fillna(0) == 0) & (exps.done.fillna(0) > 0)]
m = st.columns(4)
m[0].metric("Cases studied", done_cases)
m[1].metric("Experiments completed", len(complete))
m[2].metric("Experiments with analyses still to run",
            int(((exps.pending.fillna(0) > 0) & (exps.done.fillna(0) > 0)).sum()))
m[3].metric("Analyses completed", int(exps.done.fillna(0).sum()))

effects, review, cons = [], [], []
for exp_id in exps[exps.done.fillna(0) > 0].experiment_id:
    exp, obs, cases = common.experiment_bundle(c, exp_id)
    conds = {k.id: k for k in exp.conditions}
    cmp = analysis.compare(obs)
    n_valid = cmp.set_index(["case_id", "model_key", "condition_id"]).n_valid
    for r in cmp[cmp.delta_pp.notna()].itertuples():
        n_ref = n_valid.get((r.case_id, r.model_key, r.ref), 0)
        what = common.describe_change(cases[r.case_id], conds[r.condition_id])
        where = f"{exp_id} / {r.model_key}"
        width = r.ci_hi - r.ci_lo if not np.isnan(r.ci_lo) else np.inf
        if r.n_valid >= 3 and n_ref >= 3:  # rank only well-measured changes
            effects.append({"experiment": exp_id, "AI model": r.model_key, "case": r.case_id, "change": what,
                            "analyses (before/after)": f"{n_ref}/{r.n_valid}", "delta": r.delta_pp,
                            "lo": r.ci_lo, "hi": r.ci_hi})
        if r.invalid_rate > 0.1:
            review.append((exp_id, f"{r.invalid_rate:.0%} of responses could not be interpreted or were declined "
                                   f"({where}: {r.condition_id})."))
        if width > 30:
            review.append((exp_id, f"Very uncertain result for '{r.condition_id}' ({where}): add repeats or cases."))
        focal = r.focal_item if isinstance(r.focal_item, str) else None
        if focal and r.factor in ("presence", "corroboration") and not np.isnan(r.ci_lo):
            direction = cases[r.case_id].item(focal).direction
            if (direction == "inculpatory" and r.ci_hi < 0) or (direction == "exculpatory" and r.ci_lo > 0):
                review.append((exp_id, f"Adding evidence that points one way moved the AI the other way "
                                       f"('{r.condition_id}', {where}). Worth a close reading of its explanations."))
        if focal and not np.isnan(r.cited_rate) and r.cited_rate < 0.3:
            review.append((exp_id, f"The AI rarely cited the changed evidence ('{r.condition_id}', {where}: "
                                   f"{r.cited_rate:.0%} of analyses)."))
    row = exps.set_index("experiment_id").loc[exp_id]
    if row.pending and not runner.running_pid(common.db_path(), exp_id):
        review.append((exp_id, f"{int(row.pending)} analyses have not run yet (stopped or paused)."))
    if row.failed:
        review.append((exp_id, f"{int(row.failed)} analyses failed. See the run log in Models & Settings."))
    cons.append(analysis.consistency(obs).assign(experiment=exp_id))

st.subheader("Which evidence changes moved the AI's judgment most?")
st.caption(f"{common.MEASUREMENT_NOTE} Each row is one change in one case, measured against its own comparison "
           "scenario, so different cases are never averaged together. Bars show the 95% range over repeated "
           "analyses; only changes with at least 3 interpreted analyses on each side are shown.")
if effects:
    e = pd.DataFrame(effects)
    e = e.reindex(e.delta.abs().sort_values(ascending=False).index).head(12)
    e["label"] = e.experiment + " · " + e["AI model"] + " · " + e.change.str.slice(0, 70)
    base = alt.Chart(e).encode(
        y=alt.Y("label:N", sort=list(e.label), title=None, axis=alt.Axis(labelLimit=420)),
        tooltip=["experiment", "AI model", "case", "change", "analyses (before/after)",
                 alt.Tooltip("delta:Q", format="+.1f", title="change (points)"),
                 alt.Tooltip("lo:Q", format="+.1f", title="range from"), alt.Tooltip("hi:Q", format="+.1f", title="to")])
    chart = (alt.Chart(pd.DataFrame({"x": [0]})).mark_rule(color="#898781").encode(x="x:Q")
             + base.mark_rule(strokeWidth=2, color="#2a78d6").encode(
                 x=alt.X("lo:Q", title="Change in stated likelihood of guilt (points)"), x2="hi:Q")
             + base.mark_circle(size=90, opacity=1, color="#2a78d6").encode(x="delta:Q"))
    st.altair_chart(chart.properties(height=max(160, 36 * len(e))), width="stretch")
    with st.expander("Table view"):
        st.dataframe(e.drop(columns=["label"]).rename(columns={"delta": "change (points)", "lo": "range from",
                                                              "hi": "range to"}).round(1),
                     hide_index=True, width="stretch")
else:
    st.caption("No change has at least 3 interpreted analyses on each side yet.")

st.subheader("How consistent are the AI's answers?")
st.caption("When the identical case is analysed again, does the AI reach the same verdict and a similar likelihood?")
if cons:
    cc = pd.concat(cons)
    if not cc.empty:
        st.dataframe(cc.rename(columns={"model_key": "AI model", "cells": "scenarios",
                                        "runs_per_cell_used": "repeats", "mean_sd_p": "typical spread (points)",
                                        "mean_modal_agreement": "agreement on verdict",
                                        "fleiss_kappa_verdict": "Fleiss' kappa"})
                     [["experiment", "AI model", "scenarios", "repeats", "agreement on verdict",
                       "typical spread (points)", "Fleiss' kappa"]].round(2),
                     hide_index=True, width="stretch")
        st.caption("Agreement: share of repeats giving the most common verdict. Fleiss' kappa is blank when every "
                   "repeat gave the same verdict (perfect agreement leaves nothing to compare).")

st.subheader("Needs review")
if not review:
    st.caption("Nothing flagged.")
for k, (exp_id, msg) in enumerate(review):
    cols = st.columns([8, 1])
    cols[0].markdown(f"- **{exp_id}:** {msg}")
    if cols[1].button("Open", key=f"rev-{k}"):
        st.session_state["results_exp"] = exp_id
        st.switch_page("ui/results.py")
