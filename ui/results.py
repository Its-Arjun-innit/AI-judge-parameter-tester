"""Results: before-and-after cards for each evidence change, with audit details one click away."""
import json

import altair as alt
import numpy as np
import pandas as pd
import streamlit as st
import yaml

from judgelab import analysis
from judgelab.cases import template
from ui import common

st.title("Results")
c = common.con()
exps = common.experiments(c)
exps = exps[exps.done.fillna(0) > 0]
if exps.empty:
    st.info("No completed analyses yet. Design and run an experiment first.")
    st.page_link("ui/experiments.py", label="Go to Experiments", icon=":material/science:")
    st.stop()

ids = exps.experiment_id.tolist()
want = st.session_state.get("results_exp")
exp_id = st.selectbox("Experiment", ids, index=ids.index(want) if want in ids else 0)
st.session_state["results_exp"] = exp_id
exp, obs, cases = common.experiment_bundle(c, exp_id)  # cases: the exact versions this experiment used
conds = {k.id: k for k in exp.conditions}

models = sorted(obs.model_key.unique())
model = st.radio("AI model", models, horizontal=True) if len(models) > 1 else models[0]
o = obs[obs.model_key == model]
done = o[o.status == "done"]
valid = done[done.parse_status == "valid"]
returned = done.model_returned.dropna().unique()
st.caption(f"{len(done)} of {len(o)} analyses complete · {len(valid)} interpreted · "
           f"model: {', '.join(returned) if len(returned) else model}")
if (o.status == "pending").any():
    st.info(f"{(o.status == 'pending').sum()} analyses are still waiting to run. Continue them on the Experiments screen.")


def summary(rows: pd.DataFrame) -> str:
    v = rows[rows.parse_status == "valid"]
    if v.empty:
        return "No interpretable responses yet."
    n = len(v)
    counts = v.verdict.value_counts()
    verdicts = ", ".join(f"{k} in {counts[k]} of {n}" for k in counts.index).capitalize()
    p = v.probability
    return (f"{verdicts}  \nStated likelihood of guilt: **{p.mean():.0f}** on average (range {p.min():.0f}-{p.max():.0f})"
            f"  \nConfidence in verdict: {v.confidence.median():.0f} of 7 (median)")


def cited_often(rows: pd.DataFrame) -> set:
    v = rows[rows.parse_status == "valid"]
    counts = {}
    for s in v.key_evidence_json.dropna():
        for item in {k["item_id"] for k in json.loads(s) if k["item_id"]}:
            counts[item] = counts.get(item, 0) + 1
    return {i for i, k in counts.items() if k >= len(v) / 2}


def audit(rows: pd.DataFrame):
    first = rows.iloc[0]
    tid = int(first.trial_id)
    st.markdown("**Instructions given to the AI (general)**")
    st.code(template("system", first.prompt_version), language=None)
    st.markdown("**Case record exactly as shown to the AI (first analysis)**")
    st.code(c.execute("SELECT user_prompt FROM trials WHERE trial_id=?", (tid,)).fetchone()[0], language=None)
    sent = c.execute("SELECT request_json FROM attempts WHERE trial_id=? ORDER BY attempt_id DESC LIMIT 1",
                     (tid,)).fetchone()
    st.markdown("**Settings requested** (experiment design) and **actually sent** (first analysis)")
    st.json({"requested": json.loads(first.params_json),
             "sent": json.loads(sent[0]) if sent and sent[0] else "not recorded (older analysis)"}, expanded=False)
    raw = pd.read_sql_query(
        f"SELECT t.run_index AS repeat, a.attempt_no, a.parse_status, a.model_returned, a.error, a.raw_text "
        f"FROM attempts a JOIN trials t USING (trial_id) WHERE t.trial_id IN ({','.join('?' * len(rows))}) "
        f"ORDER BY t.run_index, a.attempt_no", c, params=[int(x) for x in rows.trial_id])
    st.markdown("**Raw responses and errors**")
    st.dataframe(raw, hide_index=True, width="stretch")


cmp = analysis.compare(o)

# ---- overview chart ----
d = cmp[cmp.delta_pp.notna()].copy()
if not d.empty:
    st.subheader("Change in the AI's stated likelihood of guilt")
    st.caption(f"{common.MEASUREMENT_NOTE} Dots: average change in percentage points; bars: 95% range over "
               "repeated analyses (wider bars = less certain). Zero means no change.")
    order = [k for k in conds if k in set(d.condition_id)]
    base = alt.Chart(d).encode(
        y=alt.Y("condition_id:N", sort=order, title=None, axis=alt.Axis(labelLimit=220)),
        tooltip=[alt.Tooltip("condition_id", title="scenario"), alt.Tooltip("ref", title="compared with"),
                 alt.Tooltip("delta_pp:Q", format="+.1f", title="change (points)"),
                 alt.Tooltip("ci_lo:Q", format="+.1f", title="range from"), alt.Tooltip("ci_hi:Q", format="+.1f", title="to"),
                 alt.Tooltip("n_valid", title="interpreted analyses")])
    chart = (alt.Chart(pd.DataFrame({"x": [0]})).mark_rule(color="#898781").encode(x="x:Q")
             + base.mark_rule(strokeWidth=2, color="#2a78d6").encode(
                 x=alt.X("ci_lo:Q", title="Change in stated likelihood of guilt (points)"), x2="ci_hi:Q")
             + base.mark_circle(size=90, opacity=1, color="#2a78d6").encode(x="delta_pp:Q"))
    st.altair_chart(chart.properties(height=max(160, 40 * len(order))), width="stretch")

# ---- one card per evidence change ----
st.subheader("What each change did")
for case_id, case in cases.items():
    oc = o[o.case_id == case_id]
    if len(cases) > 1:
        st.markdown(f"### Case {case_id}")
    with st.container(border=True):
        st.markdown("**Original evidence set**")
        st.markdown(summary(oc[(oc.condition_id == "baseline") & (oc.status == "done")]))
    for i, (cid, cond) in enumerate([(k, v) for k, v in conds.items() if k != "baseline"], 1):
        after = oc[(oc.condition_id == cid) & (oc.status == "done")]
        before = oc[(oc.condition_id == cond.ref) & (oc.status == "done")]
        row = cmp[(cmp.case_id == case_id) & (cmp.condition_id == cid)]
        with st.container(border=True):
            st.markdown(f"**{i}. What changed:** {common.describe_change(case, cond)}")
            ref_name = "the original evidence set" if cond.ref == "baseline" else f"scenario '{cond.ref}'"
            st.caption(f"Compared with {ref_name}.{' Exploratory probe.' if cond.exploratory else ''}")
            if after.empty:
                st.info("Not analysed yet.")
                continue
            b, a, m = st.columns(3)
            b.markdown("**Before**  \n" + summary(before))
            a.markdown("**After**  \n" + summary(after))
            r = row.iloc[0] if not row.empty else None
            lines = []
            if r is not None and not np.isnan(r.delta_pp):
                rng = (f" (95% range {r.ci_lo:+.0f} to {r.ci_hi:+.0f})" if not np.isnan(r.ci_lo)
                       else " (too few analyses to estimate a range)")
                lines.append(f"Likelihood of guilt: **{r.delta_pp:+.0f} points**{rng}")
                va = after[after.parse_status == "valid"]
                lines.append(f"Verdict differed from the usual 'before' verdict in {round(r.flip_rate * len(va))} of {len(va)}")
                if len(va) > 1:
                    agree = va.verdict.value_counts(normalize=True).iloc[0]
                    lines.append(f"Consistency: repeats agreed on the verdict {agree:.0%} of the time; "
                                 f"likelihood varied by ±{va.probability.std(ddof=1):.0f} points")
            new, gone = cited_often(after) - cited_often(before), cited_often(before) - cited_often(after)
            if new:
                lines.append("Now usually cited: " + "; ".join(common.item_name(case, x, 40) for x in new))
            if gone:
                lines.append("No longer usually cited: " + "; ".join(common.item_name(case, x, 40) for x in gone))
            m.markdown("**Measured change**  \n" + "  \n".join(lines or ["Not enough interpreted analyses."]))
            bad = after[after.parse_status != "valid"]
            if not bad.empty:
                st.warning(" · ".join(f"{k} of {len(after)}: {common.label(common.OUTCOME, s)}"
                                      for s, k in bad.parse_status.value_counts().items()))
            with st.expander("The AI's own explanation (its words, not a measurement)"):
                for title, rows in (("Before", before), ("After", after)):
                    st.markdown(f"**{title}**")
                    for _, x in rows[rows.parse_status == "valid"].head(3).iterrows():
                        st.markdown(f"- Repeat {x.run_index + 1}: *{x.verdict}*, likelihood {x.probability:.0f}. {x.reasoning}")
            with st.expander("Audit details"):
                audit(after)

with st.expander("Consistency across repeated analyses"):
    st.caption("How much the AI's answers vary when the identical case is analysed again. Lower spread and higher "
               "agreement mean more consistent answers.")
    cons = analysis.consistency(o).rename(columns={
        "model_key": "AI model", "cells": "scenarios", "runs_per_cell_used": "repeats used",
        "mean_sd_p": "typical spread of likelihood (points)", "mean_modal_agreement": "agreement on verdict",
        "fleiss_kappa_verdict": "Fleiss' kappa (verdict)"})
    st.dataframe(cons.round(2), hide_index=True, width="stretch")

with st.expander("Download data"):
    st.download_button("All analyses (CSV, one row per analysis)", obs.to_csv(index=False), f"{exp_id}_observations.csv")
    att = pd.read_sql_query("SELECT a.* FROM attempts a JOIN trials t USING (trial_id) WHERE t.experiment_id=?",
                            c, params=(exp_id,))
    st.download_button("Every attempt, including failures (CSV)", att.to_csv(index=False), f"{exp_id}_attempts.csv")
    st.download_button("Before-and-after comparisons (CSV)", analysis.compare(obs).to_csv(index=False),
                       f"{exp_id}_comparison.csv")
    st.download_button("Experiment design (YAML)", yaml.safe_dump(exp.model_dump(mode="json"), sort_keys=False),
                       f"{exp_id}.yaml")
