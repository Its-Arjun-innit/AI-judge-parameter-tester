"""Research Dashboard. Run: streamlit run app.py
Everything shown is an observed response difference under controlled conditions,
not a measure of how the model weighs evidence internally."""
import json
import sqlite3

import altair as alt
import pandas as pd
import streamlit as st

from judgelab import analysis
from judgelab.cases import ROOT

# Categorical slots 1-3 (validated all-pairs for 3 series); the table view is the relief for aqua's contrast.
SERIES = ["#2a78d6", "#eb6834", "#1baf7a"]

st.set_page_config(page_title="Evidence Sensitivity Lab", layout="wide")
dbs = sorted((ROOT / "data").glob("*.sqlite"))
if not dbs:
    st.info("No database yet. Run `python -m judgelab plan <spec>` first.")
    st.stop()
db = st.sidebar.selectbox("Database", dbs, format_func=lambda p: p.name)
con = sqlite3.connect(db, check_same_thread=False)
exps = [r[0] for r in con.execute("SELECT experiment_id FROM experiments ORDER BY created_at")]
if not exps:
    st.info("No experiments in this database.")
    st.stop()
exp_id = st.sidebar.selectbox("Experiment", exps)
if "dryrun" in db.name:
    st.sidebar.warning("Dry-run database: responses come from a fake model.")

obs = analysis.observations(con, exp_id)
models = sorted(obs.model_key.unique())
colors = alt.Scale(domain=models, range=SERIES[:len(models)])  # color follows the model, not its rank

st.title(exp_id)
tabs = st.tabs(["Overview", "Condition comparison", "Consistency", "Cases & prompts", "Responses", "Export"])

with tabs[0]:
    done = obs[obs.status == "done"]
    c = st.columns(5)
    c[0].metric("Cases", obs.case_id.nunique())
    c[1].metric("Conditions", obs.condition_id.nunique())
    c[2].metric("Models", len(models))
    c[3].metric(f"Trials done (of {len(obs)})", len(done))
    c[4].metric("Spend (USD)", f"{obs.cost_usd.fillna(0).sum():.2f}")
    st.subheader("Run monitor")
    st.dataframe(obs.pivot_table(index="model_key", columns="status", values="trial_id", aggfunc="count",
                                 fill_value=0), use_container_width=True)
    st.caption("Outcome of the final attempt per completed trial (refusals and invalid outputs are kept, not dropped).")
    st.dataframe(done.pivot_table(index="model_key", columns="parse_status", values="trial_id", aggfunc="count",
                                  fill_value=0), use_container_width=True)
    att = pd.read_sql_query("SELECT a.trial_id, a.attempt_no, a.http_status, a.error, a.finished_at FROM attempts a "
                            "JOIN trials t USING (trial_id) WHERE t.experiment_id=? AND a.error IS NOT NULL "
                            "ORDER BY a.finished_at DESC LIMIT 50", con, params=(exp_id,))
    st.subheader(f"API errors (latest {len(att)})")
    st.dataframe(att, use_container_width=True, hide_index=True)

with tabs[1]:
    st.caption("Observed shift in the model's stated probability (percentage points) relative to each "
               "condition's reference, with a 95% bootstrap interval over runs. These are response "
               "differences, not evidence weights.")
    cmp = analysis.compare(obs)
    case = st.selectbox("Case", sorted(cmp.case_id.unique()))
    d = cmp[(cmp.case_id == case) & cmp.delta_pp.notna()].copy()
    d["condition"] = d.condition_id
    order = d.sort_values(["factor", "condition_id"]).condition.unique().tolist()
    base = alt.Chart(d).encode(
        y=alt.Y("condition:N", sort=order, title=None, axis=alt.Axis(labelLimit=220)),
        yOffset=alt.YOffset("model_key:N", scale=alt.Scale(domain=models)),
        color=alt.Color("model_key:N", scale=colors, title="Model"),
        tooltip=["condition_id", alt.Tooltip("ref", title="compared with"), "factor", "level", "model_key",
                 alt.Tooltip("delta_pp:Q", format="+.1f", title="shift (pp)"),
                 alt.Tooltip("ci_lo:Q", format="+.1f"), alt.Tooltip("ci_hi:Q", format="+.1f"),
                 alt.Tooltip("flip_rate:Q", format=".0%"), "n_valid"])
    whisk = base.mark_rule(strokeWidth=2).encode(x=alt.X("ci_lo:Q", title="Shift in stated probability (pp)"),
                                                 x2="ci_hi:Q")
    dots = base.mark_circle(size=80, opacity=1).encode(x="delta_pp:Q")
    zero = alt.Chart(pd.DataFrame({"x": [0]})).mark_rule(color="#898781").encode(x="x:Q")
    st.altair_chart((zero + whisk + dots).properties(height=max(300, 34 * len(order))),
                    use_container_width=True)
    st.subheader("Table view")
    st.dataframe(cmp[cmp.case_id == case].round(2), use_container_width=True, hide_index=True)
    ac = analysis.across_cases(cmp)
    if (ac.k_cases >= 2).any():
        st.subheader("Across cases (case-level deltas; cluster bootstrap CI, sign-flip p)")
        st.dataframe(ac.round(3), use_container_width=True, hide_index=True)

with tabs[2]:
    st.caption("Repeatability of identical prompts. Within-cell SD of stated probability; share of runs "
               "agreeing with the cell's modal verdict; Fleiss' kappa with cells as subjects, runs as raters.")
    st.dataframe(analysis.consistency(obs).round(3), use_container_width=True, hide_index=True)
    cells = obs[obs.parse_status == "valid"].groupby(["case_id", "condition_id", "model_key"]).probability \
        .agg(["count", "mean", "std", "min", "max"]).reset_index()
    st.dataframe(cells.round(2), use_container_width=True, hide_index=True)

with tabs[3]:
    cases = pd.read_sql_query("SELECT DISTINCT c.case_id, c.case_hash, c.yaml FROM cases c JOIN trials t "
                              "USING (case_hash) WHERE t.experiment_id=?", con, params=(exp_id,))
    for _, r in cases.iterrows():
        data = json.loads(r.yaml)
        with st.expander(f"{r.case_id}  ({r.case_hash[:12]})", expanded=True):
            st.markdown(f"**Charge.** {data['law']['offence']}")
            st.dataframe(pd.DataFrame([{k: e.get(k) for k in ("id", "side", "kind", "category", "direction",
                                                               "target_fact")} | e.get("attributes", {})
                                       for e in data["evidence"]]), use_container_width=True, hide_index=True)
            st.json(data.get("researcher_notes", {}), expanded=False)
    pick = st.selectbox("Show the exact prompt of trial", obs.trial_id)
    t = obs[obs.trial_id == pick].iloc[0]
    st.caption(f"{t.case_id} / {t.condition_id} / {t.model_key} / run {t.run_index} - labels: {t.label_map_json}")
    st.code(con.execute("SELECT user_prompt FROM trials WHERE trial_id=?", (int(pick),)).fetchone()[0],
            language=None)

with tabs[4]:
    st.warning("Model-generated text - not evidence of the model's internal reasoning.")
    f = obs[obs.status == "done"]
    cond = st.selectbox("Condition", sorted(f.condition_id.unique()))
    model = st.selectbox("Model", models)
    for _, r in f[(f.condition_id == cond) & (f.model_key == model)].head(20).iterrows():
        with st.expander(f"run {r.run_index}: {r.parse_status} | P={r.probability} | {r.verdict} | conf {r.confidence}"):
            st.write(r.reasoning)
            st.caption(f"Standard applied: {r.legal_standard}")
            st.caption(f"Key evidence: {r.key_evidence_json}")
            st.caption(f"Missing information: {r.missing_info_json}")

with tabs[5]:
    st.download_button("Observations (CSV, one row per trial)", obs.to_csv(index=False), f"{exp_id}_observations.csv")
    st.download_button("Observations (JSON)", obs.to_json(orient="records"), f"{exp_id}_observations.json")
    att = pd.read_sql_query("SELECT a.* FROM attempts a JOIN trials t USING (trial_id) WHERE t.experiment_id=?",
                            con, params=(exp_id,))
    st.download_button("All attempts incl. failures (CSV)", att.to_csv(index=False), f"{exp_id}_attempts.csv")
    st.download_button("Condition comparison (CSV)", analysis.compare(obs).to_csv(index=False),
                       f"{exp_id}_comparison.csv")
