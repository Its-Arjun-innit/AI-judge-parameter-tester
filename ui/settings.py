"""Models & Settings: local AI status, installed models, data set, and advanced details."""
import os

import pandas as pd
import streamlit as st

from judgelab.adapters import ProviderError, ollama_models, ollama_show, ollama_status, ollama_url
from judgelab.cases import ROOT
from ui import common

st.title("Models & Settings")

# ---- local AI ----
st.subheader("Local AI (Ollama)")
ok, info = ollama_status()
if ok:
    st.success(f"Ollama is running on this computer (version {info}). Analyses stay on this computer.")
else:
    st.error(f"{info}")
    st.markdown("**To fix this:** start the Ollama app (or run `ollama serve`), then reload this page. "
                "No cloud service is used instead.")
try:
    st.caption(f"Address: {ollama_url()} (set with the OLLAMA_HOST environment variable; only this computer is allowed).")
except ProviderError as e:
    st.error(str(e))

hw = common.hardware()
gpu = f"{hw['gpu']} with {hw['vram_gb']} GB graphics memory" if hw["vram_gb"] else "no NVIDIA graphics card detected"
st.caption(f"This computer: {gpu}, {hw['ram_gb'] or '?'} GB memory.")

if ok:
    models = ollama_models()
    if not models:
        st.info("No models are installed yet. Install one from a terminal, for example "
                "`ollama pull qwen3:4b-instruct` (about 2.5 GB). Models are never downloaded automatically.")
    rows = []
    for m in models:
        try:
            ctx = ollama_show(m["name"])["context_length"]
        except ProviderError:
            ctx = None
        icon, fit = common.model_fit(m["size_gb"])
        rows.append({"Model": m["name"], "Size (GB)": m["size_gb"], "Parameters": m["parameters"],
                     "Max case length (tokens)": ctx, "On this computer": f"{icon} {fit}"})
    if rows:
        st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")
        st.caption("Larger models are usually more capable but slower; models that do not fit in graphics "
                   "memory run partly on the processor and can take minutes per analysis. Speed and analytical "
                   "quality vary between models, so results from different models are compared, not pooled.")

# ---- data set ----
st.subheader("Research data")
choice = st.radio("Which data do you want to work with?",
                  ["Real research data", "Practice data (simulated AI, for learning the workflow)"],
                  index=1 if common.practice() else 0, key="data_choice")
st.session_state["db_name"] = common.PRACTICE_DB if choice.startswith("Practice") else common.REAL_DB

# ---- advanced ----
st.subheader("Advanced")
with st.expander("Default settings for new experiments"):
    d = common.defaults()  # stored outside widget state so it survives switching screens
    d["max_tokens"] = st.number_input(
        "Maximum response length (tokens)", 200, 4000, d["max_tokens"], 100,
        help="The AI stops writing after this many tokens (about 0.75 words each). "
             "Responses that hit the limit are recorded as 'cut off'.")
    d["num_ctx"] = st.number_input(
        "Context size (tokens)", 2048, 32768, d["num_ctx"], 1024,
        help="How much text the local model can hold at once: the case plus the answer must fit. "
             "Larger values use more memory. Over-long cases are flagged, never silently cut.")
    d["timeout_s"] = st.number_input("Time limit per analysis (seconds)", 30, 3600, d["timeout_s"], 30,
                                     help="A timed-out analysis is retried up to 3 times, then recorded as failed.")
    t = st.text_input("Temperature (leave blank for the model's default)", d["temperature"],
                      help="Higher values make answers more varied; lower values more repetitive. "
                           "Leaving it blank uses the model's own default, which is recorded either way.")
    try:
        d["temperature"] = "" if not t.strip() else str(float(t))
    except ValueError:
        st.error("Temperature must be a number such as 0.7, or blank.")
    d["parallel"] = st.number_input("Analyses at the same time", 1, 2, d["parallel"],
                                    help="Keep this at 1 on a personal computer. 2 can be faster with spare "
                                         "graphics memory, but may slow everything down or fail.")

with st.expander("Instructions given to the AI"):
    versions = sorted({p.stem.split("_")[1] for p in (ROOT / "prompts").glob("system_*.md")})
    v = st.selectbox("Version", versions, index=len(versions) - 1)
    st.markdown("**General instructions (system)**")
    st.code((ROOT / "prompts" / f"system_{v}.md").read_text(encoding="utf-8"), language=None)
    st.markdown("**Case record template** (filled with each case's charge, law, facts and evidence)")
    st.code((ROOT / "prompts" / f"user_{v}.md").read_text(encoding="utf-8"), language=None)

with st.expander("Cloud AI providers (optional)"):
    st.warning("Cloud models send the fictional case text to that company and may cost money. "
               "They are only used when an experiment explicitly selects them.")
    for name, var in [("Anthropic (Claude)", "ANTHROPIC_API_KEY"), ("OpenAI (GPT)", "OPENAI_API_KEY"),
                      ("Google (Gemini)", "GEMINI_API_KEY")]:
        st.write(f"{name}: {'key set' if os.environ.get(var) else 'no key'}")

with st.expander("Run log and errors"):
    logs = sorted((common.DATA_DIR / "logs").glob("*.log"), key=lambda p: p.stat().st_mtime, reverse=True)
    if logs:
        pick = st.selectbox("Log", logs, format_func=lambda p: p.stem)
        st.code(pick.read_text(encoding="utf-8", errors="replace")[-6000:] or "(empty)", language=None)
    errs = pd.read_sql_query("SELECT t.experiment_id, a.trial_id, a.finished_at, a.error FROM attempts a "
                             "JOIN trials t USING (trial_id) WHERE a.error IS NOT NULL "
                             "ORDER BY a.attempt_id DESC LIMIT 30", common.con())
    st.dataframe(errs, hide_index=True, width="stretch")
