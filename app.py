"""Evidence Sensitivity Lab - researcher interface. Run: streamlit run app.py
Everything shown is an observed response difference under controlled conditions,
not a measure of how the model weighs evidence internally."""
import streamlit as st

from judgelab.__main__ import load_env
from judgelab.adapters import ollama_status
from ui import common

load_env()  # cloud keys from the gitignored .env, only if a cloud model is chosen
st.set_page_config(page_title="Evidence Sensitivity Lab", page_icon=":material/balance:", layout="wide")

pages = [
    st.Page("ui/overview.py", title="Overview", icon=":material/dashboard:", default=True),
    st.Page("ui/cases.py", title="Cases", icon=":material/gavel:"),
    st.Page("ui/experiments.py", title="Experiments", icon=":material/science:"),
    st.Page("ui/results.py", title="Results", icon=":material/fact_check:"),
    st.Page("ui/settings.py", title="Models & Settings", icon=":material/tune:"),
]
nav = st.navigation(pages)

with st.sidebar:
    ok, info = ollama_status()
    st.caption(f"Local AI (Ollama): {'🟢 running' if ok else '🔴 not running'}")
    if common.practice():
        st.warning("Practice data: responses come from a simulated AI. Switch in Models & Settings.")

nav.run()
