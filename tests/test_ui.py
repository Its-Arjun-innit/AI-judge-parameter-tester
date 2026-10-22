"""Researcher interface, headless (Streamlit AppTest). Uses temporary folders; never touches real data."""
import socket
import sqlite3
import time

import pytest
import yaml
from streamlit.testing.v1 import AppTest

from judgelab.cases import ROOT
from judgelab.schema import Case
from ui import common


def page(name, **state):
    """Open one screen through the real app navigation (links between screens only resolve there)."""
    at = AppTest.from_file(str(ROOT / "app.py"), default_timeout=60)
    for k, v in state.items():
        at.session_state[k] = v
    at.run()
    return at.switch_page(f"ui/{name}.py").run()


def texts(at) -> str:
    return "\n".join(str(x.value) for kind in ("markdown", "caption", "info", "success", "warning", "error", "text")
                     for x in getattr(at, kind))


@pytest.fixture
def sandbox(tmp_path, monkeypatch):
    for name in ("DATA_DIR", "EXPERIMENTS_DIR"):
        (tmp_path / name).mkdir()
        monkeypatch.setattr(common, name, tmp_path / name)
    return tmp_path


def test_settings_says_clearly_when_ollama_is_down(sandbox, monkeypatch):
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    port = s.getsockname()[1]
    s.close()
    monkeypatch.setenv("OLLAMA_HOST", f"127.0.0.1:{port}")
    at = page("settings")
    assert not at.exception
    assert "not running" in texts(at) and "start the ollama app" in texts(at).lower()


def test_create_case_without_technical_settings(sandbox, monkeypatch):
    monkeypatch.setattr(common, "CASES_DIR", sandbox)
    at = page("cases")
    assert not at.exception
    at.text_input(key="1-id").input("TEST-CASE-1")
    at.text_area(key="1-off").input("Theft of a bicycle under BNS s.303")
    at.text_area(key="1-prov").input("BNS s.303 (summary): theft is ...")
    at.button(key="1-add-prosecution").click().run()
    at.text_area(key="1-E01-t").input("A neighbour testified that they saw the accused take the bicycle.")
    at.button(key="1-save").click().run()
    assert not at.exception, at.exception
    saved = Case.model_validate(yaml.safe_load((sandbox / "TEST-CASE-1.yaml").read_text(encoding="utf-8")))
    assert saved.evidence[0].text.startswith("A neighbour") and saved.baseline == ["E01"]


def test_define_comparison_run_and_read_results(sandbox):
    db = {"db_name": common.PRACTICE_DB}  # practice data: simulated AI, no Ollama needed
    at = page("experiments", **db)
    assert not at.exception
    at.selectbox(key="w_case").select("IN-THEFT-001").run()
    at.selectbox(key="w_kind").select("add").run()
    at.button(key="w_add").click().run()
    assert len(at.session_state["w_changes"]) == 1 and "Added" in texts(at)
    at.number_input(key="w_repeats").set_value(3)
    at.text_input(key="w_name").input("ui-test-1")
    at.button(key="w_start").click().run()
    assert not at.exception, at.exception
    assert (sandbox / "EXPERIMENTS_DIR" / "ui-test-1.yaml").exists()

    con = sqlite3.connect(sandbox / "DATA_DIR" / common.PRACTICE_DB)  # background run with the simulated AI
    for _ in range(60):
        if con.execute("SELECT COUNT(*) FROM trials WHERE experiment_id='ui-test-1' AND status='done'").fetchone()[0] == 6:
            break
        time.sleep(0.5)
    else:
        pytest.fail("background run did not finish")

    res = page("results", **db, results_exp="ui-test-1")
    assert not res.exception, res.exception
    main = texts(res)
    assert "What changed:" in main and "Before" in main and "After" in main and "Measured change" in main
    assert '"probability_committed"' not in main  # no raw JSON in the main view
    labels = [e.label for e in res.expander]
    assert "Audit details" in labels and any("own explanation" in x for x in labels)

    over = page("overview", **db)
    assert not over.exception, over.exception
    assert "Experiments completed" in [m.label for m in over.metric]
