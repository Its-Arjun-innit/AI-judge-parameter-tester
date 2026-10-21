import copy

import pytest
import yaml

from judgelab.cases import ROOT, load_case, render
from judgelab.conditions import ConditionError, assert_single_change, expand, presentation, trial_rng
from judgelab.runner import load_experiment
from judgelab.schema import Case, Experiment

CASE_PATH = ROOT / "cases" / "IN-THEFT-001.yaml"


def raw_case():
    return yaml.safe_load(CASE_PATH.read_text(encoding="utf-8"))


def test_good_case_loads():
    assert load_case(CASE_PATH).case_id == "IN-THEFT-001"


def _break_dup(c): c["evidence"].append(copy.deepcopy(c["evidence"][0]))
def _break_baseline(c): c["baseline"].append("E99")
def _break_two_changes(c): c["evidence"][0]["variants"]["low_reliability"]["changes"]["specificity"] = "vague"
def _break_level(c): c["evidence"][0]["variants"]["enmity"]["changes"] = {"potential_bias": "grudge"}
def _break_relation(c): c["evidence"][2]["relations"] = {"corroborates": ["E77"]}


@pytest.mark.parametrize("breaker", [_break_dup, _break_baseline, _break_two_changes, _break_level, _break_relation])
def test_broken_cases_rejected(breaker):
    c = raw_case()
    breaker(c)
    with pytest.raises(ValueError):
        Case.model_validate(c)


def test_pilot_conditions_are_single_changes():
    exp, cases, _ = load_experiment(ROOT / "experiments" / "pilot-01.yaml")
    assert len(expand(exp, cases)) == len(exp.conditions)


def test_two_change_condition_rejected():
    exp, cases, _ = load_experiment(ROOT / "experiments" / "pilot-01.yaml")
    spec = exp.model_dump()
    spec["conditions"].append({"id": "double", "ops": [{"add": "E03"}, {"add": "E05"}]})
    with pytest.raises(ConditionError, match="more than one item"):
        expand(Experiment.model_validate(spec), cases)


def test_frame_change_detected():
    case = load_case(CASE_PATH)
    a = render(case, [("E01", "base"), ("E06", "base")], "v1")
    b = render(case.model_copy(update={"accused_statement": "Changed."}), [("E01", "base"), ("E03", "base"), ("E06", "base")], "v1")
    with pytest.raises(ConditionError, match="non-evidence"):
        assert_single_change(a, b)


def test_presentation_order_and_pins():
    exp, cases, _ = load_experiment(ROOT / "experiments" / "pilot-01.yaml")
    case, states = cases[0], expand(exp, cases)
    for run in range(10):
        order = presentation(case, states[case.case_id, "E01_last"], True, trial_rng(1, case.case_id, "x", run))
        sides = [case.item(i).side for i, _ in order]
        assert sides == sorted(sides, key=lambda s: s != "prosecution")  # prosecution first
        assert [i for i, _ in order if case.item(i).side == "prosecution"][-1] == "E01"


def test_labels_follow_presentation():
    case = load_case(CASE_PATH)
    p = render(case, [("E04", "base"), ("E01", "base"), ("E02", "base"), ("E06", "base")], "v1")
    assert p.label_map == {"PW-1": "E04", "PW-2": "E01", "Ex.P-1": "E02", "DW-1": "E06"}
    assert "researcher" not in p.user.lower() and "ground_truth" not in p.user
