import json

import numpy as np
import pandas as pd

from judgelab.analysis import across_cases, compare, consistency

OPS = {"baseline": "[]", "add": '[{"add": "E03"}]'}


def frame(rows):
    df = pd.DataFrame(rows, columns=["case_id", "model_key", "condition_id", "run_index", "probability", "verdict"])
    df["ref"] = "baseline"
    df["factor"] = df.condition_id
    df["level"] = df.condition_id
    df["exploratory"] = 0
    df["ops_json"] = df.condition_id.map(OPS)
    df["status"] = "done"
    df["parse_status"] = "valid"
    df["key_evidence_json"] = json.dumps([{"label": "Ex.P-1", "item_id": "E03", "bearing": "supports_conviction"}])
    return df


def test_hand_computed():
    rows = [("c1", "m", "baseline", r, p, v) for r, (p, v) in enumerate([(40, "acquit"), (50, "acquit"), (60, "convict")])]
    rows += [("c1", "m", "add", r, p, v) for r, (p, v) in enumerate([(60, "acquit"), (70, "convict"), (80, "convict")])]
    cmp = compare(frame(rows)).set_index("condition_id")
    assert cmp.loc["add", "delta_pp"] == 20.0          # 70 - 50
    assert cmp.loc["add", "mean_p"] == 70.0
    assert abs(cmp.loc["add", "flip_rate"] - 2 / 3) < 1e-9  # baseline modal verdict = acquit
    assert abs(cmp.loc["baseline", "conviction_rate"] - 1 / 3) < 1e-9
    assert cmp.loc["add", "cited_rate"] == 1.0
    assert cmp.loc["add", "ci_lo"] <= 20 <= cmp.loc["add", "ci_hi"]


def test_recovers_synthetic_effect():
    rng = np.random.default_rng(7)
    rows = []
    for k in range(12):                       # 12 cases with case-level heterogeneity
        base = rng.normal(45, 8)
        effect = rng.normal(10, 2)            # true mean shift +10 pp
        for r in range(20):
            for cond, mu in (("baseline", base), ("add", base + effect)):
                p = float(np.clip(rng.normal(mu, 8), 0, 100))
                rows.append((f"c{k}", "m", cond, r, p, "convict" if p > 60 else "acquit"))
    cmp = compare(frame(rows))
    ac = across_cases(cmp).set_index("condition_id").loc["add"]
    assert ac.k_cases == 12
    assert abs(ac.mean_delta_pp - 10) < 2.5
    assert ac.ci_lo < 10 < ac.ci_hi
    assert ac.p_signflip < 0.01
    cons = consistency(frame(rows)).iloc[0]
    assert 6 < cons.mean_sd_p < 10            # run-level SD was 8


def test_baseline_only_experiment():
    rows = [("c1", "m", "baseline", r, 50.0, "acquit") for r in range(3)]
    ac = across_cases(compare(frame(rows)))
    assert ac.empty and "k_cases" in ac.columns


def test_null_effect_not_significant():
    rng = np.random.default_rng(3)
    rows = [(f"c{k}", "m", cond, r, float(rng.normal(50, 8)), "acquit")
            for k in range(10) for r in range(10) for cond in ("baseline", "add")]
    ac = across_cases(compare(frame(rows))).set_index("condition_id").loc["add"]
    assert ac.ci_lo < 0 < ac.ci_hi
