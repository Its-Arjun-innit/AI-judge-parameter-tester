"""Analysis Engine (MVP): observed response differences between paired conditions.

Cases are the unit of generalisation; runs are repeated measurements within a cell
(case x condition x model). Mixed models / GEE come after the pilot (plan M7).
"""
import itertools
import json

import numpy as np
import pandas as pd
from statsmodels.stats.inter_rater import fleiss_kappa

OBS_SQL = """
SELECT t.trial_id, t.experiment_id, t.case_id, t.case_hash, t.condition_id, c.ref, c.factor, c.level,
       c.exploratory, c.ops_json, t.model_key, t.run_index, t.schedule_pos, t.evidence_order_json,
       t.label_map_json, t.prompt_version, t.system_hash, t.prompt_hash, t.params_json, t.status,
       t.exclusion_reason, a.attempt_no AS attempts, a.started_at, a.finished_at, a.model_returned,
       a.request_id, a.http_status, a.error, a.input_tokens, a.output_tokens, a.stop_reason, a.cost_usd,
       a.parse_status, a.probability, a.verdict, a.confidence, a.key_evidence_json, a.legal_standard,
       a.reasoning, a.missing_info_json
FROM trials t
JOIN conditions c ON c.experiment_id = t.experiment_id AND c.condition_id = t.condition_id
LEFT JOIN attempts a ON a.attempt_id =
  (SELECT attempt_id FROM attempts WHERE trial_id = t.trial_id ORDER BY attempt_no DESC LIMIT 1)
WHERE t.experiment_id = ?
ORDER BY t.case_id, t.condition_id, t.model_key, t.run_index
"""


def observations(con, experiment_id) -> pd.DataFrame:
    """One row per trial (final attempt), long format."""
    return pd.read_sql_query(OBS_SQL, con, params=(experiment_id,))


def _focal(ops_json: str) -> str | None:
    """The item a condition manipulates (for the manipulation check); None for removals."""
    for op in json.loads(ops_json):
        for k in ("add", "variant", "replace", "position"):
            if k in op:
                v = op[k]
                return v if isinstance(v, str) else list(v.values())[0] if k == "replace" else list(v)[0]
    return None


def _boot_diff(x, y, rng, b=2000):
    """Percentile CI for mean(x) - mean(y), resampling runs within each cell."""
    if len(x) < 2 or len(y) < 2:
        return np.nan, np.nan
    d = rng.choice(x, (b, len(x))).mean(1) - rng.choice(y, (b, len(y))).mean(1)
    return tuple(np.percentile(d, [2.5, 97.5]))


def compare(df: pd.DataFrame, seed: int = 0) -> pd.DataFrame:
    """Per case x model x condition: mean P, conviction rate, delta vs the condition's ref
    (percentage points, run-bootstrap 95% CI), verdict flip rate, invalid/refusal rates,
    and the share of valid runs citing the manipulated item."""
    rng = np.random.default_rng(seed)
    done = df[df.status == "done"]
    valid = done[done.parse_status == "valid"]
    rows = []
    for (case, model, cond), g in done.groupby(["case_id", "model_key", "condition_id"]):
        v = valid[(valid.case_id == case) & (valid.model_key == model) & (valid.condition_id == cond)]
        first = g.iloc[0]
        ref = valid[(valid.case_id == case) & (valid.model_key == model) & (valid.condition_id == first.ref)]
        p, pr = v.probability.to_numpy(float), ref.probability.to_numpy(float)
        is_ref = cond == "baseline"
        lo, hi = (np.nan, np.nan) if is_ref else _boot_diff(p, pr, rng)
        modal = ref.verdict.mode().iloc[0] if len(ref) else None
        focal = _focal(first.ops_json)
        cited = v.key_evidence_json.map(lambda s: any(k["item_id"] == focal for k in json.loads(s)))
        rows.append({
            "case_id": case, "model_key": model, "condition_id": cond, "ref": None if is_ref else first.ref,
            "factor": first.factor, "level": first.level, "exploratory": bool(first.exploratory),
            "n_done": len(g), "n_valid": len(v),
            "invalid_rate": 1 - len(v) / len(g), "refusal_rate": (g.parse_status == "refusal").mean(),
            "mean_p": p.mean() if len(p) else np.nan, "sd_p": p.std(ddof=1) if len(p) > 1 else np.nan,
            "conviction_rate": (v.verdict == "convict").mean() if len(v) else np.nan,
            "delta_pp": np.nan if is_ref or not len(p) or not len(pr) else p.mean() - pr.mean(),
            "ci_lo": lo, "ci_hi": hi,
            "flip_rate": np.nan if is_ref or modal is None or not len(v) else (v.verdict != modal).mean(),
            "focal_item": focal, "cited_rate": cited.mean() if focal and len(v) else np.nan,
        })
    return pd.DataFrame(rows)


def across_cases(cmp: pd.DataFrame, seed: int = 0, b: int = 5000) -> pd.DataFrame:
    """Two-stage inference: case-level deltas -> mean, case-cluster bootstrap CI and a
    sign-flip permutation p-value (exact up to 15 cases)."""
    rng = np.random.default_rng(seed)
    rows = []
    for (model, cond), g in cmp.dropna(subset=["delta_pp"]).groupby(["model_key", "condition_id"]):
        d = g.delta_pp.to_numpy(float)
        k = len(d)
        if k >= 2:
            lo, hi = np.percentile(rng.choice(d, (b, k)).mean(1), [2.5, 97.5])
            signs = (np.array(list(itertools.product([1, -1], repeat=k))) if k <= 15
                     else rng.choice([1, -1], (b, k)))
            p = (np.abs((signs * d).mean(1)) >= abs(d.mean()) - 1e-12).mean()
        else:
            lo = hi = p = np.nan
        rows.append({"model_key": model, "condition_id": cond, "factor": g.factor.iloc[0],
                     "level": g.level.iloc[0], "k_cases": k, "mean_delta_pp": d.mean(),
                     "ci_lo": lo, "ci_hi": hi, "p_signflip": p})
    return pd.DataFrame(rows)


def consistency(df: pd.DataFrame) -> pd.DataFrame:
    """Repeatability per model: mean within-cell SD of P, mean modal-verdict agreement,
    and Fleiss' kappa for verdicts (cells = subjects, runs = raters; each cell is
    truncated to the smallest run count so every subject has the same number of raters)."""
    v = df[(df.status == "done") & (df.parse_status == "valid")]
    rows = []
    for model, g in v.groupby("model_key"):
        cells = [c for _, c in g.groupby(["case_id", "condition_id"])]
        n = min(len(c) for c in cells)
        table = np.array([[(c.verdict.iloc[:n] == x).sum() for x in ("convict", "acquit")] for c in cells])
        kappa = fleiss_kappa(table) if n >= 2 and len(cells) >= 2 and table.sum(0).min() > 0 else np.nan
        rows.append({"model_key": model, "cells": len(cells), "runs_per_cell_used": n,
                     "mean_sd_p": np.nanmean([c.probability.std(ddof=1) for c in cells if len(c) > 1]),
                     "mean_modal_agreement": np.mean([c.verdict.value_counts(normalize=True).iloc[0] for c in cells]),
                     "fleiss_kappa_verdict": kappa})
    return pd.DataFrame(rows)
