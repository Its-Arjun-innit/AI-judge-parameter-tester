"""CLI. `--dry-run` (before the subcommand) uses data/dryrun.sqlite and a fake model,
so rehearsals can never mix with real data."""
import argparse
import json
import os
import sys
from pathlib import Path

import yaml

from . import analysis, runner
from .adapters import ProviderError
from .cases import ROOT, load_case
from .conditions import expand
from .db import connect
from .schema import Experiment


def load_env(path=ROOT / ".env"):
    """KEY=VALUE lines from the gitignored .env; real environment variables win."""
    if path.exists():
        for line in path.read_text(encoding="utf-8").splitlines():
            key, sep, value = line.partition("=")
            if sep and not key.strip().startswith("#"):
                os.environ.setdefault(key.strip(), value.strip())


def main(argv=None):
    load_env()
    ap = argparse.ArgumentParser(prog="python -m judgelab")
    ap.add_argument("--dry-run", action="store_true", help="fake model + data/dryrun.sqlite")
    ap.add_argument("--db", help="database path (default data/lab.sqlite)")
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("validate", help="validate case and experiment YAML files")
    s.add_argument("paths", nargs="+")
    s = sub.add_parser("plan", help="write pending trials for an experiment spec; print counts + estimate")
    s.add_argument("spec")
    s = sub.add_parser("run", help="execute pending trials")
    s.add_argument("experiment_id")
    s.add_argument("--condition")
    s.add_argument("--max-usd", type=float)
    s.add_argument("--concurrency", type=int, help="parallel analyses (default: 1 for local models, else 4)")
    s = sub.add_parser("export", help="write observations (+ all attempts) to results/<id>/")
    s.add_argument("experiment_id")
    s.add_argument("--format", choices=["csv", "json"], default="csv")
    s = sub.add_parser("analyze", help="print and save condition comparisons and consistency")
    s.add_argument("experiment_id")
    a = ap.parse_args(argv)

    if a.cmd == "validate":
        return validate(a.paths)
    con = connect(a.db or ROOT / "data" / ("dryrun.sqlite" if a.dry_run else "lab.sqlite"))
    try:
        return command(a, con)
    except (ValueError, RuntimeError, ProviderError) as e:  # plain message, no traceback
        print(f"Error: {e}", flush=True)
        return 1


def command(a, con):
    if a.cmd == "plan":
        print(json.dumps(runner.plan(con, a.spec), indent=2))
    elif a.cmd == "run":
        print(json.dumps(runner.run(con, a.experiment_id, a.condition, a.max_usd, a.dry_run, a.concurrency),
                         indent=2))
    elif a.cmd == "export":
        out = ROOT / "results" / a.experiment_id
        out.mkdir(parents=True, exist_ok=True)
        obs = analysis.observations(con, a.experiment_id)
        att = analysis.pd.read_sql_query(
            "SELECT a.* FROM attempts a JOIN trials t USING (trial_id) WHERE t.experiment_id=?",
            con, params=(a.experiment_id,))
        if a.format == "csv":
            obs.to_csv(out / "observations.csv", index=False)
            att.to_csv(out / "attempts.csv", index=False)
        else:
            spec = con.execute("SELECT spec_yaml FROM experiments WHERE experiment_id=?",
                               (a.experiment_id,)).fetchone()[0]
            (out / "export.json").write_text(json.dumps(
                {"experiment_spec": yaml.safe_load(spec), "observations": obs.to_dict("records"),
                 "attempts": att.to_dict("records")}, indent=1, default=str), encoding="utf-8")
        print(f"{len(obs)} observations, {len(att)} attempts -> {out}")
    elif a.cmd == "analyze":
        obs = analysis.observations(con, a.experiment_id)
        out = ROOT / "results" / a.experiment_id
        out.mkdir(parents=True, exist_ok=True)
        cmp = analysis.compare(obs)
        for name, frame in [("comparison", cmp), ("across_cases", analysis.across_cases(cmp)),
                            ("consistency", analysis.consistency(obs))]:
            frame.to_csv(out / f"{name}.csv", index=False)
            print(f"\n== {name} (observed response differences, not evidence weights) ==")
            print(frame.round(2).to_string(index=False))


def validate(paths) -> int:
    ok = True
    for p in paths:
        files = sorted(Path(p).glob("*.yaml")) if Path(p).is_dir() else [Path(p)]
        for f in files:
            try:
                data = yaml.safe_load(f.read_text(encoding="utf-8"))
                if "experiment_id" in data:
                    exp = Experiment.model_validate(data)
                    expand(exp, [load_case(ROOT / "cases" / f"{c}.yaml") for c in exp.cases])
                else:
                    load_case(f)
                print(f"OK    {f}")
            except Exception as e:
                ok = False
                print(f"FAIL  {f}\n      {e}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
