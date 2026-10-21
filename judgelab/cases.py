"""Case Manager: load, validate, hash and render cases."""
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from string import Template

import yaml

from .schema import Case

ROOT = Path(__file__).resolve().parent.parent
_PREFIX = {("prosecution", "witness"): "PW", ("defence", "witness"): "DW",
           ("prosecution", "document"): "Ex.P", ("defence", "document"): "Ex.D",
           ("prosecution", "object"): "MO", ("defence", "object"): "MO"}


@dataclass(frozen=True)
class Prompt:
    system: str
    user: str
    label_map: dict   # label -> item id
    order: list       # item ids in presentation order
    blocks: list      # (side, unlabelled text) in presentation order - for the diff check
    frame: str        # user text with evidence sections blanked - for the diff check


def load_case(path) -> Case:
    return Case.model_validate(yaml.safe_load(Path(path).read_text(encoding="utf-8")))


def sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def case_hash(case: Case) -> str:
    return sha256(json.dumps(case.model_dump(mode="json"), sort_keys=True))


def template(name: str, version: str) -> str:
    return (ROOT / "prompts" / f"{name}_{version}.md").read_text(encoding="utf-8")


def render(case: Case, order: list[tuple[str, str]], version: str) -> Prompt:
    """order: (item_id, variant) pairs in presentation order; variant 'base' = item.text."""
    counters, label_map, sections, blocks = {}, {}, {"prosecution": [], "defence": []}, []
    for item_id, variant in order:
        it = case.item(item_id)
        text = it.text if variant == "base" else it.variants[variant].text
        prefix = _PREFIX[(it.side, it.kind)]
        counters[prefix] = counters.get(prefix, 0) + 1
        label = f"{prefix}-{counters[prefix]}"
        label_map[label] = item_id
        sections[it.side].append(f"{label}: {text.strip()}")
        blocks.append((it.side, text.strip()))

    def bullets(xs):
        return "\n".join(f"- {x}" for x in xs) or "- None."

    fields = dict(
        offence=case.law.offence, provisions=case.law.provisions_text.strip(),
        standard=case.law.standard_of_proof, principles=bullets(case.law.principles),
        facts=bullets(f.text for f in case.facts), disputed=bullets(f.text for f in case.disputed_facts),
        accused_statement=case.accused_statement.strip())
    t = Template(template("user", version))
    user = t.substitute(fields,
                        prosecution="\n\n".join(sections["prosecution"]) or "No prosecution evidence was led.",
                        defence="\n\n".join(sections["defence"]) or "No defence evidence was led.")
    frame = t.substitute(fields, prosecution="<EVIDENCE>", defence="<EVIDENCE>")
    return Prompt(template("system", version), user, label_map, [i for i, _ in order], blocks, frame)
