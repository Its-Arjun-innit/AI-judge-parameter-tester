"""Versioned data schemas (v1): cases, experiments, and the model response."""
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

Category = Literal["eyewitness", "electronic", "physical", "forensic", "expert",
                   "confession", "documentary", "circumstantial", "character", "procedural"]


class Strict(BaseModel):
    model_config = ConfigDict(extra="forbid")


class Attributes(Strict):
    reliability: Literal["high", "low"] = "high"
    independence: str = Field("independent", pattern=r"^(independent|derived:\w+)$")
    specificity: Literal["specific", "vague"] = "specific"
    source_credibility: Literal["neutral_official", "interested", "related", "hostile"] = "neutral_official"
    admissibility: Literal["admitted", "contested_admitted", "inadmissible_heard"] = "admitted"
    potential_bias: Literal["none", "enmity", "motive_to_implicate"] = "none"


class Relations(Strict):
    corroborates: list[str] = []
    contradicts: list[str] = []


class Variant(Strict):
    """An alternative text for an item. Must declare exactly one manipulation:
    one attribute change (content) or one presentation label (wording)."""
    text: str
    changes: dict[str, str] = {}
    presentation: str | None = None


class EvidenceItem(Strict):
    id: str
    side: Literal["prosecution", "defence"]
    kind: Literal["witness", "document", "object"]  # -> PW/DW, Ex.P/Ex.D, MO labels
    category: Category
    target_fact: str | None = None
    direction: Literal["inculpatory", "exculpatory", "neutral"]
    attributes: Attributes = Attributes()
    relations: Relations = Relations()
    text: str
    variants: dict[str, Variant] = {}

    @model_validator(mode="after")
    def _variants_single_change(self):
        for name, v in self.variants.items():
            if len(v.changes) + (v.presentation is not None) != 1:
                raise ValueError(f"{self.id}.variants.{name}: declare exactly one change or one presentation")
            Attributes(**{**self.attributes.model_dump(), **v.changes})  # levels must be valid
        return self


class Fact(Strict):
    id: str
    text: str


class Law(Strict):
    offence: str
    provisions_text: str
    standard_of_proof: str
    principles: list[str] = []


class Case(Strict):
    schema_version: Literal[1]
    case_id: str
    category: str
    law: Law
    facts: list[Fact]
    disputed_facts: list[Fact] = []
    accused_statement: str
    evidence: list[EvidenceItem]
    baseline: list[str]
    researcher_notes: dict = {}  # never rendered to the model

    @model_validator(mode="after")
    def _refs(self):
        ids = [e.id for e in self.evidence]
        if len(ids) != len(set(ids)):
            raise ValueError("duplicate evidence ids")
        fact_ids = {f.id for f in self.facts + self.disputed_facts}
        for i in self.baseline:
            if i not in ids:
                raise ValueError(f"baseline references unknown item {i}")
        for e in self.evidence:
            if e.target_fact and e.target_fact not in fact_ids:
                raise ValueError(f"{e.id}.target_fact {e.target_fact} not a fact id")
            for r in e.relations.corroborates + e.relations.contradicts:
                if r not in ids:
                    raise ValueError(f"{e.id}.relations references unknown item {r}")
            dep = e.attributes.independence
            if dep.startswith("derived:") and dep[8:] not in ids:
                raise ValueError(f"{e.id}.independence references unknown item {dep[8:]}")
        return self

    def item(self, item_id: str) -> EvidenceItem:
        return next(e for e in self.evidence if e.id == item_id)


class Op(Strict):
    """One edit relative to the condition's `ref`. Exactly one field is set."""
    add: str | None = None
    remove: str | None = None
    variant: dict[str, str] | None = None       # {item: variant_name}
    replace: dict[str, str] | None = None       # {old_item: new_item} (category swap)
    position: dict[str, Literal["first", "last"]] | None = None

    @model_validator(mode="after")
    def _one(self):
        if sum(v is not None for v in self.model_dump().values()) != 1:
            raise ValueError("each op sets exactly one of add/remove/variant/replace/position")
        return self


class Condition(Strict):
    id: str
    factor: str = "baseline"
    level: str = "baseline"
    ref: str = "baseline"
    ops: list[Op] = []
    exploratory: bool = False


class ModelConfig(Strict):
    key: str
    provider: Literal["anthropic", "openai", "gemini", "fake"]
    model: str
    temperature: float | None = None  # None = provider default (not sent)
    effort: str | None = None         # anthropic output_config.effort / openai reasoning.effort / gemini thinking_level
    thinking: bool | None = None      # False = thinking off (anthropic) / budget 0 (gemini); None = provider default
    rpm: float | None = None          # pace requests to stay under per-minute limits
    max_tokens: int = 8000
    price_in: float = 0.0             # USD per 1M input tokens - you maintain these
    price_out: float = 0.0            # USD per 1M output tokens


class Budget(Strict):
    max_usd: float
    max_calls: int


class Experiment(Strict):
    experiment_id: str
    cases: list[str]
    prompt_version: str = "v1"
    models: list[ModelConfig]
    runs: int = Field(ge=1)
    seed: int
    order: Literal["randomize_within_side", "fixed"] = "randomize_within_side"
    budget: Budget
    conditions: list[Condition]

    @model_validator(mode="after")
    def _conds(self):
        ids = [c.id for c in self.conditions]
        if len(ids) != len(set(ids)) or "baseline" not in ids:
            raise ValueError("condition ids must be unique and include 'baseline'")
        for c in self.conditions:
            if c.id != "baseline" and c.ref not in ids:
                raise ValueError(f"condition {c.id}: unknown ref {c.ref}")
        if len({m.key for m in self.models}) != len(self.models):
            raise ValueError("model keys must be unique")
        return self


# ---- model response (what every provider must return) ----
# Field order is part of the protocol: reasoning first, decision last.

class KeyEvidence(Strict):
    label: str = Field(description="Evidence label as shown in the record, e.g. PW-2 or Ex.P-1")
    bearing: Literal["supports_conviction", "supports_acquittal"]


class Judgment(Strict):
    reasoning: str = Field(max_length=4000, description="Your reasoning, at most about 250 words")
    legal_standard_applied: str = Field(max_length=1000, description="The legal standard you applied, one or two sentences")
    key_evidence: list[KeyEvidence] = Field(max_length=5, description="Up to 5 items that most influenced the decision")
    missing_information: list[str] = Field(max_length=5, description="Up to 5 pieces of missing information or uncertainty")
    probability_committed: int = Field(ge=0, le=100, description="Integer 0-100: your probability that the accused committed the offence as charged")
    verdict: Literal["convict", "acquit"]
    confidence: int = Field(ge=1, le=7, description="Integer 1-7: confidence in your verdict (1 = not at all, 7 = completely)")


_UNSUPPORTED = {"title", "minimum", "maximum", "maxLength", "minLength", "maxItems", "minItems", "default"}


def response_schema() -> dict:
    """Judgment as a provider-portable JSON Schema: refs inlined, range keywords dropped
    (several providers reject them; ranges are in descriptions and enforced by parse())."""
    s = Judgment.model_json_schema()
    defs = s.pop("$defs", {})

    def clean(n, key=None):
        if isinstance(n, dict):
            if "$ref" in n:
                return clean(defs[n["$ref"].rsplit("/", 1)[-1]])
            # property names live under "properties"; never drop those
            return {k: clean(v, k) for k, v in n.items() if key == "properties" or k not in _UNSUPPORTED}
        if isinstance(n, list):
            return [clean(x) for x in n]
        return n
    return clean(s)
