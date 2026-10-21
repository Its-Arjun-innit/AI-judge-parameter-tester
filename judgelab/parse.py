"""Response Parser. Never raises: every outcome becomes a parse_status."""
import json

from pydantic import ValidationError

from .adapters import Raw
from .schema import Judgment


def parse(raw: Raw, label_map: dict) -> dict:
    """Returns attempt columns: parse_status, the judgment fields, and parse_error."""
    out = {"parse_status": "valid", "parse_error": None}
    if raw.stop_reason == "refusal":
        return out | {"parse_status": "refusal"}
    try:
        data = json.loads(raw.text)
    except (json.JSONDecodeError, TypeError) as e:
        status = "truncated" if raw.stop_reason == "max_tokens" else "invalid_json"
        return out | {"parse_status": status, "parse_error": str(e)}
    try:
        j = Judgment.model_validate(data)
    except ValidationError as e:
        return out | {"parse_status": "schema_violation", "parse_error": str(e)}
    cited = [{"label": k.label, "item_id": label_map.get(k.label.strip()), "bearing": k.bearing}
             for k in j.key_evidence]
    return out | {"probability": j.probability_committed, "verdict": j.verdict, "confidence": j.confidence,
                  "key_evidence_json": json.dumps(cited), "legal_standard": j.legal_standard_applied,
                  "reasoning": j.reasoning, "missing_info_json": json.dumps(j.missing_information)}
