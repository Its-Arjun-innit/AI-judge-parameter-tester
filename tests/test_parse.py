import json

from judgelab.adapters import Raw
from judgelab.parse import parse
from judgelab.schema import response_schema

GOOD = {"reasoning": "r", "legal_standard_applied": "BRD", "missing_information": [],
        "key_evidence": [{"label": "PW-1", "bearing": "supports_conviction"},
                         {"label": "PW-9", "bearing": "supports_acquittal"}],
        "probability_committed": 62, "verdict": "acquit", "confidence": 4}
LABELS = {"PW-1": "E01"}


def raw(text, stop="end"):
    return Raw(text, "m", 1, 1, stop, None, 0)


def test_valid_maps_labels():
    out = parse(raw(json.dumps(GOOD)), LABELS)
    assert out["parse_status"] == "valid" and out["probability"] == 62
    assert [k["item_id"] for k in json.loads(out["key_evidence_json"])] == ["E01", None]


def test_failure_modes():
    assert parse(raw("not json"), LABELS)["parse_status"] == "invalid_json"
    assert parse(raw('{"reasoning": "cut', "max_tokens"), LABELS)["parse_status"] == "truncated"
    assert parse(raw(json.dumps(GOOD | {"probability_committed": 150})), LABELS)["parse_status"] == "schema_violation"
    assert parse(raw(json.dumps(GOOD | {"verdict": "guilty"})), LABELS)["parse_status"] == "schema_violation"
    assert parse(raw(json.dumps({k: v for k, v in GOOD.items() if k != "confidence"})), LABELS)["parse_status"] == "schema_violation"
    assert parse(raw(json.dumps(GOOD | {"extra": 1})), LABELS)["parse_status"] == "schema_violation"
    assert parse(raw("", "refusal"), LABELS)["parse_status"] == "refusal"
    assert parse(raw(None), LABELS)["parse_status"] == "invalid_json"


def test_provider_schema_is_portable():
    s = response_schema()
    text = json.dumps(s)
    assert "$ref" not in text and "maximum" not in text and "maxLength" not in text
    assert list(s["properties"])[-3:] == ["probability_committed", "verdict", "confidence"]  # decision last
    assert set(s["required"]) == set(s["properties"]) and s["additionalProperties"] is False
