"""Cases: create and edit hypothetical cases. Forms write the same YAML files, through the same validation."""
import re

import pandas as pd
import streamlit as st
import yaml
from pydantic import ValidationError

from judgelab.cases import case_hash
from judgelab.schema import Attributes, Case
from ui import common

st.title("Cases")
st.caption("Fictional cases only. Everything here except the researcher notes is shown to the AI.")

BLANK = {"schema_version": 1, "case_id": "", "category": "",
         "law": {"offence": "", "provisions_text": "",
                 "standard_of_proof": "The prosecution must prove every element of the offence beyond reasonable "
                                      "doubt. If a reasonable doubt remains, the accused must be acquitted.",
                 "principles": []},
         "facts": [], "disputed_facts": [], "accused_statement": "The accused denied the allegations.",
         "evidence": [], "baseline": [], "researcher_notes": {}}
LEVELS = {name: list(f.annotation.__args__) for name, f in Attributes.model_fields.items() if name != "independence"}

files = common.case_files()
if "case_pick_next" in st.session_state:  # after saving a new case, select it (must happen before the widget)
    st.session_state["case_pick"] = st.session_state.pop("case_pick_next")
if msg := st.session_state.pop("case_saved_msg", None):
    st.success(msg)
if st.session_state.get("case_pick") not in ["(new case)"] + list(files):  # first visit, or file removed
    st.session_state["case_pick"] = st.session_state.get("case_loaded") if st.session_state.get(
        "case_loaded") in files else "(new case)"
left, right = st.columns([3, 1])
choice = left.selectbox("Case", ["(new case)"] + list(files), key="case_pick")
if choice != st.session_state.get("case_loaded", "(new case)") or "case_draft" not in st.session_state:
    st.session_state["case_draft"] = (yaml.safe_load(files[choice].read_text(encoding="utf-8"))
                                      if choice in files else yaml.safe_load(yaml.safe_dump(BLANK)))
    st.session_state["case_loaded"] = choice
    st.session_state["case_rev"] = st.session_state.get("case_rev", 0) + 1
d = st.session_state["case_draft"]
rev = st.session_state["case_rev"]
is_new = choice not in files
k = lambda *parts: "-".join(map(str, (rev,) + parts))  # noqa: E731  widget keys tied to this draft


def next_id(prefix: str, existing: list[str]) -> str:
    nums = [int(m.group(1)) for x in existing if (m := re.fullmatch(prefix + r"(\d+)", x))]
    return f"{prefix}{max(nums, default=0) + 1:02d}"


# ---- identity ----
if is_new:
    d["case_id"] = st.text_input("Case reference", d["case_id"], key=k("id"), placeholder="e.g. IN-ROBBERY-001",
                                 help="A short unique name: letters, numbers and dashes. It cannot be changed later.")
else:
    st.markdown(f"**Case reference:** {d['case_id']}")
d["category"] = st.text_input("Type of case", d.get("category", ""), key=k("cat"), placeholder="e.g. property/theft")

# ---- charge and law ----
st.subheader("Charge and law")
law = d["law"]
law["offence"] = st.text_area("Alleged offence (the charge)", law["offence"], key=k("off"), height=80)
law["provisions_text"] = st.text_area("Applicable law, as given to the AI", law["provisions_text"], key=k("prov"),
                                      height=160, help="The AI decides only on the law stated here.")
law["standard_of_proof"] = st.text_area("Standard of proof", law["standard_of_proof"], key=k("std"), height=80)
law["principles"] = [p.strip() for p in st.text_area(
    "Guiding principles (one per line)", "\n".join(law.get("principles", [])), key=k("pr"), height=100).splitlines()
    if p.strip()]

# ---- facts ----
st.subheader("Facts")
st.caption("Agreed facts are stipulated and shown as undisputed. Facts in dispute are what the evidence must resolve; "
           "evidence items say which disputed fact they bear on.")
for field, title, prefix in (("facts", "Agreed facts (stipulated)", "F"), ("disputed_facts", "Facts in dispute", "D")):
    st.markdown(f"**{title}**")
    edited = st.data_editor(pd.DataFrame(d.get(field) or [], columns=["id", "text"]), num_rows="dynamic",
                            key=k(field), width="stretch", hide_index=True,
                            column_config={"id": st.column_config.TextColumn("Ref", disabled=True, width="small"),
                                           "text": st.column_config.TextColumn("Text", width="large")})
    rows, ids = [], [r for r in edited["id"].dropna().tolist() if r]
    for r in edited.itertuples():
        if isinstance(r.text, str) and r.text.strip():
            rid = r.id if isinstance(r.id, str) and r.id else next_id(prefix, ids)
            ids.append(rid)
            rows.append({"id": rid, "text": r.text.strip()})
    d[field] = rows
d["accused_statement"] = st.text_area("Statement of the accused (BNSS s.351)", d["accused_statement"],
                                      key=k("acc"), height=70)

# ---- evidence ----
fact_ids = [f["id"] for f in d["facts"] + d["disputed_facts"]]
fact_text = {f["id"]: f["text"] for f in d["facts"] + d["disputed_facts"]}
all_ids = [e["id"] for e in d["evidence"]]
for side in ("prosecution", "defence"):
    st.subheader(f"{side.capitalize()} evidence")
    for e in [e for e in d["evidence"] if e["side"] == side]:
        eid = e["id"]
        title = f"{eid}: {common.label(common.CATEGORY, e.get('category'))} - {(e.get('text') or 'new item')[:70]}"
        with st.expander(title, expanded=not e.get("text")):
            e["text"] = st.text_area("Description, as shown to the AI", e.get("text", ""), key=k(eid, "t"), height=100,
                                     help="Refer to people by role only (e.g. 'a tea-stall owner'); "
                                          "never refer to other items' labels.")
            c1, c2, c3 = st.columns(3)
            e["kind"] = c1.selectbox("Form", list(common.KIND), list(common.KIND).index(e.get("kind", "witness")),
                                     format_func=common.KIND.get, key=k(eid, "kind"))
            e["category"] = c2.selectbox("Type of evidence", list(common.CATEGORY),
                                         list(common.CATEGORY).index(e.get("category", "eyewitness")),
                                         format_func=common.CATEGORY.get, key=k(eid, "cat"))
            e["direction"] = c3.selectbox("Points", list(common.DIRECTION),
                                          list(common.DIRECTION).index(e.get("direction", "inculpatory")),
                                          format_func=common.DIRECTION.get, key=k(eid, "dir"))
            opts = [None] + fact_ids
            tf = st.selectbox("Bears on", opts, opts.index(e.get("target_fact")) if e.get("target_fact") in opts else 0,
                              key=k(eid, "tf"), format_func=lambda f: "(no particular fact)" if f is None
                              else f"{f}: {fact_text[f][:90]}")
            e["target_fact"] = tf
            in_base = st.checkbox("Include in the original evidence set", eid in d["baseline"], key=k(eid, "base"),
                                  help=common.HELP["baseline"])
            d["baseline"] = [x for x in d["baseline"] if x != eid] + ([eid] if in_base else [])
            st.markdown("**Attributes**")
            a = e.setdefault("attributes", {})
            cols = st.columns(3)
            for j, (attr, levels) in enumerate(LEVELS.items()):
                cur = a.get(attr, Attributes().model_dump()[attr])
                a[attr] = cols[j % 3].selectbox(common.ATTRIBUTE[attr], levels, levels.index(cur), key=k(eid, attr),
                                                format_func=lambda v: common.label(common.LEVEL, v),
                                                help=common.HELP[attr])
            dep_opts = ["independent"] + [f"derived:{x}" for x in all_ids if x != eid]
            cur = a.get("independence", "independent")
            a["independence"] = cols[2].selectbox(
                "Independence", dep_opts, dep_opts.index(cur) if cur in dep_opts else 0, key=k(eid, "ind"),
                help=common.HELP["independence"],
                format_func=lambda v: "independent source" if v == "independent" else f"repeats {v[8:]}")
            rel = e.setdefault("relations", {})
            others = [x for x in all_ids if x != eid]
            r1, r2 = st.columns(2)
            rel["corroborates"] = r1.multiselect("Supports (corroborates)", others,
                                                 [x for x in rel.get("corroborates", []) if x in others],
                                                 key=k(eid, "cor"))
            rel["contradicts"] = r2.multiselect("Contradicts", others,
                                                [x for x in rel.get("contradicts", []) if x in others],
                                                key=k(eid, "con"))
            st.markdown("**Alternative versions**")
            st.caption(common.HELP["variant"])
            variants = e.setdefault("variants", {})
            for vname in list(variants):
                v = variants[vname]
                with st.container(border=True):
                    st.markdown(f"Version **{vname}**")
                    v["text"] = st.text_area("Text of this version", v.get("text", ""), key=k(eid, vname, "t"),
                                             height=90)
                    what_opts = ["wording"] + list(LEVELS) + ["independence"]
                    what_cur = next(iter(v.get("changes") or {}), "wording")
                    what = st.selectbox("This version changes", what_opts, what_opts.index(what_cur),
                                        key=k(eid, vname, "w"),
                                        format_func=lambda w: "only the wording (same content)" if w == "wording"
                                        else f"the {common.ATTRIBUTE[w].lower()}")
                    if what == "wording":
                        v["presentation"] = st.text_input("Describe the wording change", v.get("presentation") or "",
                                                          key=k(eid, vname, "p"), placeholder="e.g. formal_register")
                        v.pop("changes", None)
                    else:
                        lv = dep_opts if what == "independence" else LEVELS[what]
                        cur = (v.get("changes") or {}).get(what, lv[0])
                        v["changes"] = {what: st.selectbox("Changed to", lv, lv.index(cur) if cur in lv else 0,
                                                           key=k(eid, vname, "l"),
                                                           format_func=lambda x: common.label(common.LEVEL, x))}
                        v.pop("presentation", None)
                    if st.button("Delete this version", key=k(eid, vname, "del")):
                        variants.pop(vname)
                        st.rerun()
            nv = st.text_input("New version name", key=k(eid, "nvn"), placeholder="e.g. low_reliability")
            if st.button("Add an alternative version", key=k(eid, "nv")) and nv.strip():
                variants[re.sub(r"\W+", "_", nv.strip())] = {"text": e["text"], "presentation": "reworded"}
                st.rerun()
            if st.button("Delete this item", key=k(eid, "del")):
                d["evidence"] = [x for x in d["evidence"] if x["id"] != eid]
                d["baseline"] = [x for x in d["baseline"] if x != eid]
                st.rerun()
    if st.button(f"Add {side} evidence", key=k("add", side)):
        nid = next_id("E", all_ids + st.session_state.setdefault("used_ids", []))
        st.session_state["used_ids"].append(nid)
        d["evidence"].append({"id": nid, "side": side, "kind": "witness", "category": "eyewitness",
                              "direction": "inculpatory" if side == "prosecution" else "exculpatory", "text": ""})
        d["baseline"].append(nid)  # new items start in the original set; untick to keep one for comparisons
        st.rerun()

# ---- notes and save ----
st.subheader("Notes not shown to the AI")
notes = st.text_area("Researcher notes (e.g. intended scenario, legal review status)",
                     yaml.safe_dump(d.get("researcher_notes") or {}, sort_keys=False, allow_unicode=True)
                     if d.get("researcher_notes") else "", key=k("notes"), height=120)
try:
    parsed = yaml.safe_load(notes) if notes.strip() else {}
    d["researcher_notes"] = parsed if isinstance(parsed, dict) else {"notes": notes}
except yaml.YAMLError:
    d["researcher_notes"] = {"notes": notes}


def plain_errors(err: ValidationError) -> list[str]:
    out = []
    for e in err.errors():
        loc = list(e["loc"])
        if len(loc) > 1 and loc[0] == "evidence" and isinstance(loc[1], int) and loc[1] < len(d["evidence"]):
            loc[1] = d["evidence"][loc[1]]["id"]
        out.append(f"{' > '.join(map(str, loc)) or 'case'}: {e['msg'].replace('Value error, ', '')}")
    return out


if st.button("Save case", type="primary", key=k("save")):
    d["evidence"] = [{key: val for key, val in e.items() if val not in ({}, [], None, "")} | {"text": e.get("text", "")}
                     for e in d["evidence"]]
    try:
        case = Case.model_validate(d)
        if is_new and (not re.fullmatch(r"[A-Za-z0-9_-]+", case.case_id) or case.case_id in files):
            raise ValueError("Choose a new, unique case reference using letters, numbers and dashes.")
        path = common.CASES_DIR / f"{case.case_id}.yaml"
        used = common.con().execute("SELECT COUNT(*) FROM cases WHERE case_id=? AND case_hash != ?",
                                    (case.case_id, case_hash(case))).fetchone()[0]
        common.write_yaml(path, case.model_dump(mode="json", exclude_defaults=False))
        st.session_state["case_loaded"] = st.session_state["case_pick_next"] = case.case_id
        st.session_state["case_saved_msg"] = (f"Saved {case.case_id}." + (
            " Experiments already run keep the version of the case they used." if used else ""))
        st.rerun()
    except ValidationError as e:
        st.error("The case could not be saved:\n\n" + "\n".join(f"- {m}" for m in plain_errors(e)))
    except ValueError as e:
        st.error(str(e))
