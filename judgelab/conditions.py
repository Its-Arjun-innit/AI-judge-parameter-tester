"""Experiment Builder: turn condition ops into evidence orders, and check that each
condition differs from its reference in exactly one evidence item."""
import difflib
import hashlib
import random
from dataclasses import dataclass, field

from .cases import Prompt, render
from .schema import Case, Condition, Experiment


@dataclass
class State:
    items: list = field(default_factory=list)  # (item_id, variant) in canonical order
    pins: dict = field(default_factory=dict)   # item_id -> "first" | "last"


class ConditionError(ValueError):
    pass


def apply(case: Case, conds: dict[str, Condition], cid: str) -> State:
    """Canonical state of condition `cid`: its ref's state with its ops applied."""
    if cid == "baseline":
        return State([(i, "base") for i in case.baseline])
    c = conds[cid]
    if c.ref == cid:
        raise ConditionError(f"{cid}: ref cannot be itself")
    base = apply(case, conds, c.ref)
    st = State(list(base.items), dict(base.pins))
    ids = lambda: [i for i, _ in st.items]
    known = {e.id for e in case.evidence}

    def need(x, present=True):
        if x not in known:
            raise ConditionError(f"{case.case_id}/{cid}: unknown item {x}")
        if (x in ids()) != present:
            raise ConditionError(f"{case.case_id}/{cid}: item {x} {'not ' if present else 'already '}present")

    for op in c.ops:
        if op.add:
            need(op.add, present=False)
            st.items.append((op.add, "base"))
        elif op.remove:
            need(op.remove)
            st.items = [p for p in st.items if p[0] != op.remove]
            st.pins.pop(op.remove, None)
        elif op.variant:
            for x, v in op.variant.items():
                need(x)
                if v != "base" and v not in case.item(x).variants:
                    raise ConditionError(f"{case.case_id}/{cid}: {x} has no variant {v}")
                st.items = [(i, v if i == x else w) for i, w in st.items]
        elif op.replace:
            for old, new in op.replace.items():
                need(old)
                need(new, present=False)
                st.items = [(new, "base") if i == old else (i, w) for i, w in st.items]
                if old in st.pins:
                    st.pins[new] = st.pins.pop(old)
        elif op.position:
            for x, where in op.position.items():
                need(x)
                pair = next(p for p in st.items if p[0] == x)
                st.items.remove(pair)
                st.items.insert(0, pair) if where == "first" else st.items.append(pair)
                st.pins[x] = where
    return st


def presentation(case: Case, st: State, randomize: bool, rng: random.Random) -> list:
    """Order for one trial: prosecution then defence; within a side, pinned items go
    first/last and the rest keep canonical order or are shuffled."""
    out = []
    for side in ("prosecution", "defence"):
        items = [p for p in st.items if case.item(p[0]).side == side]
        first = [p for p in items if st.pins.get(p[0]) == "first"]
        last = [p for p in items if st.pins.get(p[0]) == "last"]
        free = [p for p in items if p[0] not in st.pins]
        if randomize:
            rng.shuffle(free)
        out += first + free + last
    return out


def trial_rng(seed: int, case_id: str, cid: str, run: int) -> random.Random:
    h = hashlib.sha256(f"{seed}|{case_id}|{cid}|{run}".encode()).hexdigest()
    return random.Random(int(h[:16], 16))


def assert_single_change(ref: Prompt, cond: Prompt, where: str = "") -> None:
    """Non-evidence text must be identical; the evidence may differ by exactly one
    inserted, deleted or substituted block, or by one block moved."""
    if ref.frame != cond.frame or ref.system != cond.system:
        raise ConditionError(f"{where}: non-evidence prompt text differs from its reference")
    a, b = ref.blocks, cond.blocks
    if a == b:
        raise ConditionError(f"{where}: identical to its reference (no manipulation)")
    ops = [o for o in difflib.SequenceMatcher(a=a, b=b, autojunk=False).get_opcodes() if o[0] != "equal"]
    if len(ops) == 1:
        tag, i1, i2, j1, j2 = ops[0]
        if (tag, i2 - i1, j2 - j1) in {("insert", 0, 1), ("delete", 1, 0), ("replace", 1, 1)}:
            return
    if sorted(a) == sorted(b):  # pure move: removing one block from both leaves equal sequences
        for x in set(a):
            if [y for y in a if y != x] == [y for y in b if y != x]:
                return
    raise ConditionError(f"{where}: evidence differs from its reference in more than one item")


def expand(exp: Experiment, cases: list[Case]) -> dict:
    """Validate every condition on every case. Returns {(case_id, cid): State}."""
    conds = {c.id: c for c in exp.conditions}
    states = {}
    for case in cases:
        for c in exp.conditions:
            states[case.case_id, c.id] = apply(case, conds, c.id)
        for c in exp.conditions:
            if c.id == "baseline":
                continue
            canon = lambda cid: render(case, presentation(case, states[case.case_id, cid], False, None),
                                       exp.prompt_version)
            ref, cur = canon(c.ref), canon(c.id)
            if ref.blocks == cur.blocks and states[case.case_id, c.ref].pins != states[case.case_id, c.id].pins:
                continue  # pin-only change (e.g. "first" vs randomised): visible only per trial
            assert_single_change(ref, cur, f"{case.case_id}/{c.id} vs {c.ref}")
    return states
