"""Shape decisions: "two cells" or "one cell" for a split / merge configuration.

A split and a merge describe the same geometry (manipulation_votes): a large
footprint ("whole") covering two smaller ones ("pieces"). Where the sessions
are divided about it, the curator proposes both, again and again. A decision
settles it once for the two cells, in every session:

- "split" (two cells): a footprint covering both pieces is split with the
  blueprint pieces,
- "merge" (one cell): two footprints covering one piece each are merged with
  the blueprint whole.

The blueprint is three detected footprints, (session, footprint ID) each:
they keep their shape whatever happens to the assignments, so a decision
finds its place by geometry, not by neuron IDs. It is stored with the
assignments (``Assignments.shape_decisions``). The curator's votes follow it
for configurations whose pieces are the blueprint's (SessionVoter).
"""

from dataclasses import dataclass

import numpy as np

from catan.core.structures import NeuronComponent
from catan.gui.data.manipulation_votes import (
    COVER_FRACTION,
    INSIDE_FRACTION,
    SessionVoter,
)

from catan.tracking.structures.shape_decision import (  # noqa: F401
    KINDS,
    LABELS,
    ShapeDecision,
)

APPLY, DONE, CONFLICT = "apply", "done", "conflict"


@dataclass(frozen=True)
class DecisionStep:
    """What a decision means for one session."""

    session_id: int
    state: str  # APPLY, DONE or CONFLICT
    targets: tuple = ()  # split: (whole,), merge: the two pieces
    sources: tuple = ()  # split: the blueprint pieces, merge: (whole,)
    note: str = ""


def decisions_of(assignments):
    """{decision ID: ShapeDecision} of an assignment set."""
    return dict(getattr(assignments, "shape_decisions", None) or {})


def add_decision(assignments, decision):
    decisions = getattr(assignments, "shape_decisions", None)
    if decisions is None:
        decisions = assignments.shape_decisions = {}
    decision_id = max(decisions, default=-1) + 1
    decisions[decision_id] = decision
    return decision_id


def owner(ids, session_id, footprint_id):
    """Row holding footprint ``footprint_id`` of a session, or None."""
    if not 0 <= session_id < ids.shape[1]:
        return None
    rows = np.flatnonzero(ids[:, session_id] == footprint_id)
    return int(rows[0]) if len(rows) else None


def _dense(voter, ref):
    """0/1 support of a detected footprint (session, footprint ID)."""
    if not 0 <= ref[0] < len(voter.sessions):
        raise ValueError(f"Session {ref[0]} of the decision no longer exists.")
    supports, _, _ = voter._session_supports(ref[0])
    if not 0 <= ref[1] < supports.shape[1]:
        raise ValueError(
            f"Footprint {ref[1]} of session {ref[0]} (decision blueprint) no longer exists."
        )
    dense = np.zeros(supports.shape[0], dtype=np.float32)
    start, stop = supports.indptr[ref[1]], supports.indptr[ref[1] + 1]
    dense[supports.indices[start:stop]] = 1.0
    return dense


def plan(decision, sessions, ids, *, active=None):
    """DecisionSteps of all sessions where the configuration is found:
    APPLY (the manipulation to make), DONE (already as decided) or CONFLICT
    (found, but cannot be applied; ``note`` says why)."""
    ids = np.asarray(ids)
    refs = (decision.whole, *decision.pieces)
    if any(not 0 <= ref[0] < min(ids.shape[1], len(sessions)) for ref in refs):
        raise ValueError("The sessions of the decision are not loaded.")
    voter = SessionVoter(sessions, ids)

    a, b, whole = (_dense(voter, ref) for ref in (*decision.pieces, decision.whole))
    region = np.maximum(np.maximum(a, b), whole)
    size_a, size_b = max(float(a.sum()), 1.0), max(float(b.sum()), 1.0)

    piece_rows = [owner(ids, *ref) for ref in decision.pieces]
    whole_row = owner(ids, *decision.whole)

    steps = []
    for session_id in range(ids.shape[1]):
        if active is not None and not active(session_id):
            continue

        supports, sizes, included = voter._session_supports(session_id)
        inside = np.asarray(supports.T @ region).ravel()
        members = np.flatnonzero(
            included & (sizes > 0) & (inside >= INSIDE_FRACTION * sizes)
        )
        if not len(members):
            continue

        columns = supports[:, members]
        cover_a = np.asarray(columns.T @ a).ravel() / size_a
        cover_b = np.asarray(columns.T @ b).ravel() / size_b
        joint = (cover_a >= COVER_FRACTION) & (cover_b >= COVER_FRACTION)
        only_a = (cover_a >= COVER_FRACTION) & ~joint
        only_b = (cover_b >= COVER_FRACTION) & ~joint
        separate = only_a.any() and only_b.any()

        def step(state, targets=(), sources=(), note=""):
            steps.append(DecisionStep(session_id, state, targets, sources, note))

        if decision.kind == "split":
            if not joint.any():
                if separate:
                    step(DONE)
                continue

            footprint = int(members[joint][np.argmax(inside[members][joint])])
            row = owner(ids, session_id, footprint)
            if row is None:
                continue

            target = NeuronComponent(row, session_id)
            if None in piece_rows:
                step(CONFLICT, (target,), note="a blueprint piece is no longer assigned")
                continue
            if piece_rows[0] == piece_rows[1]:
                step(
                    CONFLICT, (target,),
                    note=f"both blueprint pieces now belong to neuron {piece_rows[0]}",
                )
                continue

            occupied = [
                (r, int(ids[r, session_id]))
                for r in piece_rows
                if r != row and ids[r, session_id] >= 0
            ]
            if occupied:
                r, f = occupied[0]
                step(
                    CONFLICT, (target,),
                    note=f"neuron {r} already has footprint {f} here",
                )
                continue

            sources = tuple(
                NeuronComponent(r, ref[0]) for r, ref in zip(piece_rows, decision.pieces)
            )
            step(APPLY, (target,), sources)

        else:
            if not separate:
                if joint.any():
                    step(DONE)
                continue

            first = int(members[only_a][np.argmax(cover_a[only_a])])
            second = int(members[only_b][np.argmax(cover_b[only_b])])
            rows = [owner(ids, session_id, first), owner(ids, session_id, second)]
            if None in rows:
                continue

            # The merged footprint goes to the first target's neuron: the
            # blueprint whole's neuron if it holds one of the pieces.
            if rows[1] == whole_row:
                rows.reverse()
            targets = tuple(NeuronComponent(r, session_id) for r in rows)

            if whole_row is None:
                step(CONFLICT, targets, note="the blueprint whole is no longer assigned")
                continue
            if session_id == decision.whole[0]:
                step(CONFLICT, targets, note="the blueprint whole's own session")
                continue

            step(APPLY, targets, (NeuronComponent(whole_row, decision.whole[0]),))

    return steps


def summary(steps):
    """(to apply, already as decided, conflicts) of a plan."""
    counts = {APPLY: 0, DONE: 0, CONFLICT: 0}
    for step in steps:
        counts[step.state] += 1
    return counts[APPLY], counts[DONE], counts[CONFLICT]


def matching_decision(voter, decisions, pieces):
    """The decision whose blueprint pieces are ``pieces`` (two footprints,
    as NeuronComponents in the voter's assignments), or None: each piece
    covers one blueprint piece and lies mostly on it."""
    if not decisions:
        return None

    given = [voter._support(piece) for piece in pieces]
    sizes = [max(float(s.sum()), 1.0) for s in given]

    for decision in decisions:
        try:
            blueprint = [_dense(voter, ref) for ref in decision.pieces]
        except ValueError:
            continue
        blueprint_sizes = [max(float(s.sum()), 1.0) for s in blueprint]

        def fits(i, j):
            shared = float(given[i] @ blueprint[j])
            return shared / min(sizes[i], blueprint_sizes[j]) >= COVER_FRACTION

        if (fits(0, 0) and fits(1, 1)) or (fits(0, 1) and fits(1, 0)):
            return decision

    return None
