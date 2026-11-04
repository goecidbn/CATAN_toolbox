"""Shape decisions ("two cells" / "one cell"): plan per session, curator
vote override, saving with the assignments."""

from types import SimpleNamespace

import numpy as np
from scipy import sparse

from catan.core.structures import NeuronComponent
from catan.gui.data.manipulation_votes import SessionVoter
from catan.gui.data.shape_decisions import (
    APPLY,
    CONFLICT,
    DONE,
    ShapeDecision,
    matching_decision,
    plan,
    summary,
)

N_PIXELS = 100
A, B, FAR = range(0, 30), range(30, 50), range(80, 95)
WHOLE = range(0, 50)


def _column(pixels):
    column = np.zeros(N_PIXELS)
    column[list(pixels)] = 1.0
    return column


def _session(*pixel_sets):
    return SimpleNamespace(
        footprints=sparse.csc_matrix(np.stack([_column(p) for p in pixel_sets], axis=1)),
        included=np.ones(len(pixel_sets), dtype=bool),
    )


def _scenario():
    """Neurons 0 (cell A) and 1 (cell B); neuron 2 holds the whole in S1.
    S0, S2: A and B apart. S1: the whole (neuron 2). S3: the whole held by
    neuron 0. S4: the whole (neuron 2), but neuron 0 has a far footprint."""
    sessions = [
        _session(A, B),
        _session(WHOLE),
        _session(A, B),
        _session(WHOLE),
        _session(WHOLE, FAR),
    ]
    ids = np.full((3, 5), -1)
    ids[0] = [0, -1, 0, 0, 1]
    ids[1] = [1, -1, 1, -1, -1]
    ids[2] = [-1, 0, -1, -1, 0]
    return sessions, ids


def _steps(kind):
    sessions, ids = _scenario()
    decision = ShapeDecision(kind, (1, 0), ((0, 0), (0, 1)))
    return {step.session_id: step for step in plan(decision, sessions, ids)}


def test_two_cells_splits_every_whole():
    steps = _steps("split")
    assert steps[0].state == DONE and steps[2].state == DONE
    assert steps[1].state == APPLY
    assert steps[1].targets == (NeuronComponent(2, 1),)
    assert steps[1].sources == (NeuronComponent(0, 0), NeuronComponent(1, 0))
    # The whole held by neuron 0 itself: neuron 0's slot is the origin's.
    assert steps[3].state == APPLY and steps[3].targets == (NeuronComponent(0, 3),)
    # Neuron 0 already has another footprint in S4.
    assert steps[4].state == CONFLICT and "neuron 0" in steps[4].note
    assert summary(steps.values()) == (2, 2, 1)


def test_one_cell_merges_every_pair():
    steps = _steps("merge")
    assert steps[1].state == DONE and steps[3].state == DONE and steps[4].state == DONE
    for session_id in (0, 2):
        step = steps[session_id]
        assert step.state == APPLY
        assert step.targets == (NeuronComponent(0, session_id), NeuronComponent(1, session_id))
        assert step.sources == (NeuronComponent(2, 1),)


def test_decision_overrides_the_votes():
    sessions, ids = _scenario()
    whole, pieces = NeuronComponent(2, 1), (NeuronComponent(0, 0), NeuronComponent(1, 0))

    undecided = SessionVoter(sessions, ids).votes(whole, pieces)
    assert undecided.decided is None

    for kind, other in (("split", "merge"), ("merge", "split")):
        decision = ShapeDecision(kind, (1, 0), ((0, 0), (0, 1)))
        votes = SessionVoter(sessions, ids, [decision]).votes(whole, pieces)
        assert votes.decided == kind
        assert votes.supports(kind) and not votes.supports(other)
        assert votes.describe().startswith("decided:")

    # Pieces of another cell pair: no decision applies.
    voter = SessionVoter(sessions, ids)
    unrelated = ShapeDecision("split", (1, 0), ((4, 1), (4, 1)))
    assert matching_decision(voter, [unrelated], pieces) is None


def test_decisions_are_saved_with_the_assignments(tmp_path):
    from catan.core.io import save_file
    from catan.tracking.structures import Assignments

    assignments = Assignments()
    assignments.ids = np.array([[0, 1], [1, -1]])
    assignments.review_status = np.zeros(2, dtype=np.uint8)
    assignments.stats = {
        "p_matched": np.full((2, 2, 2), 0.9),
        "shifts": np.zeros((2, 2, 2)),
    }
    assignments.shape_decisions = {
        0: ShapeDecision("split", (1, 0), ((0, 0), (0, 1))),
        3: ShapeDecision("merge", (0, 1), ((1, 0), (1, 2))),
    }

    prepared = assignments.prepare_save()
    path = tmp_path / "decisions.hdf5"
    save_file(
        path, prepared["data"], prepared["fields"],
        root_attributes=prepared["attributes"], root="/",
    )
    loaded = Assignments._from_file(str(path))
    assert loaded.shape_decisions == assignments.shape_decisions
