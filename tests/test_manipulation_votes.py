"""Session votes deciding between a footprint split and a merge."""

from types import SimpleNamespace

import numpy as np
from scipy import sparse

from catan.core.structures import NeuronComponent as C
from catan.gui.data.manipulation_votes import SessionVoter

DIMS = (20, 20)


def disk(cx, cy, r=3):
    y, x = np.mgrid[: DIMS[0], : DIMS[1]]
    return ((x - cx) ** 2 + (y - cy) ** 2 <= r**2).astype(float).ravel()


def session(*footprints):
    return SimpleNamespace(
        footprints=sparse.csc_matrix(np.stack(footprints, axis=1)),
        included=np.ones(len(footprints), dtype=bool),
    )


A, B = disk(6, 10), disk(13, 10)
WHOLE = np.maximum(A, B)


def voter(*sessions):
    # Neuron 0: footprint 0 of every session, neuron 1: footprint 1.
    ids = np.full((2, len(sessions)), -1)
    for sid, s in enumerate(sessions):
        ids[: s.footprints.shape[1], sid] = np.arange(s.footprints.shape[1])
    return SessionVoter(sessions, ids)


def test_separate_cells_support_a_split():
    v = voter(session(WHOLE), session(A, B), session(A, B), session(WHOLE))
    votes = v.votes(C(0, 0), (C(0, 1), C(1, 1)))
    assert votes.votes == "SSJ"
    assert votes.supports("split") and not votes.supports("merge")


def test_joint_footprints_support_a_merge():
    v = voter(session(WHOLE), session(A, B), session(WHOLE), session(WHOLE))
    votes = v.votes(C(0, 0), (C(0, 1), C(1, 1)))
    assert (votes.joint, votes.separate) == (2, 1)
    assert votes.supports("merge") and not votes.supports("split")


def test_tie_supports_neither():
    v = voter(session(WHOLE), session(A, B), session(WHOLE))
    votes = v.votes(C(0, 0), (C(0, 1), C(1, 1)))
    assert not votes.supports("merge") and not votes.supports("split")


def test_footprints_outside_the_whole_abstain():
    v = voter(session(WHOLE), session(A, B), session(disk(3, 3, 2)))
    assert v.votes(C(0, 0), (C(0, 1), C(1, 1))).votes == "S."
