from types import SimpleNamespace

import numpy as np
import pytest

pytest.importorskip("PySide6")
pytestmark = pytest.mark.gui

from catan.gui.data.statistics.engine import active_session_view  # noqa: E402


class _Data(SimpleNamespace):
    # As Data: the current assignment set.
    @property
    def assignments(self):
        return self._assignments[self._current_assignments]


def _data(ids, included, neuron_included, active=(True, True)):
    sessions = [
        SimpleNamespace(active=a, included=np.asarray(flags, dtype=bool))
        for a, flags in zip(active, included)
    ]
    assignments = SimpleNamespace(
        ids=np.asarray(ids),
        stats={},
        union=SimpleNamespace(included=np.asarray(neuron_included, dtype=bool)),
    )
    data = _Data(
        sessions=sessions,
        _assignments={"local": assignments},
        _current_assignments="local",
    )
    state = SimpleNamespace(assignments=np.asarray(ids))
    return data, state


def test_excluded_footprints_and_neurons_are_absent():
    ids = [[0, 0], [1, 1], [2, -1]]
    # S0 footprint 1 excluded; neuron 2 excluded as a whole.
    data, state = _data(ids, [[1, 0, 1], [1, 1]], [True, True, False])
    bound_data, bound_state = active_session_view(data, state)
    expected = [[0, 0], [-1, 1], [-1, -1]]
    assert np.array_equal(bound_state.assignments, expected)
    assert np.array_equal(bound_data.assignments.ids, expected)
    # The original data is untouched.
    assert np.array_equal(data.assignments.ids, ids)


def test_nothing_masked_returns_the_originals():
    data, state = _data([[0, 0]], [[1], [1]], [True])
    assert active_session_view(data, state) == (data, state)


def test_inactive_sessions_still_masked():
    data, state = _data([[0, 0]], [[1], [1]], [True], active=(True, False))
    _, bound_state = active_session_view(data, state)
    assert np.array_equal(bound_state.assignments, [[0, -1]])
