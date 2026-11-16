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


def test_include_excluded_masks_inactive_sessions_only():
    ids = [[0, 0], [1, 1], [2, -1]]
    data, state = _data(
        ids, [[1, 0, 1], [1, 1]], [True, True, False], active=(True, False)
    )
    _, bound_state = active_session_view(data, state, include_excluded=True)
    assert np.array_equal(bound_state.assignments, [[0, -1], [1, -1], [2, -1]])


def test_occurrence_of_neurons_without_footprints_is_nan():
    from catan.gui.data.statistics.calculations import calculate_occurrence

    state = SimpleNamespace(assignments=np.asarray([[0, -1], [-1, -1], [1, 2]]))
    values = calculate_occurrence(None, state)
    assert np.array_equal(values[[0, 2]], [[1, 0], [1, 1]])
    assert np.all(np.isnan(values[1]))


def test_query_keeps_include_excluded():
    from catan.gui.data.statistics.queries import ReductionSpec, StatisticQuery

    query = StatisticQuery(
        "occurence", (("neuron", ReductionSpec("keep")),), include_excluded=True
    )
    assert StatisticQuery.from_dict(query.to_dict()) == query
    assert "include_excluded" not in StatisticQuery("occurence", ()).to_dict()


def _disk(dims, center, radius):
    y, x = np.mgrid[: dims[0], : dims[1]]
    distance = np.hypot(y - center[0], x - center[1])
    return np.clip(1 - distance / (radius + 1), 0, None).ravel()


def test_codetected_containment():
    from scipy import sparse

    from catan.gui.data.statistics.codetected_containment import (
        codetected_containment_values,
    )

    dims = (30, 30)
    cell = _disk(dims, (15, 15), 6)
    piece = _disk(dims, (13, 15), 2)  # inside the cell
    neighbour = _disk(dims, (15, 26), 3)  # beside it
    sessions = []
    for columns in ([cell], [cell, piece, neighbour], [cell]):
        footprints = sparse.csc_matrix(np.column_stack(columns))
        sessions.append(
            SimpleNamespace(
                dims=dims, footprints=footprints, included=np.ones(len(columns), bool)
            )
        )
    # Neuron 0: the cell in all sessions; 1: the piece, 2: the neighbour (S1).
    ids = np.asarray([[0, 0, 0], [-1, 1, -1], [-1, 2, -1]])
    data = SimpleNamespace(sessions=sessions)
    state = SimpleNamespace(assignments=ids)

    values = codetected_containment_values(
        data, state, [1, 2, 0, 0], [0, 0, 1, 0], [1, 1, 1, 1]
    )
    assert values[0] == pytest.approx(1.0)  # the piece lies inside the cell
    assert values[1] < 0.2  # the neighbour's core does not
    assert np.isnan(values[2])  # the shorter track is no reference
    assert values[3] == pytest.approx(1.0)  # own neuron: reference values
