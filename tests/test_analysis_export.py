"""Analysis output: detected footprints only (tracking/analysis_export)."""

from types import SimpleNamespace

import numpy as np
from scipy import sparse

from catan.tracking.analysis_export import analysis_assignments, overlap_score
from catan.tracking.structures import Assignments

N_PIXELS = 100


def _column(pixels, value=1.0):
    column = np.zeros(N_PIXELS)
    column[list(pixels)] = value
    return column


def _session(columns, *, n_detected, included):
    footprints = sparse.csc_matrix(np.stack(columns, axis=1))
    n = footprints.shape[1]
    return SimpleNamespace(
        name="s",
        path=None,
        n_neurons=n,
        footprints=footprints,
        included=np.asarray(included, dtype=bool),
        synthetic=np.arange(n) >= n_detected,
        replaced_footprints={},
    )


def _scenario():
    """Session 1: O (0) split into A (4, neuron 0) and B (5, neuron 1) plus a
    small remainder R (7, neuron 7); M1 (1) and M2 (2) merged into U (6,
    neuron 2); X (3) excluded by the user (its split was undone)."""
    a, b = range(0, 30), range(30, 50)
    m1, m2 = range(50, 80), range(80, 90)
    x = range(90, 100)
    s0 = _session([_column(range(0, 10)), _column(range(10, 20)), _column(range(20, 30)),
                   _column(range(30, 40))], n_detected=4, included=[1, 1, 1, 1])
    s1 = _session(
        [
            _column([*a, *b]),  # 0 O (split origin)
            _column(m1),  # 1 M1 (merge origin)
            _column(m2),  # 2 M2 (merge origin)
            _column(x),  # 3 X (user exclusion)
            _column(a),  # 4 A (synthetic)
            _column(b),  # 5 B (synthetic)
            _column([*m1, *m2]),  # 6 U (synthetic)
            _column(range(45, 50)),  # 7 R (synthetic remainder)
        ],
        n_detected=4,
        included=[0, 0, 0, 0, 1, 1, 1, 1],
    )
    assignments = Assignments()
    assignments.ids = np.array(
        [
            [0, 4],  # neuron 0
            [1, 5],  # neuron 1
            [2, 6],  # neuron 2
            [-1, 0],  # excluded singletons
            [-1, 1],
            [-1, 2],
            [-1, 3],
            [-1, 7],  # neuron 7: the remainder only
            [3, -1],  # neuron 8
        ]
    )
    n = len(assignments.ids)
    assignments.review_status = np.arange(n, dtype=np.uint8) % 3
    assignments.stats = {
        "p_matched": np.full((n, 2, 2), 0.9),
        "shifts": np.zeros((n, 2, 2)),
    }

    def ref(fp, neuron=0):
        return {"session_id": 1, "footprint_id": fp, "neuron_id": neuron}

    assignments.manipulations = {
        0: {"type": "split", "origin": [ref(0)], "sources": [], "results": []},
        1: {"type": "merge", "origin": [ref(1), ref(2)], "sources": [], "results": []},
        2: {"type": "split", "origin": [ref(3)], "sources": [], "results": [], "undone": True},
    }
    assignments.manipulation_id = {0: 0, 1: 0, 2: 1}
    assignments.next_manipulation_id = 3
    return assignments, [s0, s1]


def test_overlap_score_ranks_split_and_merge():
    whole, piece = _column(range(0, 50)), _column(range(0, 30))
    assert np.isclose(overlap_score(whole, piece), 0.6)
    merged = _column(range(50, 90))
    assert overlap_score(_column(range(50, 80)), merged) > overlap_score(
        _column(range(80, 90)), merged
    )


def test_analysis_output_returns_originals():
    assignments, sessions = _scenario()
    original_ids = assignments.ids.copy()
    result, summary = analysis_assignments(assignments, sessions)

    ids = result.ids
    rows = {tuple(row) for row in ids.tolist()}
    # Split: O to the neuron of the larger piece (A); B's neuron loses
    # session 1; the remainder's neuron disappears.
    assert (0, 0) in rows
    assert (1, -1) in rows
    # Merge: the larger original stays with the merged neuron, the other
    # stays excluded in its own row.
    assert (2, 1) in rows
    assert (-1, 2) in rows
    # The user's exclusion stays as it was.
    assert (-1, 3) in rows and (3, -1) in rows
    # Only detected footprints are referenced.
    assert ids[:, 1].max() < 4
    assert len(ids) == 6

    assert summary.returned == 2
    assert summary.left_out == 1
    assert summary.cleared_slots == 2  # B's and R's slots
    assert summary.dropped_neurons == 1  # the remainder's neuron
    assert summary.changed

    inclusion = result.session_included["sessions"]
    assert inclusion["s000001"]["n_footprints"] == 4
    assert inclusion["s000001"]["included"].tolist() == [True, True, False, False]
    assert "included" not in inclusion["s000000"]

    assert result.manipulations == {} and result.manipulation_id == {}
    # Rows keep their review status; changed slots lose their statistics.
    assert result.review_status.tolist() == [0, 1, 2, 2, 0, 2]
    assert np.isnan(result.stats["p_matched"][0, 1]).all()
    assert np.isnan(result.stats["p_matched"][1, 1]).all()
    assert np.allclose(result.stats["p_matched"][0, 0], 0.9)
    # The curated assignments are untouched.
    assert np.array_equal(assignments.ids, original_ids)
    assert len(assignments.manipulations) == 3


def test_without_manipulations_nothing_changes():
    assignments, sessions = _scenario()
    assignments.manipulations = {}
    sessions[1].synthetic[:] = False
    sessions[1].included[:] = True
    result, summary = analysis_assignments(assignments, sessions)
    assert not summary.changed
    assert np.array_equal(result.ids, assignments.ids)


def test_analysis_file_round_trip(tmp_path):
    from catan.core.io import save_file
    from catan.tracking.analysis_export import prepare_analysis_save

    assignments, sessions = _scenario()
    prepared, _ = prepare_analysis_save(assignments, sessions)
    assert prepared["attributes"]["catan_analysis_export"] == 1
    path = tmp_path / "x.analysis.hdf5"
    save_file(
        path, prepared["data"], prepared["fields"],
        root_attributes=prepared["attributes"], root="/",
    )
    import h5py

    with h5py.File(path, "r") as handle:
        assert handle.attrs["catan_analysis_export"] == 1
    loaded = Assignments._from_file(str(path))
    expected, _ = analysis_assignments(*_scenario())
    assert np.array_equal(np.asarray(loaded.ids), expected.ids)
    assert loaded.manipulations == {}
    assert loaded.session_included["sessions"]["s000001"]["included"].tolist() == [
        True, True, False, False,
    ]
