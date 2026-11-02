"""Footprint-keyed manipulation records (format 2): creation, undo of one
record and of a dependent chain, old records, save / load."""

from types import SimpleNamespace

import numpy as np

from catan.tracking.structures import Assignments
from catan.tracking.structures import manipulation_records as records


def _assignments(ids):
    assignments = Assignments()
    assignments.register_data(assignments={"ids": np.asarray(ids)})
    return assignments


def _sessions(n_sessions, n_footprints):
    return [
        SimpleNamespace(included=np.ones(n_footprints, dtype=bool))
        for _ in range(n_sessions)
    ]


class _Worker:
    """What curation_actions._undo_record needs of a Data worker."""

    def __init__(self, assignments, sessions):
        self.assignments = assignments
        self.sessions = sessions
        self.state = SimpleNamespace(assignments=assignments.ids)
        self.moves = []

    def update_moved_footprint_stats(self, moves):
        self.moves += list(moves)


def _record(assignments, sessions, edit, kind="reassign"):
    """Apply ``edit`` (on ids / sessions) as a recorded manipulation."""
    snapshot = records.FootprintSnapshot.capture(assignments, sessions)
    first = assignments.next_manipulation_id
    edit()
    assignments.register_manipulation(
        manipulation_type=kind, origin=[], sources=[], results=[], affected_neurons=[]
    )
    return records.attach_diff(
        assignments, first, records.footprint_diff(snapshot, assignments, sessions)
    )


def _groups(ids):
    """Neurons as sets of footprints (row numbers do not matter)."""
    return {
        frozenset((s, int(f)) for s, f in enumerate(row) if f >= 0)
        for row in np.asarray(ids)
        if np.any(row >= 0)
    }


def _undo(assignments, sessions, record_ids):
    from catan.gui.data.curation_actions import _undo_record

    worker = _Worker(assignments, sessions)
    notes = []
    for record_id in sorted(record_ids, reverse=True):
        _undo_record(worker, assignments.manipulations[record_id], lambda s: sessions[s], notes)
        assignments.manipulations[record_id]["undone"] = True
    return notes


IDS = [
    [0, 0, 0],  # neuron 0
    [1, 1, 1],  # neuron 1
    [2, 2, -1],  # neuron 2
]


def _move(assignments, source_row, target_row, session_id):
    def edit():
        ids = assignments.ids
        ids[target_row, session_id] = ids[source_row, session_id]
        ids[source_row, session_id] = -1

    return edit


def test_record_fields():
    a, sessions = _assignments(IDS), _sessions(3, 3)
    rid = _record(a, sessions, _move(a, 1, 2, 2))  # footprint (2, 1) to neuron 2
    record = a.manipulations[rid]
    assert records.is_new(record) and records.undoable(record)
    assert sorted(record["before"]) == [((0, 1), (1, 1), (2, 1)), ((0, 2), (1, 2))]
    assert sorted(record["after"]) == [((0, 1), (1, 1)), ((0, 2), (1, 2), (2, 1))]
    assert record["created"] == [] and record["excluded"] == []


def test_reassign_and_undo_restores_groups():
    a, sessions = _assignments(IDS), _sessions(3, 3)
    before = _groups(a.ids)
    rid = _record(a, sessions, _move(a, 1, 2, 2))  # footprint (2, 1) to neuron 2
    record = a.manipulations[rid]
    assert records.moved(record) == {(2, 1)}
    assert _groups(a.ids) != before

    notes = _undo(a, sessions, [rid])
    assert not notes
    assert _groups(a.ids) == before


def test_independent_records_undo_alone():
    ids = IDS + [[3, 3, 3], [4, -1, -1]]
    a, sessions = _assignments(ids), _sessions(3, 5)
    first = _record(a, sessions, _move(a, 1, 2, 2))  # (2,1) -> neuron 2
    second = _record(a, sessions, _move(a, 3, 4, 1))  # (1,3) -> neuron 4
    assert records.dependents(a.manipulations, first) == []
    after_second = _groups(a.ids)

    _undo(a, sessions, [first])
    groups = _groups(a.ids)
    # The second manipulation is still in place, the first one is not.
    assert frozenset({(0, 4), (1, 3)}) in groups
    assert frozenset({(0, 1), (1, 1), (2, 1)}) in groups
    assert groups != after_second
    assert records.undoable(a.manipulations[second])


def test_dependent_record_is_found_and_chain_undone():
    ids = IDS + [[-1, -1, -1]]
    a, sessions = _assignments(ids), _sessions(3, 3)
    before = _groups(a.ids)
    first = _record(a, sessions, _move(a, 1, 2, 2))  # (2,1) -> neuron 2
    second = _record(a, sessions, _move(a, 2, 3, 2))  # the same footprint on
    assert records.dependents(a.manipulations, first) == [second]

    _undo(a, sessions, [second, first])
    assert _groups(a.ids) == before


def test_filled_slot_is_a_dependency():
    ids = IDS + [[-1, -1, 5]]
    a, sessions = _assignments(ids), _sessions(3, 6)
    first = _record(a, sessions, _move(a, 1, 2, 2))  # neuron 1 loses session 2
    second = _record(a, sessions, _move(a, 3, 1, 2))  # (2,5) fills it
    assert records.dependents(a.manipulations, first) == [second]


def test_created_footprints_become_excluded_singletons():
    a, sessions = _assignments(IDS), _sessions(3, 3)
    before = _groups(a.ids)

    def split():
        # Footprint (1, 1) retired to an excluded singleton, a new piece
        # (1, 3) in its place.
        sessions[1].included = np.append(sessions[1].included, True)
        a.pad_empty(n_neurons=1, n_sessions=0)
        a.ids[3, 1] = 1
        sessions[1].included[1] = False
        a.ids[1, 1] = 3

    rid = _record(a, sessions, split, kind="split")
    record = a.manipulations[rid]
    assert records.created(record) == {(1, 3)}
    assert (1, 1) in record["excluded"]

    _undo(a, sessions, [rid])
    groups = _groups(a.ids)
    assert frozenset({(0, 1), (1, 1), (2, 1)}) in groups
    assert frozenset({(1, 3)}) in groups  # kept, excluded, alone
    assert sessions[1].included[1] and not sessions[1].included[3]
    assert before <= groups


def test_old_records_are_listed_not_undoable():
    a = _assignments(IDS)
    a.register_manipulation(
        manipulation_type="reassign", origin=[], sources=[], results=[], affected_neurons=[1]
    )
    record = a.manipulations[0]
    assert records.listed(record) and not records.undoable(record)
    # Old records serialize exactly as before (no new fields).
    serialized = a._serialize_manipulations()["0"]
    assert set(serialized) == {"type", "origin", "sources", "results"}


def test_save_load_round_trip(tmp_path):
    ids = IDS + [[-1, -1, -1]]
    a, sessions = _assignments(ids), _sessions(3, 3)
    first = _record(a, sessions, _move(a, 1, 2, 2))
    second = _record(a, sessions, _move(a, 2, 3, 2))
    a.manipulations[second]["undone"] = True
    a.register_manipulation(
        manipulation_type="merge", origin=[], sources=[], results=[], affected_neurons=[]
    )

    path = tmp_path / "set.assignments.hdf5"
    a.save(str(path))
    loaded = Assignments._from_file(str(path))

    for record_id in (first, second):
        original, restored = a.manipulations[record_id], loaded.manipulations[record_id]
        for key in ("format", "before", "after", "created", "excluded", "included"):
            assert list(original[key]) == list(restored[key]) if key != "format" else original[key] == restored[key], key
        assert records.moved(original) == records.moved(restored)
    assert loaded.manipulations[second].get("undone")
    assert records.undoable(loaded.manipulations[first])
    assert not records.is_new(loaded.manipulations[2])


def test_retire_from_session():
    a, sessions = _assignments(IDS), _sessions(3, 3)
    rid = _record(a, sessions, _move(a, 1, 2, 2))
    records.retire_from_session(a.manipulations, 1)
    assert not records.undoable(a.manipulations[rid])


def test_batch_tag_round_trip_and_old_flags():
    a, sessions = _assignments(IDS), _sessions(3, 3)
    single = _record(a, sessions, _move(a, 1, 2, 2))
    first = a.next_manipulation_id
    one = _record(a, sessions, _move(a, 0, 1, 2))
    two = _record(a, sessions, _move(a, 0, 2, 1))
    records.tag_batch(a.manipulations, first)
    assert records.batch_of(a.manipulations[single]) is None
    assert records.batch_of(a.manipulations[one]) == records.batch_of(a.manipulations[two]) == first

    fields = records.serialize_fields(a.manipulations[two])
    assert records.deserialize_fields(fields)["batch"] == first
    # Files written before batches have three flags.
    fields["flags"] = fields["flags"][:3]
    assert "batch" not in records.deserialize_fields(fields)


def test_undo_chain_of_several_records():
    from types import SimpleNamespace

    from catan.gui.data.curation_actions import undo_chain

    a, sessions = _assignments(IDS), _sessions(3, 3)
    first = _record(a, sessions, _move(a, 1, 2, 2))
    second = _record(a, sessions, _move(a, 0, 1, 0))
    data = SimpleNamespace(assignments=a)
    assert undo_chain(data, [first, second]) == tuple(sorted({first, second}, reverse=True))
    assert undo_chain(data, first)[-1] == first
