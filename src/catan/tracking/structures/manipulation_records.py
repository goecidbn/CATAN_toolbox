"""Footprint-keyed manipulation records (format 2).

Neuron rows are not stable (empty rows are removed, the rest renumbered),
footprints are: a footprint is ``(session_id, footprint_id)``. A record
stores the neurons a manipulation changed as *groups* of footprints, before
and after, plus the footprints it created (synthetic) and whose inclusion
it changed. From these, undo can restore the earlier grouping of exactly
the footprints the manipulation moved, leaving everything else as it is:

- ``moved``: footprints that left the neuron they continued with. Each
  after-group is matched to the before-group it shares most footprints
  with (one to one); a footprint is moved if its after-group is not
  matched to its before-group.
- ``anchors`` of a before-group: its footprints that were not moved; on
  undo, the moved footprints go back to the neuron now holding them.

Records written before (no ``format``) only describe the session-level
effects (split/merge pieces, replaced shapes) and cannot be undone.
"""

from collections import Counter
from dataclasses import dataclass

import numpy as np

FORMAT = 2


def ref(session_id, footprint_id):
    return (int(session_id), int(footprint_id))


@dataclass
class FootprintSnapshot:
    """Assignment rows, inclusion and column counts before an edit."""

    ids: np.ndarray
    included: list
    n_columns: list
    review_status: np.ndarray

    @classmethod
    def capture(cls, assignments, sessions):
        return cls(
            ids=np.array(assignments.ids, copy=True),
            included=[
                None if s is None or s.included is None else np.array(s.included, dtype=bool)
                for s in sessions
            ],
            n_columns=[
                0 if s is None or s.included is None else len(s.included) for s in sessions
            ],
            review_status=np.array(assignments.review_status, copy=True),
        )


def _row_groups(ids, rows):
    groups = []
    for row in rows:
        sessions = np.flatnonzero(ids[row] >= 0)
        groups.append(tuple(ref(s, ids[row, s]) for s in sessions))
    return groups


def _changed_rows(ids_a, ids_b):
    """Rows of ``ids_a`` (with footprints) whose exact footprint set does
    not occur as a row of ``ids_b``."""
    present_b = {row.tobytes() for row in ids_b}
    return [
        row
        for row in range(len(ids_a))
        if np.any(ids_a[row] >= 0) and ids_a[row].tobytes() not in present_b
    ]


def footprint_diff(before: FootprintSnapshot, assignments, sessions) -> dict:
    """The format-2 fields of the change from ``before`` to the current
    ``assignments`` / ``sessions``."""
    ids_after = np.asarray(assignments.ids)
    ids_before = before.ids
    if ids_before.shape[1] != ids_after.shape[1]:
        raise ValueError("Sessions changed during the manipulation.")

    rows_before = _changed_rows(ids_before, ids_after)
    rows_after = _changed_rows(ids_after, ids_before)

    created, excluded, included = [], [], []
    for session_id, session in enumerate(sessions):
        if session is None or session.included is None:
            continue
        flags = np.asarray(session.included, dtype=bool)
        n_before = before.n_columns[session_id]
        created += [ref(session_id, f) for f in range(n_before, len(flags))]
        old = before.included[session_id]
        if old is None or not n_before:
            continue
        common = flags[:n_before]
        excluded += [ref(session_id, f) for f in np.flatnonzero(old & ~common)]
        included += [ref(session_id, f) for f in np.flatnonzero(~old & common)]

    return {
        "format": FORMAT,
        "before": _row_groups(ids_before, rows_before),
        "after": _row_groups(ids_after, rows_after),
        "review_before": [int(before.review_status[row]) for row in rows_before],
        "created": created,
        "excluded": excluded,
        "included": included,
    }


def attach_diff(assignments, first_id, diff, *, manipulation_type=None):
    """Attach ``diff`` to the records registered since ``first_id`` (the
    last one is the main record; side records, e.g. a detached occupying
    footprint, become its parts). Registers a record if none was."""
    new_ids = [i for i in sorted(assignments.manipulations) if i >= first_id]
    if not new_ids:
        main = assignments.register_manipulation(
            manipulation_type=manipulation_type or "manipulation",
            origin=[],
            sources=[],
            results=[],
            affected_neurons=[],
        )
        new_ids = [main]
    main = new_ids[-1]
    assignments.manipulations[main].update(diff)
    for other in new_ids[:-1]:
        assignments.manipulations[other]["part_of"] = main
    return main


def is_new(record):
    return int(record.get("format", 1)) >= FORMAT


def is_undone(record):
    return bool(record.get("undone", False))


def is_part(record):
    return record.get("part_of") is not None


def batch_of(record):
    """ID of the curator batch the record was applied in (None: single)."""
    return record.get("batch")


def tag_batch(manipulations, first_id):
    """Records registered since ``first_id`` belong to one batch (its ID:
    ``first_id``)."""
    for record_id, record in manipulations.items():
        if record_id >= first_id:
            record["batch"] = int(first_id)


def listed(record):
    """Shown as a manipulation of its own (not a part of another)."""
    return not is_part(record)


def undoable(record):
    return is_new(record) and not is_undone(record) and not is_part(record)


def _match(before, after):
    """after index -> before index (greedy by shared footprints)."""
    owner = {}
    for b, group in enumerate(before):
        for footprint in group:
            owner[footprint] = b
    overlaps = Counter()
    for a, group in enumerate(after):
        for footprint in group:
            b = owner.get(footprint)
            if b is not None:
                overlaps[(a, b)] += 1
    matched, used_a, used_b = {}, set(), set()
    for (a, b), _ in sorted(overlaps.items(), key=lambda item: (-item[1], item[0])):
        if a in used_a or b in used_b:
            continue
        matched[a] = b
        used_a.add(a)
        used_b.add(b)
    return matched


def _groups(record, key):
    return [tuple(map(tuple, group)) for group in record.get(key, [])]


def moved(record):
    """Footprints that left the neuron they continued with."""
    before, after = _groups(record, "before"), _groups(record, "after")
    matched = _match(before, after)
    origin = {fp: b for b, group in enumerate(before) for fp in group}
    result = set()
    for a, group in enumerate(after):
        for footprint in group:
            b = origin.get(footprint)
            if b is not None and matched.get(a) != b:
                result.add(footprint)
    # Footprints of before-groups that appear in no after-group (should not
    # happen: every footprint stays assigned) count as moved too.
    in_after = {fp for group in after for fp in group}
    result |= {fp for group in before for fp in group if fp not in in_after}
    return result


def created(record):
    return {tuple(x) for x in record.get("created", [])}


def changed_footprints(record):
    """Footprints this record moved or created."""
    return moved(record) | created(record)


def anchors(record):
    """Per before-group: its footprints that stayed (not moved)."""
    gone = moved(record)
    return [tuple(fp for fp in group if fp not in gone) for group in _groups(record, "before")]


def _vacated(record):
    """(anchor set, session) of every slot a moved footprint left."""
    gone = moved(record)
    result = []
    for group, anchor in zip(_groups(record, "before"), anchors(record)):
        for footprint in group:
            if footprint in gone and anchor:
                result.append((set(anchor), footprint[0]))
    return result


def _fills(later, vacated):
    """``later`` moved a footprint into a vacated slot: a group of its
    after-state with one of the anchors and the moved footprint in that
    session."""
    gone = changed_footprints(later)
    for group in _groups(later, "after"):
        members = set(group)
        for anchor, session in vacated:
            if members & anchor and any(fp in gone and fp[0] == session for fp in group):
                return True
    return False


def dependents(manipulations, record_id):
    """Later (undoable) records that depend on ``record_id``, oldest first:
    they moved a footprint it moved or created, or filled a slot it
    vacated (transitively)."""
    record = manipulations[record_id]
    touched = set(changed_footprints(record))
    vacated = _vacated(record)
    result = []
    for later_id in sorted(i for i in manipulations if i > record_id):
        later = manipulations[later_id]
        if not undoable(later):
            continue
        footprints = changed_footprints(later)
        if footprints & touched or _fills(later, vacated):
            result.append(later_id)
            touched |= footprints
            vacated += _vacated(later)
    return result


def retire_from_session(manipulations, from_session_id):
    """Records touching footprints of sessions ``>= from_session_id`` (about
    to be registered again) become old-style records (listed, not undoable)."""
    if from_session_id is None:
        return
    for record in manipulations.values():
        if not is_new(record):
            continue
        footprints = [
            fp
            for key in ("before", "after")
            for group in record.get(key, [])
            for fp in group
        ] + list(record.get("created", []))
        if any(int(fp[0]) >= int(from_session_id) for fp in footprints):
            record["format"] = 1


def parts(manipulations, record_id):
    return [i for i, r in manipulations.items() if r.get("part_of") == record_id]


# --------------------------------------------------------------------
# Serialization (arrays, as the HDF5 writer stores them)
# --------------------------------------------------------------------

def _groups_array(groups):
    rows = [(g, s, f) for g, group in enumerate(groups) for s, f in group]
    return np.asarray(rows, dtype=np.int64).reshape(-1, 3)


def _groups_from_array(values):
    values = np.asarray(values, dtype=np.int64).reshape(-1, 3)
    groups = {}
    for g, s, f in values:
        groups.setdefault(int(g), []).append(ref(s, f))
    return [tuple(groups[g]) for g in sorted(groups)]


def _refs_array(refs):
    return np.asarray([tuple(r) for r in refs], dtype=np.int64).reshape(-1, 2)


def _refs_from_array(values):
    return [ref(s, f) for s, f in np.asarray(values, dtype=np.int64).reshape(-1, 2)]


def serialize_fields(record) -> dict:
    """Format-2 fields and flags of a record as arrays ({} for old ones
    that were never undone)."""
    flags = np.asarray(
        [
            int(record.get("format", 1)),
            int(bool(record.get("undone", False))),
            -1 if record.get("part_of") is None else int(record["part_of"]),
            -1 if record.get("batch") is None else int(record["batch"]),
        ],
        dtype=np.int64,
    )
    if not is_new(record) and not is_undone(record) and not is_part(record):
        return {}
    fields = {"flags": flags}
    if is_new(record):
        fields.update(
            fp_before=_groups_array(record.get("before", [])),
            fp_after=_groups_array(record.get("after", [])),
            review_before=np.asarray(record.get("review_before", []), dtype=np.int64),
            fp_created=_refs_array(record.get("created", [])),
            fp_excluded=_refs_array(record.get("excluded", [])),
            fp_included=_refs_array(record.get("included", [])),
        )
    return fields


def deserialize_fields(data) -> dict:
    if "flags" not in data:
        return {}
    flags = np.asarray(data["flags"], dtype=np.int64).ravel()
    fields = {"format": int(flags[0])}
    if flags[1]:
        fields["undone"] = True
    if flags[2] >= 0:
        fields["part_of"] = int(flags[2])
    if len(flags) > 3 and flags[3] >= 0:  # files before batches: 3 flags
        fields["batch"] = int(flags[3])
    if fields["format"] >= FORMAT:
        fields.update(
            before=_groups_from_array(data.get("fp_before", np.empty((0, 3)))),
            after=_groups_from_array(data.get("fp_after", np.empty((0, 3)))),
            review_before=[
                int(v) for v in np.asarray(data.get("review_before", []), dtype=np.int64).ravel()
            ],
            created=_refs_from_array(data.get("fp_created", np.empty((0, 2)))),
            excluded=_refs_from_array(data.get("fp_excluded", np.empty((0, 2)))),
            included=_refs_from_array(data.get("fp_included", np.empty((0, 2)))),
        )
    return fields
