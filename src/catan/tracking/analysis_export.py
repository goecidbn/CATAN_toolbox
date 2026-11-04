"""Assignments for activity analysis: detected footprints only.

Splits and merges replace detected footprints by synthetic ones (copies of
other sessions' footprints, remainders), which have no activity traces;
replaced shapes keep their footprint and traces but not their detected
shape. The analysis output undoes this per footprint, from the current
state (no replay of the manipulation records):

- every synthetic footprint is dropped;
- a detected footprint a split or merge consumed (origin of a record that
  is not undone, still excluded) goes back, whole, to the neuron whose
  synthetic footprint in its session overlaps it most. Score of an original
  O and a synthetic piece P: the share of O's intensity on P's support times
  the share of P's intensity on O's support (a split piece lies inside O:
  its share of O decides; a merged shape covers its originals: each
  original's share of it decides). Greedy, one original per neuron slot and
  one slot per original;
- originals that win no slot stay excluded in their own row (left out);
  neurons left without footprints are dropped;
- replaced shapes: the footprint keeps its neuron (its detected shape is
  what the session file holds).

The result is an ordinary assignments file (no manipulations) on the
sessions as detected; it opens in CATAN as an assignment set.
"""

from dataclasses import dataclass, field

import numpy as np
from scipy import sparse

ANALYSIS_SUFFIX = ".analysis.hdf5"
SHAPE_TYPES = ("split", "merge", "cut")


@dataclass
class AnalysisSummary:
    returned: int = 0  # consumed originals given back to a neuron
    left_out: int = 0  # consumed originals without a slot
    cleared_slots: int = 0  # neuron slots holding a synthetic footprint, now empty
    dropped_neurons: int = 0  # neurons left without footprints
    replaced_shapes: int = 0  # footprints with a replaced shape (kept, detected shape)
    pairs: list = field(default_factory=list)  # (session, original, neuron, score)

    @property
    def changed(self):
        return bool(
            self.returned or self.left_out or self.cleared_slots or self.replaced_shapes
        )

    def describe(self):
        if not self.changed:
            return "No split, merge or shape replacement: same as the curated assignments."
        lines = [
            f"{self.returned} detected footprint(s) returned to the neuron "
            "with the most overlapping split/merge result",
        ]
        if self.left_out:
            lines.append(f"{self.left_out} left out (another original overlapped more)")
        if self.cleared_slots:
            lines.append(f"{self.cleared_slots} neuron slot(s) without a detected footprint")
        if self.dropped_neurons:
            lines.append(f"{self.dropped_neurons} neuron(s) without footprints dropped")
        if self.replaced_shapes:
            lines.append(f"{self.replaced_shapes} replaced shape(s) back to the detected shape")
        return "\n".join(lines)


def detected_count(session):
    """Number of detected (non-synthetic) footprints of ``session``."""
    synthetic = getattr(session, "synthetic", None)
    n = int(session.n_neurons)
    if synthetic is None:
        return n
    synthetic = np.asarray(synthetic, dtype=bool)[:n]
    return int(np.argmax(synthetic)) if synthetic.any() else n


def consumed_footprints(assignments, sessions):
    """{session: set of detected footprints} consumed by a split or merge
    that is not undone and still excluded."""
    result = {}
    for record in assignments.manipulations.values():
        if record.get("type") not in SHAPE_TYPES or record.get("undone"):
            continue
        for ref in record.get("origin", []):
            session_id, fp = int(ref["session_id"]), int(ref["footprint_id"])
            session = sessions[session_id]
            if fp >= detected_count(session):
                continue  # a synthetic footprint split again
            if bool(np.asarray(session.included)[fp]):
                continue  # included again since
            result.setdefault(session_id, set()).add(fp)
    return result


def _original_shape(session, fp):
    replaced = getattr(session, "replaced_footprints", None) or {}
    original = replaced.get(int(fp))
    if original is not None:
        return sparse.csc_matrix(original)
    return sparse.csc_matrix(session.footprints)[:, int(fp)]


def overlap_score(original, piece):
    """Share of ``original``'s intensity on ``piece``'s support times the
    share of ``piece``'s intensity on ``original``'s support (columns)."""
    o = np.abs(sparse.csc_matrix(original).toarray().ravel())
    p = np.abs(sparse.csc_matrix(piece).toarray().ravel())
    o_total, p_total = o.sum(), p.sum()
    if o_total <= 0 or p_total <= 0:
        return 0.0
    return float(o[p > 0].sum() / o_total * p[o > 0].sum() / p_total)


def analysis_ids(assignments, sessions):
    """(ids, kept rows, inclusion per session [detected footprints only],
    AnalysisSummary) of the analysis output of ``assignments``."""
    ids = np.array(assignments.ids, dtype=int, copy=True)
    summary = AnalysisSummary()
    consumed = consumed_footprints(assignments, sessions)

    included = []
    for session_id, session in enumerate(sessions):
        n_detected = detected_count(session)
        flags = np.asarray(session.included, dtype=bool)[:n_detected].copy()
        included.append(flags)
        summary.replaced_shapes += sum(
            1 for fp in (getattr(session, "replaced_footprints", None) or {}) if fp < n_detected
        )

        column = ids[:, session_id]
        slots = np.flatnonzero(column >= n_detected)  # synthetic footprints
        synthetic_included = np.asarray(session.included, dtype=bool)
        live = [
            int(row)
            for row in slots
            if column[row] < len(synthetic_included) and synthetic_included[column[row]]
        ]
        originals = sorted(consumed.get(session_id, ()))

        scores = []
        if live and originals:
            footprints = sparse.csc_matrix(session.footprints)
            for fp in originals:
                shape = _original_shape(session, fp)
                for row in live:
                    score = overlap_score(shape, footprints[:, int(column[row])])
                    if score > 0:
                        scores.append((score, fp, row))

        # Every synthetic slot empties (also excluded synthetic singletons).
        ids[slots, session_id] = -1

        given, filled = set(), set()
        for score, fp, row in sorted(scores, key=lambda item: (-item[0], item[1], item[2])):
            if fp in given or row in filled:
                continue
            # Out of its excluded singleton row, into the neuron.
            ids[ids[:, session_id] == fp, session_id] = -1
            ids[row, session_id] = fp
            flags[fp] = True
            given.add(fp)
            filled.add(row)
            summary.pairs.append((session_id, fp, row, score))

        # Left out: they stay excluded where they are.
        summary.left_out += len(originals) - len(given)
        summary.returned += len(given)
        summary.cleared_slots += len(set(live) - filled)

    # Dropped neurons: rows that had an included footprint (not the
    # excluded singletons of synthetic footprints).
    old = np.asarray(assignments.ids)
    had_footprints = np.zeros(len(old), dtype=bool)
    for session_id, session in enumerate(sessions):
        flags = np.asarray(session.included, dtype=bool)
        column = old[:, session_id]
        present = column >= 0
        had_footprints[present] |= flags[column[present]]
    has_footprints = np.any(ids >= 0, axis=1)
    summary.dropped_neurons = int(np.sum(had_footprints & ~has_footprints))
    keep = np.flatnonzero(has_footprints)
    return ids[keep], keep, included, summary


def analysis_inclusion(sessions, included):
    """Session inclusion document (as ``snapshot_inclusion``) of the
    sessions as detected."""
    records = {}
    for index, (session, flags) in enumerate(zip(sessions, included)):
        record = {
            "source_path": str(session.path or ""),
            "n_footprints": int(len(flags)),
        }
        if np.any(~flags):
            record["included"] = np.asarray(flags, dtype=bool)
        records[f"s{index:06d}"] = record
    return {"version": 1, "sessions": records}


def analysis_assignments(assignments, sessions):
    """(Assignments, AnalysisSummary): a copy of ``assignments`` with the
    analysis ids, no manipulations and the inclusion of the sessions as
    detected (``session_included``; save with ``prepare_save()`` without
    sessions). Statistics of changed slots are NaN."""
    if assignments.ids.shape[1] != len(sessions):
        raise ValueError("The assignments do not match the sessions.")
    ids, keep, included, summary = analysis_ids(assignments, sessions)

    result = assignments.copy()
    changed = ids != np.asarray(assignments.ids)[keep]

    result.ids = ids
    result.review_status = np.asarray(assignments.review_status)[keep].copy()
    stats = {}
    for name, values in assignments.stats.items():
        values = np.array(np.asarray(values)[keep], dtype=float)
        values[changed] = np.nan
        stats[name] = values
    result.stats = stats
    result.matched_status = np.any(ids >= 0, axis=0)
    result.manipulations = {}
    result.manipulation_id = {}
    result.shape_decisions = {}
    result.next_manipulation_id = 0
    result.session_included = analysis_inclusion(sessions, included)
    result.path = ""
    return result, summary


def prepare_analysis_save(assignments, sessions, *, provenance=None):
    """(prepared save {"data", "fields", "attributes"}, AnalysisSummary)."""
    result, summary = analysis_assignments(assignments, sessions)
    prepared = result.prepare_save(provenance=provenance)
    prepared["attributes"]["catan_analysis_export"] = 1
    return prepared, summary
