"""Session-pair relations relative to a neuron's detections.

``session_i`` is the compared session, ``session_j`` the reference. Unlike
"same" and "with previous", these relations do not link the two session
axes: both remain, and rows are filtered by a predicate. Detections are
read from the assignments of the reference neuron (neuron_j, or the shared
neuron of within-neuron statistics). The session itself never qualifies.
"""

import numpy as np

# relation -> menu label (session filter "with ...")
PREVIOUS_RELATIONS = {
    "first instance": "first instance",
    "previous instance": "previous instance",
    "all previous": "all previous",
}
FOLLOWING_RELATIONS = {
    "next instance": "next instance",
    "last instance": "last instance",
    "all following": "all following",
}

ORDER_RELATIONS = tuple(PREVIOUS_RELATIONS) + tuple(FOLLOWING_RELATIONS)

# Relations needing the reference neuron's detections.
NEURON_RELATIONS = (
    "first instance",
    "previous instance",
    "next instance",
    "last instance",
)

DESCRIPTIONS = {
    "first instance": "the first session in which the reference neuron is detected",
    "previous instance": "the most recent earlier session with a detection",
    "all previous": "every earlier session",
    "next instance": "the next later session with a detection",
    "last instance": "the last session in which the reference neuron is detected",
    "all following": "every later session",
}


def is_order_relation(relation):
    return relation in ORDER_RELATIONS


class DetectionReference:
    """Per-neuron reference sessions from assignment IDs (neurons × sessions)."""

    def __init__(self, ids):
        present = np.asarray(ids) >= 0
        n, s = present.shape
        sessions = np.arange(s)

        any_present = present.any(axis=1)
        self.first = np.where(any_present, present.argmax(axis=1), -1)
        self.last = np.where(any_present, s - 1 - present[:, ::-1].argmax(axis=1), -1)

        # previous[k, t]: latest detection before t (-1 if none);
        # next[k, t]: earliest detection after t (s if none).
        latest = np.maximum.accumulate(np.where(present, sessions, -1), axis=1)
        self.previous = np.full((n, s), -1, dtype=np.int64)
        self.previous[:, 1:] = latest[:, :-1]

        earliest = np.minimum.accumulate(
            np.where(present, sessions, s)[:, ::-1], axis=1
        )[:, ::-1]
        self.next = np.full((n, s), s, dtype=np.int64)
        self.next[:, :-1] = earliest[:, 1:]

    def reference(self, relation, neurons, sessions):
        """The single reference session (-1: none) for neuron relations."""
        neurons = np.asarray(neurons, dtype=np.int64)
        sessions = np.asarray(sessions, dtype=np.int64)

        if relation == "first instance":
            return self.first[neurons]
        if relation == "last instance":
            return self.last[neurons]
        if relation == "previous instance":
            return self.previous[neurons, sessions]
        if relation == "next instance":
            result = self.next[neurons, sessions]
            return np.where(result >= self.next.shape[1], -1, result)

        raise ValueError(f"{relation!r} has no single reference session.")


def relation_mask(relation, session_i, session_j, neurons=None, reference=None):
    """Rows (session_i, session_j[, reference neuron]) satisfying ``relation``."""
    session_i = np.asarray(session_i, dtype=np.int64)
    session_j = np.asarray(session_j, dtype=np.int64)

    if relation == "all previous":
        return session_j < session_i

    if relation == "all following":
        return session_j > session_i

    if relation not in NEURON_RELATIONS:
        raise ValueError(f"Unsupported session relation: {relation!r}")

    if neurons is None or reference is None:
        raise ValueError(
            f"Session relation {relation!r} needs the reference neuron of each pair."
        )

    shape = np.broadcast(session_i, session_j, np.asarray(neurons)).shape
    session_i, session_j, neurons = (
        np.broadcast_to(v, shape) for v in (session_i, session_j, np.asarray(neurons))
    )

    target = reference.reference(relation, neurons, session_i)

    return (session_j == target) & (session_j != session_i) & (target >= 0)
