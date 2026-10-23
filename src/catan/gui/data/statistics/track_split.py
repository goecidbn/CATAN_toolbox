"""Best two-group partition of each neuron's footprints by centroid position.

A neuron assembled from two distinct cells has two groups of footprints
that are close within and far apart between groups: a single cut in time
(cascaded mismatch) or interleaved sessions (alternating assignments).

Per neuron, the partition maximizing the gap (smallest distance between
the groups) is chosen over all partitions with at least ``min_group``
footprints per group (exact, by single linkage). A cut in time is one such
partition, so every temporal split is found as well.

Values are defined for footprint pairs across the groups only, oriented
from the group containing the first detection (session_i) to the other
(session_j), so relationship sources and targets are the two groups.
"""

import numpy as np

from .calculations import drop_indexed_axes, requested_indices
from .distance_statistics import geometry

MEASURES = ("gap", "separation")


def measure_choices(data):
    return MEASURES


def _tree(dist):
    """Minimum spanning tree edges (Prim), as (length, a, b)."""
    k = len(dist)
    inside = np.zeros(k, dtype=bool)
    inside[0] = True
    closest = dist[0].copy()
    parent = np.zeros(k, dtype=int)
    edges = []

    for _ in range(k - 1):
        candidates = np.where(inside, np.inf, closest)
        node = int(np.argmin(candidates))
        edges.append((float(closest[node]), int(parent[node]), node))
        inside[node] = True

        closer = dist[node] < closest
        closest[closer] = dist[node, closer]
        parent[closer] = node

    return edges


def _components(edges, k):
    """Component label of each node given tree edges (union-find)."""
    parent = list(range(k))

    def root(node):
        while parent[node] != node:
            parent[node] = parent[parent[node]]
            node = parent[node]
        return node

    for _, a, b in edges:
        parent[root(a)] = root(b)

    return np.array([root(node) for node in range(k)])


def _balanced_subset(sizes, min_group):
    """Component indices whose sizes sum closest to half (>= min_group on
    both sides), or None."""
    k = sum(sizes)

    # reachable[total] = component indices summing to total
    reachable = {0: ()}

    for index, size in enumerate(sizes):
        for total, chosen in list(reachable.items()):
            reachable.setdefault(total + size, chosen + (index,))

    admissible = [t for t in reachable if min_group <= t <= k - min_group]

    if not admissible:
        return None

    return reachable[min(admissible, key=lambda t: (abs(2 * t - k), t))]


def _clusters(dist, min_group):
    """Exact maximum gap (single linkage): remove the longest tree edges
    until the components can form two admissible groups. The most balanced
    grouping is used; the group of the first detection stays first."""
    k = len(dist)
    edges = sorted(_tree(dist), key=lambda edge: -edge[0])

    for removed in range(1, len(edges) + 1):
        labels = _components(edges[removed:], k)
        roots = np.unique(labels)
        chosen = _balanced_subset(
            [int((labels == r).sum()) for r in roots], min_group
        )

        if chosen is None:
            continue

        second = np.isin(labels, roots[list(chosen)])
        return ~second if second[0] else second

    return None


def split_values(positions, *, measure="gap", min_group=2):
    """(neuron, session_i, session_j) values; NaN outside cross pairs."""
    if measure not in MEASURES:
        raise ValueError(f"Unknown measure: {measure!r}")

    min_group = int(min_group)

    if min_group < 1:
        raise ValueError("The minimum group size must be at least 1.")

    n, s, _ = positions.shape
    values = np.full((n, s, s), np.nan)

    for neuron in range(n):
        sessions = np.flatnonzero(np.isfinite(positions[neuron, :, 0]))

        if len(sessions) < 2 * min_group:
            continue

        points = positions[neuron, sessions].astype(np.float64)
        dist = np.linalg.norm(points[:, None] - points[None], axis=-1)

        second = _clusters(dist, min_group)

        if second is None:
            continue

        if measure == "gap":
            value = dist[np.ix_(~second, second)].min()
        else:
            value = np.linalg.norm(
                np.median(points[~second], axis=0) - np.median(points[second], axis=0)
            )

        values[neuron][np.ix_(sessions[~second], sessions[second])] = value

    return values


def calculate_track_split(
    data,
    state,
    indexers=None,
    filters=(),
    measure="gap",
    min_group=2,
):
    indexers = indexers or {}
    positions, _ = geometry(data, state)
    n, s = positions.shape[:2]

    neurons = requested_indices(n, "neuron", indexers)

    # The partition depends on all of a neuron's sessions.
    values = split_values(
        positions[neurons],
        measure=measure,
        min_group=min_group,
    )

    if "session_i" in indexers:
        values = values[:, [indexers["session_i"]]]
    if "session_j" in indexers:
        values = values[:, :, [indexers["session_j"]]]

    return drop_indexed_axes(values, ("neuron", "session_i", "session_j"), indexers)
