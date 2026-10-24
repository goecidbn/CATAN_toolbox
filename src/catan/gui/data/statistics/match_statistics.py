"""Match probability: registration probability of a session footprint for
the union of its own neuron (pair statistic footprint → union)."""

from types import SimpleNamespace

import numpy as np

from .queries import ReductionSpec
from .sparse_pairs import SparsePairCollector
from .union_similarity import footprint_union_requested_pairs

NAMES = ("neuron_i", "neuron_j", "session")


def match_probability_values(data, ni, nj, sid):
    """p_matched of footprint (ni, session) for its own union (ni = nj);
    NaN for other unions and for footprints without a registration
    probability (manual assignments)."""
    ni, nj, sid = (np.asarray(values, dtype=np.int64) for values in (ni, nj, sid))
    values = np.full(len(ni), np.nan)

    p_matched = data.assignments.stats.get("p_matched")
    if p_matched is None:
        return values

    own = (ni == nj) & (ni < p_matched.shape[0]) & (sid < p_matched.shape[1])
    values[own] = p_matched[ni[own], sid[own], 0]
    return values


def calculate_match_probability(data, state, indexers=None, filters=(), **parameters):
    """Rows (footprint of neuron_i in session, union of neuron_j = neuron_i)."""
    from .distance_statistics import pair_plan

    indexers = indexers or {}
    ids = np.asarray(state.assignments)
    n, s = ids.shape

    query = SimpleNamespace(
        reduction_dict=lambda: {
            dim: ReductionSpec("single", int(index)) for dim, index in indexers.items()
        },
        filters=tuple(filters),
    )
    dims, aliases, references, _, _ = pair_plan(NAMES, n, s, query)

    collector = SparsePairCollector(
        NAMES,
        dims,
        aliases,
        references,
        n,
        s,
        query,
        lambda values: np.ones(np.shape(values), dtype=bool),
    )

    if collector.relations.get("neuron", "all") == "different":
        return collector.result()

    own = np.intersect1d(collector.domain("neuron_i"), collector.domain("neuron_j"))

    for session_id in collector.domain("session").tolist():
        present = own[ids[own, session_id] >= 0]
        if not len(present):
            continue

        sid = np.full(len(present), session_id, dtype=np.int64)
        values = match_probability_values(data, present, present, sid)
        known = np.isfinite(values)

        if known.any():
            collector.emit(
                {"neuron_i": present[known], "neuron_j": present[known], "session": sid[known]},
                values[known],
            )

    return collector.result()


def match_probability_requested_pairs(data, state, query, *, requested_pairs, **parameters):
    """Requested rows (source footprint, target union: session -1)."""
    return footprint_union_requested_pairs(
        state,
        query,
        requested_pairs,
        lambda ni, nj, sid: match_probability_values(data, ni, nj, sid),
    )
