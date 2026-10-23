"""Sampled value distributions for threshold suggestions.

Strategies, in order of preference:

- pairs: pair statistics with an independent partner neuron and unreduced
  axes are evaluated on explicit same-neuron and nearest-other-neuron
  pairs (``requested_pairs``), drawn from a random subset of session pairs.
- full: cached results and cheap (foreground-safe) statistics use all values.
- neurons: neuron-local statistics with kept neuron axes are evaluated on
  growing random neuron subsets; all other assignment rows are masked, so
  every sampled row has exactly the value of the full evaluation.

Sampling stops once ``max_samples`` values are collected or ``budget``
seconds have passed (checked between evaluations).
"""

from copy import copy
from dataclasses import dataclass, replace
from functools import partial
from math import ceil
from time import perf_counter
from types import SimpleNamespace
from typing import Literal

import numpy as np
from scipy.spatial import cKDTree

from catan.gui.background_tasks.runtime import current_task_context, reports_progress

from .dimensions import canonical_dim
from .session_relations import (
    NEURON_RELATIONS,
    DetectionReference,
    is_order_relation,
    relation_mask,
)
from .queries import PairFilter, ReductionSpec
from .table import PickTable

SUGGESTION_PERCENTILES = (5, 25, 50, 75, 95)

# Session pairs evaluated between two budget checks.
SESSION_PAIR_CHUNK = 16
MIN_PAIRS_PER_SESSION_PAIR = 8
FIRST_NEURON_BATCH = 8


@dataclass(frozen=True, slots=True)
class ValueSample:
    label: str
    values: np.ndarray

    # Sampled units (values, neurons or source footprints) and how many
    # exist in total; None when the total is unknown.
    sampled: int
    available: int | None
    unit: str = "values"

    @property
    def exact(self) -> bool:
        return self.available is not None and self.sampled >= self.available

    def percentiles(self, levels=SUGGESTION_PERCENTILES):
        if not len(self.values):
            return None

        return np.percentile(np.asarray(self.values, dtype=np.float64), levels)


@dataclass(frozen=True, slots=True)
class StatisticSample:
    method: Literal["full", "neurons", "pairs", "unsupported"]
    samples: tuple[ValueSample, ...] = ()
    note: str = ""
    elapsed: float = 0.0


def _finite(values):
    values = np.asarray(values, dtype=np.float64).reshape(-1)
    return values[np.isfinite(values)]


def _compact(stat, definition):
    """Match the precision of the values the curator thresholds."""
    from .engine import _compact_statistic_values

    if definition.compact_values:
        _compact_statistic_values(stat)


def _base_getter(definition):
    getter = definition.query_getter

    while isinstance(getter, partial):
        getter = getter.func

    return getter


def _neuron_relation(query):
    for f in query.filters:
        if f.target == "neuron":
            return f.relation

    return "all"


def neuron_rows_local(definition, query) -> bool:
    """Rows are single neurons whose values depend only on that neuron."""
    axes = [dim for dim in definition.dims if canonical_dim(dim) == "neuron"]

    if not axes:
        return False

    reductions = query.reduction_dict()

    if any(reductions.get(axis, ReductionSpec("keep")).method != "keep" for axis in axes):
        return False

    return len(axes) == 1 or _neuron_relation(query) == "same"


def partner_endpoints(definition, query):
    """Relationship axes if the partner is an independent, unreduced neuron."""
    endpoints = definition.relationship_axes

    if endpoints is None or endpoints[0][0] == endpoints[1][0]:
        return None

    if _neuron_relation(query) == "same":
        return None

    if any(spec.method not in ("keep", "single") for spec in query.reduction_dict().values()):
        return None

    if definition.key == "footprint_similarity" or definition.pairs_getter is not None:
        return endpoints

    getter = _base_getter(definition)

    if not getattr(getter, "supports_requested_pairs", False):
        return None

    return endpoints


@reports_progress("Sampling statistic values")
def sample_statistic(
    engine,
    query,
    definition,
    *,
    max_samples=4000,
    budget=2.0,
    seed=0,
    allow_full=False,
) -> StatisticSample:
    started = perf_counter()
    rng = np.random.default_rng(seed)

    endpoints = partner_endpoints(definition, query)

    if endpoints is not None:
        result = _sample_pairs(
            engine, query, definition, endpoints, max_samples=max_samples, budget=budget, rng=rng
        )
    else:
        found, _ = engine.peek_cached(query)

        if found or definition.foreground_safe:
            result = _sample_full(engine, query)
        elif (
            definition.neuron_local
            and neuron_rows_local(definition, query)
            and engine.data.assignments is not None
        ):
            result = _sample_neurons(
                engine, query, definition, max_samples=max_samples, budget=budget, rng=rng
            )
        elif allow_full:
            result = _sample_full(engine, query)
        else:
            result = StatisticSample(
                "unsupported",
                note=(
                    "No fast sampling for this statistic and reduction. "
                    "Calculate all values instead (may be slow)."
                ),
            )

    return replace(result, elapsed=perf_counter() - started)


def _sample_full(engine, query):
    table = engine.evaluate_table(query)

    if table is None:
        return StatisticSample("unsupported", note="No values available.")

    values = _finite(table.values)

    return StatisticSample(
        "full",
        (ValueSample("Condition values", values, len(values), len(values)),),
    )


# ----------------------------------------------------------------------
# Neuron subsets
# ----------------------------------------------------------------------


def _masked_bindings(data, state, ids):
    """Data/state views whose assignments contain only the given rows.

    Mirrors StatisticEngine.evaluate_assignment_table(): small identity
    objects are copied, spatial and temporal arrays stay shared.
    """
    assignments = data.assignments

    bound_assignments = copy(assignments)
    bound_assignments.ids = ids
    bound_assignments.stats = dict(assignments.stats)

    if assignments.union is not None:
        bound_assignments.union = copy(assignments.union)

    bound_state = SimpleNamespace(
        assignments=ids,
        data_version=getattr(state, "data_version", None),
    )

    name = data._current_assignments

    bound_data = copy(data)
    bound_data._current_assignments = name
    bound_data._assignments = {name: bound_assignments}
    bound_data.sessions = list(data.sessions)
    bound_data.state = bound_state

    return bound_data, bound_state


def _masked_values(engine, query, definition, ids, rows):
    masked = np.full_like(ids, -1)
    masked[rows] = ids[rows]

    data, state = _masked_bindings(engine.data, engine.state, masked)

    stat = engine._evaluate_uncached(query, definition, data=data, state=state)
    _compact(stat, definition)
    table = PickTable.from_stat(stat)

    remaining = tuple(
        f for f in (query.filters or ()) if f.target not in stat.applied_pair_filters
    )

    if remaining:
        table = table.filtered(remaining)

    if not table.n_rows:
        return np.empty(0)

    refs = table.refs_for_rows(np.arange(table.n_rows))
    neurons = refs.get("neuron", refs.get("neuron_i"))

    if neurons is None:
        raise ValueError("Sampled table has no neuron coordinate.")

    values = np.asarray(table.values, dtype=np.float64)
    keep = np.isin(np.asarray(neurons), rows) & np.isfinite(values)

    return values[keep]


def _sample_neurons(engine, query, definition, *, max_samples, budget, rng):
    started = perf_counter()
    ctx = current_task_context()

    ids = np.array(engine.state.assignments, dtype=np.int64, copy=True)
    candidates = np.flatnonzero(np.any(ids >= 0, axis=1))
    order = rng.permutation(candidates)

    parts, used, collected = [], 0, 0
    # Small first batch: per-neuron cost grows with occurrence² for
    # session-pair statistics.
    batch = min(len(order), FIRST_NEURON_BATCH)

    while used < len(order):
        if ctx is not None:
            ctx.check_cancelled()

        rows = np.sort(order[used : used + batch])
        used += len(rows)

        batch_started = perf_counter()
        values = _masked_values(engine, query, definition, ids, rows)
        seconds = perf_counter() - batch_started

        parts.append(values)
        collected += len(values)

        remaining = budget - (perf_counter() - started)

        if collected >= max_samples or remaining <= 0:
            break

        # Size the next batch to the remaining time and missing values.
        per_neuron = seconds / len(rows)
        values_per_neuron = max(len(values) / len(rows), 1e-3)
        by_time = int(0.8 * remaining / max(per_neuron, 1e-6))
        by_need = int(ceil(1.1 * (max_samples - collected) / values_per_neuron))
        batch = max(1, min(by_time, by_need))

    values = np.concatenate(parts) if parts else np.empty(0)

    return StatisticSample(
        "neurons",
        (
            ValueSample(
                "Condition values",
                values,
                sampled=used,
                available=len(candidates),
                unit="neurons",
            ),
        ),
    )


# ----------------------------------------------------------------------
# Same-neuron vs nearest-other-neuron pairs
# ----------------------------------------------------------------------


def _axis_domain(query, axis, size):
    if axis is None:
        return np.array([-1], dtype=np.int64)

    spec = query.reduction_dict().get(axis, ReductionSpec("keep"))

    if spec.method == "single":
        return np.array([int(spec.index)], dtype=np.int64)

    return np.arange(size, dtype=np.int64)


def _session_pairs(query, endpoints, n_sessions, available):
    (_, si_axis), (_, sj_axis) = endpoints

    left = _axis_domain(query, si_axis, n_sessions)
    right = _axis_domain(query, sj_axis, n_sessions)

    left = left[(left < 0) | available[np.maximum(left, 0)]]
    right = right[(right < 0) | available[np.maximum(right, 0)]]

    if si_axis is None or sj_axis is None:
        return [(int(a), int(b)) for a in left for b in right]

    if si_axis == sj_axis:
        return [(int(a), int(a)) for a in np.intersect1d(left, right)]

    relation = "all"

    for f in query.filters:
        if f.target == "session":
            relation = f.relation

    right_set = set(right.tolist())

    if relation == "same":
        return [(int(a), int(a)) for a in left if int(a) in right_set]

    if relation == "with previous":
        return [(int(a), int(a) - 1) for a in left if int(a) - 1 in right_set]

    if is_order_relation(relation):
        # Rows are filtered per neuron when the populations are drawn.
        direction = {
            "previous instance": -1,
            "all previous": -1,
            "next instance": 1,
            "all following": 1,
        }.get(relation, 0)

        return [
            (int(a), int(b))
            for a in left
            for b in right
            if a != b and (direction == 0 or np.sign(b - a) == direction)
        ]

    return [
        (int(a), int(b))
        for a in left
        for b in right
        if relation != "different" or a != b
    ]


def _pair_values(engine, query, definition, pairs):
    if not len(pairs):
        return np.empty(0)

    pairs = np.asarray(pairs, dtype=np.int64)

    if definition.key == "footprint_similarity":
        from .footprint_pairs import footprint_similarity_for_pairs

        parameters = dict(query.parameters)
        values = footprint_similarity_for_pairs(
            engine.data,
            engine.state,
            pairs,
            gamma=parameters.get("gamma", 0.1),
        )

        if definition.compact_values:
            values = SimpleNamespace(values=values)
            _compact(values, definition)
            values = values.values
    else:
        stat = engine._evaluate_uncached(query, definition, requested_pairs=pairs)
        _compact(stat, definition)
        values = stat.values

    return _finite(values)


def _sample_pairs(engine, query, definition, endpoints, *, max_samples, budget, rng):
    from .distance_statistics import geometry, union_positions

    started = perf_counter()
    ctx = current_task_context()

    data, state = engine.data, engine.state
    n, s = state.assignments.shape

    (ni_axis, si_axis), (nj_axis, sj_axis) = endpoints

    positions, available = geometry(data, state)
    union = union_positions(data, n) if None in (si_axis, sj_axis) else None

    def endpoint_positions(session_id):
        return union if session_id < 0 else positions[:, session_id]

    source_neurons = np.zeros(n, dtype=bool)
    source_neurons[_axis_domain(query, ni_axis, n)] = True
    target_neurons = np.zeros(n, dtype=bool)
    target_neurons[_axis_domain(query, nj_axis, n)] = True

    session_pairs = _session_pairs(query, endpoints, s, available)

    if not session_pairs:
        return StatisticSample(
            "pairs", note="No aligned sessions match the condition's session axes."
        )

    order = rng.permutation(len(session_pairs))
    limit = max(1, int(ceil(max_samples / MIN_PAIRS_PER_SESSION_PAIR)))
    session_pairs = [session_pairs[k] for k in order[:limit]]
    quota = max(MIN_PAIRS_PER_SESSION_PAIR, int(ceil(max_samples / len(session_pairs))))

    # Both populations are rows of the same statistic; only the neuron
    # relation differs from the condition.
    evaluation_query = replace(
        query,
        filters=tuple(f for f in query.filters if f.target != "neuron")
        + (PairFilter("neuron", "all"),),
    )

    relation = next(
        (f.relation for f in query.filters if f.target == "session"), "all"
    )
    ordered = is_order_relation(relation) and None not in (si_axis, sj_axis)
    detection = (
        DetectionReference(state.assignments)
        if ordered and relation in NEURON_RELATIONS
        else None
    )

    trees = {}

    def target_tree(session_id):
        if session_id not in trees:
            points = endpoint_positions(session_id)
            targets = np.flatnonzero(target_neurons & np.all(np.isfinite(points), axis=1))
            trees[session_id] = (
                (cKDTree(points[targets]), targets) if len(targets) >= 2 else None
            )
        return trees[session_id]

    same_parts, other_parts = [], []
    counts = {"same": [0, 0], "other": [0, 0]}  # drawn, candidates
    processed = 0

    for start in range(0, len(session_pairs), SESSION_PAIR_CHUNK):
        if ctx is not None:
            ctx.check_cancelled()

        same_pairs, other_pairs = [], []

        for a, b in session_pairs[start : start + SESSION_PAIR_CHUNK]:
            left = endpoint_positions(a)
            right = endpoint_positions(b)

            sources = np.flatnonzero(source_neurons & np.all(np.isfinite(left), axis=1))

            if not len(sources):
                continue

            # Same neuron, other footprint (identical endpoints carry no value).
            if a != b:
                same = sources[target_neurons[sources] & np.all(np.isfinite(right[sources]), axis=1)]

                if ordered:
                    same = same[relation_mask(relation, a, b, same, detection)]

                counts["same"][1] += len(same)

                if len(same) > quota:
                    same = rng.choice(same, quota, replace=False)

                counts["same"][0] += len(same)
                same_pairs.extend((k, k, a, b) for k in same.tolist())

            # Nearest footprint of another neuron in the target session.
            found = target_tree(b)

            if found is None:
                continue

            tree, targets = found
            counts["other"][1] += len(sources)

            if len(sources) > quota:
                sources = rng.choice(sources, quota, replace=False)

            _, nearest = tree.query(left[sources], k=2)
            first, second = targets[nearest[:, 0]], targets[nearest[:, 1]]
            partner = np.where(first == sources, second, first)

            if ordered:
                related = relation_mask(relation, a, b, partner, detection)
                sources, partner = sources[related], partner[related]

            counts["other"][0] += len(sources)
            other_pairs.extend(
                (k, j, a, b) for k, j in zip(sources.tolist(), partner.tolist())
            )

        same_parts.append(_pair_values(engine, evaluation_query, definition, same_pairs))
        other_parts.append(_pair_values(engine, evaluation_query, definition, other_pairs))
        processed = start + SESSION_PAIR_CHUNK

        collected = min(
            sum(map(len, same_parts)) if counts["same"][1] else max_samples,
            sum(map(len, other_parts)),
        )

        if collected >= max_samples or perf_counter() - started >= budget:
            break

    complete = processed >= len(session_pairs) and len(order) <= limit

    samples = []

    for label, key, parts in (
        ("Same neuron", "same", same_parts),
        ("Nearest other neuron", "other", other_parts),
    ):
        drawn, candidates = counts[key]

        if not candidates:
            continue

        samples.append(
            ValueSample(
                label,
                np.concatenate(parts) if parts else np.empty(0),
                sampled=drawn,
                available=candidates if complete else None,
                unit="pairs",
            )
        )

    return StatisticSample("pairs", tuple(samples))
