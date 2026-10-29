"""Why does a neuron (not) match a curator preset?

``check_neuron`` evaluates the preset as the curator does (verdicts of the
neuron for every group) and, for every condition, the statistic values of
the rows involving the neuron (or the partners a bound subgroup checks)
against the resolved threshold.

AND relationship groups evaluate progressively: a later child only sees
the pairs that passed the earlier ones, so a neuron failing an earlier
child is "not reached" by the later ones.
"""

from copy import deepcopy
from dataclasses import dataclass, field, replace

import numpy as np

from catan.gui.background_tasks.runtime import progress_scope, progress_step
from catan.gui.data.curation_filter import (
    CurationFilterError,
    CurationFilterEvaluator,
    NeuronComponent,
    RelationshipMatch,
    _component_refs_for_row,
    is_filter_group,
    match_neurons,
)
from catan.gui.data.statistics.queries import ReductionSpec

NEURON_DIMS = ("neuron", "neuron_i", "neuron_j")

# Rows kept per condition (sorted by value; summaries use all rows).
MAX_ROWS = 500

# Partner neurons evaluated one by one for bound subgroups.
MAX_PARTNERS = 12

# Rows checked for duplicate orientations of one pair.
DEDUPLICATE_ROWS = 100_000


@dataclass(frozen=True)
class Scope:
    """Whose rows a condition lists: the checked neuron, or the partners a
    bound subgroup checks (their footprints; ``within``: only pairs among
    them, for "between" bindings)."""

    neurons: tuple
    footprints: frozenset | None = None
    within: bool = False
    bound: bool = False
    # The curator evaluated the bound subgroup for these partners (the
    # parent matched); otherwise they come from the parent's candidates.
    reached: bool = False


@dataclass
class ConditionCheck:
    condition: object  # CurationFilterCondition, thresholds resolved
    columns: dict = field(default_factory=dict)  # dim -> values per row
    values: np.ndarray = field(default_factory=lambda: np.empty(0))
    passes: np.ndarray = field(default_factory=lambda: np.empty(0, dtype=bool))
    n_rows: int = 0  # before truncation to MAX_ROWS
    n_passing: int = 0
    summary: tuple | None = None  # (min, median, max) of the finite values
    component_axes: tuple | None = None
    neurons: tuple = ()  # neurons the rows were restricted to
    note: str = ""
    error: str = ""
    # All rows (not only the listed ones): ordered footprints of the row
    # (source first for pairs) -> value.
    lookup: dict = field(default_factory=dict)

    def row_components(self, index):
        return _component_refs_for_row(self.columns, index, self.component_axes)


@dataclass
class GroupCheck:
    group: object  # CurationFilterGroup
    # True / False for the neuron; None: not evaluated for it.
    passed: bool | None
    note: str = ""
    count: str = ""  # the neuron's count for groups with a count condition
    children: list = field(default_factory=list)  # GroupCheck | ConditionCheck
    # Verdicts of conditions as children of this group (by condition id).
    condition_verdicts: dict = field(default_factory=dict)


@dataclass
class NeuronCheck:
    neuron_id: int
    component: NeuronComponent | None
    passed: bool | None
    component_passed: bool | None = None  # footprint-level root only
    root: GroupCheck | None = None
    error: str = ""
    error_node: str | None = None
    # Decision paths per candidate (curation_candidates.Candidate).
    candidates: list = field(default_factory=list)


def check_neuron(engine, root, neuron_id, component=None):
    neuron_id = int(neuron_id)
    evaluator = CurationFilterEvaluator(engine)

    # Resolved once here, so the shown thresholds are the evaluated ones
    # (resolving again in evaluate() leaves them unchanged).
    root = deepcopy(root)
    check = NeuronCheck(neuron_id, component, None)

    try:
        with progress_scope(0, 5, "Resolving thresholds"):
            evaluator.resolve_thresholds(root)
        with progress_scope(5, 60):
            result = evaluator.evaluate(root)
    except CurationFilterError as exc:
        check.error = str(exc)
        check.error_node = exc.node_id
        result = None
    except KeyError as exc:
        check.error = f"Statistic unavailable: {exc.args[0]}"
        result = None
    except Exception as exc:
        check.error = f"{type(exc).__name__}: {exc}"
        result = None

    if result is not None:
        check.passed = neuron_id in result.neurons
        if component is not None and root.match_level == "footprint":
            check.component_passed = component in result.root_result.matches

    conditions = []
    check.root = _group_check(
        root, result, neuron_id, None, conditions, evaluator, Scope((neuron_id,))
    )

    for index, (condition_check, scope, group_check) in enumerate(conditions):
        with progress_step(index, len(conditions), start=60, stop=90):
            _fill_condition(engine, condition_check, scope)
        # Partners: the curator's verdict concerns the checked neuron, the
        # rows the partners.
        if scope.bound and not condition_check.error:
            group_check.condition_verdicts[condition_check.condition.id] = (
                condition_check.n_passing > 0 if condition_check.n_rows else None
            )

    if result is not None:
        from catan.gui.data.curation_candidates import build_candidates

        with progress_scope(90, 100, "Collecting candidates"):
            try:
                check.candidates = build_candidates(
                    engine, evaluator, result, check, neuron_id
                )
            except Exception as exc:  # the details tab remains
                check.candidates = []
                check.error = check.error or f"Candidates: {type(exc).__name__}: {exc}"

    return check


def _group_passed(result, group_id, neuron_id):
    if result is None:
        return None

    group_result = result.group_results.get(group_id)
    if group_result is None:
        return None

    return any(neuron_id in match_neurons(match) for match in group_result.matches)


def _candidate_pairs(evaluator, result, group, neuron_id):
    """Source → target pairs of ``neuron_id`` (as target) passing the
    ordinary children of ``group``, before its count and bound subgroups."""
    pair_sets = []

    for child in group.children:
        if is_filter_group(child):
            if child.apply_to != "same":
                continue
            child_result = result.group_results.get(child.id)
            if child_result is None or child_result.match_level != "relationship":
                continue
            matches = child_result.matches
        else:
            condition_result = result.condition_results.get(child.id)
            if condition_result is None:
                return set()
            try:
                matches = evaluator._condition_result_for_level(
                    condition_result, "relationship"
                ).matches
            except Exception:
                continue

        pair_sets.append(
            {
                match
                for match in matches
                if isinstance(match, RelationshipMatch)
                and match.target.neuron_id == neuron_id
            }
        )

    return set.intersection(*pair_sets) if pair_sets else set()


def _partner_scope(evaluator, result, group, parent, neuron_id):
    """Scope and note of a bound subgroup (None: not bound)."""
    if group.apply_to == "same" or parent is None:
        return None, ""

    if result is None:
        return Scope((), bound=True), ""

    mapping = result.inspection_targets(neuron_id, "group", parent.id) or {}
    reached = bool(mapping)
    note = ""

    if not mapping:
        # The parent did not match: its candidate pairs.
        pairs = _candidate_pairs(evaluator, result, parent, neuron_id)
        for pair in pairs:
            mapping.setdefault(pair.target, set()).add(pair.source)
        if mapping:
            note = (
                "The parent group did not match this neuron: showing the "
                "partners of the pairs passing its conditions."
            )

    sources = {s for values in mapping.values() for s in values}
    targets = set(mapping)

    if group.apply_to in ("source", "between_sources"):
        endpoints = sources
    elif group.apply_to in ("target", "between_targets"):
        endpoints = targets
    else:
        endpoints = sources | targets

    neurons = tuple(sorted({int(c.neuron_id) for c in endpoints}))
    footprints = frozenset(c for c in endpoints if c.session_id is not None)

    if neurons:
        described = ", ".join(
            f"N{c.neuron_id} S{c.session_id}"
            for c in sorted(footprints, key=lambda c: (c.neuron_id, c.session_id))
        ) or ", ".join(f"neuron {n}" for n in neurons)
        note = "\n".join(filter(None, [f"Partners: {described}", note]))
    else:
        note = "No partners of this neuron reached this subgroup."

    return (
        Scope(
            neurons,
            footprints or None,
            within=group.apply_to.startswith("between"),
            bound=True,
            reached=reached,
        ),
        note,
    )


def _group_count(result, group, neurons, footprints=None):
    group_result = None if result is None else result.group_results.get(group.id)
    if group.count is None or group_result is None or not group_result.counts:
        return ""

    counts = [
        (subject, value)
        for subject, value in group_result.counts.items()
        if set(match_neurons(subject)) & set(neurons)
        and (
            footprints is None
            or not isinstance(subject, NeuronComponent)
            or subject in footprints
        )
    ]
    if not counts:
        return ""
    if len(counts) == 1 and len(neurons) == 1:
        return f"found {counts[0][1]}"

    def label(subject):
        session = getattr(subject, "session_id", None)
        prefix = f"N{subject.neuron_id} " if len(neurons) > 1 else ""
        return prefix + ("" if session is None else f"S{session}")

    return "found " + ", ".join(
        f"{value} ({label(subject)})"
        for subject, value in sorted(counts, key=lambda item: str(item[0]))
    )


def _group_check(group, result, neuron_id, parent, conditions, evaluator, scope):
    passed = _group_passed(result, group.id, neuron_id)
    partner_scope, note = _partner_scope(evaluator, result, group, parent, neuron_id)
    if partner_scope is not None:
        scope = partner_scope

    subjects = scope.neurons if scope.bound else (neuron_id,)
    if scope.bound:
        # The curator evaluates bound subgroups only for the parent's
        # matching pairs; their verdicts concern the partners.
        passed = (
            any(_group_passed(result, group.id, n) for n in subjects)
            if scope.reached and subjects
            else None
        )

    check = GroupCheck(group, passed, note, _group_count(result, group, subjects, scope.footprints))

    progressive = group.operator == "and" and group.match_level == "relationship"
    failed_before = False

    for child in group.children:
        if is_filter_group(child):
            child_check = _group_check(
                child, result, neuron_id, group, conditions, evaluator, scope
            )
            verdict = child_check.passed
        else:
            child_check = ConditionCheck(child)
            conditions.append((child_check, scope, check))
            verdict = None if scope.bound else _condition_passed(
                evaluator, result, child.id, group.match_level, neuron_id
            )

        if (
            progressive
            and failed_before
            and not verdict
            and getattr(child, "apply_to", "same") == "same"
        ):
            verdict = None
            if is_filter_group(child):
                child_check.passed = None
                child_check.note = (
                    child_check.note + "\n" if child_check.note else ""
                ) + "Not reached: an earlier condition already failed."

        if verdict is False and getattr(child, "apply_to", "same") == "same":
            failed_before = True

        check.children.append(child_check)
        if not is_filter_group(child):
            check.condition_verdicts[child.id] = verdict

    return check


def _condition_passed(evaluator, result, condition_id, match_level, neuron_id):
    """Whether the condition alone matches the neuron at the group's match
    level (the stored projection is already restricted to the group's
    surviving matches)."""
    if result is None:
        return None

    condition_result = result.condition_results.get(condition_id)
    if condition_result is None:
        return None

    try:
        level_result = evaluator._condition_result_for_level(condition_result, match_level)
    except Exception:
        return neuron_id in condition_result.neurons

    return any(neuron_id in match_neurons(match) for match in level_result.matches)


def neuron_queries(engine, query, neuron):
    """(fixed dim, query) restricting each kept neuron axis of ``query`` to
    ``neuron``; empty if no neuron axis is kept."""
    definition = engine.registry.get(query.statistic_key)
    if definition is None:
        return []

    reductions = query.reduction_dict()
    kept = [
        dim
        for dim in NEURON_DIMS
        if dim in definition.dims
        and reductions.get(dim, ReductionSpec("keep")).method == "keep"
    ]

    return [
        (
            dim,
            replace(
                query,
                reductions=tuple(
                    sorted({**reductions, dim: ReductionSpec("single", index=int(neuron))}.items())
                ),
            ),
        )
        for dim in kept
    ]


def _neuron_rows(table, rows, neuron):
    """The ``rows`` involving ``neuron`` (blockwise, without materializing
    the table's columns). Pair filters may collapse a fixed neuron axis
    (e.g. "same neuron"), so the fixed index alone does not restrict."""
    for dim in NEURON_DIMS:
        axis = table.stat.dimensions.get(dim)
        if axis is not None and axis.mode == "fixed" and axis.parameter == neuron:
            return rows

    dims = [dim for dim in NEURON_DIMS if dim in table.refs]
    if not dims:
        return rows
    involved = np.unique(
        np.concatenate([table.rows_matching_ref_values(dim, [neuron]) for dim in dims])
    )
    return rows[np.isin(rows, involved)]


def _rows_for_neurons(engine, query, neurons):
    """Columns, values, component axes and a note for the finite rows of
    ``query`` involving any of ``neurons`` (each row once).

    Kept neuron axes are fixed to each neuron in turn (as the statistics
    panel's selection series), so only the neuron's rows are materialized;
    the full statistic is never evaluated for them. Without a kept neuron
    axis the values are not per neuron and the (cached) statistic is used
    as it is.
    """
    parts = []
    component_axes = None
    note = ""

    for neuron in neurons:
        fixed = neuron_queries(engine, query, neuron)
        tables = (
            [engine.evaluate_table(single, cache=False) for _, single in fixed]
            if fixed
            else [engine.evaluate_table(query)]
        )
        if not fixed:
            note = "The values are not per neuron (its neuron axes are reduced)."

        for table in tables:
            if table is None:
                continue
            component_axes = table.stat.component_axes
            values = np.asarray(table.values, dtype=np.float64)
            rows = np.flatnonzero(np.isfinite(values))
            if fixed:
                rows = _neuron_rows(table, rows, neuron)
            refs = table.refs_for_rows(rows, include_fixed=True)
            parts.append(({k: np.asarray(v) for k, v in refs.items()}, values[rows]))

        if not fixed:
            break

    if not parts:
        return {}, np.empty(0), component_axes, note or "No statistic is available."

    names = sorted(set.intersection(*(set(columns) for columns, _ in parts)))
    columns = {name: np.concatenate([c[name] for c, _ in parts]) for name in names}
    values = np.concatenate([v for _, v in parts])

    # Both orientations (and partners sharing a pair) contain the same row.
    if names and len(parts) > 1:
        keys = np.stack([columns[name].astype(np.int64) for name in names], axis=1)
        _, first = np.unique(keys, axis=0, return_index=True)
        first = np.sort(first)
        columns = {name: column[first] for name, column in columns.items()}
        values = values[first]

    return columns, values, component_axes, note


def ordered_components(columns, index, component_axes=None):
    """Footprints of a row in axis order (source / i side first)."""
    if component_axes is not None:
        sides = tuple(
            NeuronComponent(int(columns[n][index]), int(columns[s][index]))
            for n, s in component_axes
            if n in columns and s in columns
        )
        if sides:
            return sides

    # Without a session column the side is a neuron (session None).
    sides = []
    for suffix in ("i", "j"):
        neuron = f"neuron_{suffix}"
        session = f"session_{suffix}" if f"session_{suffix}" in columns else "session"
        if neuron in columns:
            sides.append(
                NeuronComponent(
                    int(columns[neuron][index]),
                    int(columns[session][index]) if session in columns else None,
                )
            )
    if sides:
        return tuple(sides)

    if "neuron" in columns:
        sessions = [k for k in ("session_i", "session_j", "session") if k in columns]
        if not sessions:
            return (NeuronComponent(int(columns["neuron"][index]), None),)
        return tuple(
            NeuronComponent(int(columns["neuron"][index]), int(columns[k][index]))
            for k in sessions
        )

    return ()


def _scope_rows(columns, component_axes, scope):
    """Rows of ``columns`` within the partner footprints of ``scope`` (all
    of the row's footprints for "between", any of them otherwise)."""
    n_rows = len(next(iter(columns.values()))) if columns else 0
    keep = np.ones(n_rows, dtype=bool)

    if scope.footprints is not None:
        for index in range(n_rows):
            components = _component_refs_for_row(columns, index, component_axes)
            if not components:
                continue
            inside = [c in scope.footprints for c in components]
            keep[index] = all(inside) if scope.within else any(inside)
    elif scope.within:
        neurons = set(scope.neurons)
        for dim in NEURON_DIMS:
            if dim in columns:
                keep &= np.isin(columns[dim], list(neurons))

    return keep


def _fill_condition(engine, check, scope):
    neurons = tuple(scope.neurons)
    check.neurons = neurons

    if not neurons:
        check.note = "No rows: no partner reached this subgroup."
        return

    notes = []
    if len(neurons) > MAX_PARTNERS:
        notes.append(f"Showing the first {MAX_PARTNERS} of {len(neurons)} partners.")
        neurons = neurons[:MAX_PARTNERS]

    threshold = check.condition.threshold

    try:
        columns, values, component_axes, note = _rows_for_neurons(
            engine, check.condition.query, neurons
        )
        mask = threshold.mask(values)
    except Exception as exc:
        check.error = f"{type(exc).__name__}: {exc}"
        return

    if note:
        notes.append(note)

    if scope.bound and len(values):
        keep = _scope_rows(columns, component_axes, scope)
        columns = {name: column[keep] for name, column in columns.items()}
        values = values[keep]
        mask = None if mask is None else np.asarray(mask)[keep]

    # A symmetric pair appears in both orientations (same footprints and
    # value); directional statistics differ and stay.
    if 1 < len(values) <= DEDUPLICATE_ROWS and columns:
        seen, keep = set(), np.ones(len(values), dtype=bool)
        for index in range(len(values)):
            key = (
                frozenset(_component_refs_for_row(columns, index, component_axes)),
                float(values[index]),
            )
            if key[0] and key in seen:
                keep[index] = False
            seen.add(key)
        columns = {name: column[keep] for name, column in columns.items()}
        values = values[keep]
        mask = None if mask is None else np.asarray(mask)[keep]

    passes = np.zeros(values.shape, dtype=bool) if mask is None else np.asarray(mask)

    for index in range(len(values)):
        key = ordered_components(columns, index, component_axes)
        if key:
            check.lookup.setdefault(key, float(values[index]))

    check.n_rows = int(values.size)
    check.n_passing = int(passes.sum())
    check.component_axes = component_axes
    if values.size:
        check.summary = (float(values.min()), float(np.median(values)), float(values.max()))

    # Listed rows: passing ones first, then the failing ones nearest the
    # threshold.
    distance = np.abs(values - threshold.value)
    if threshold.direction == "between" and threshold.upper_value is not None:
        distance = np.minimum(distance, np.abs(values - threshold.upper_value))
    keep = np.lexsort((distance, ~passes))
    if values.size > MAX_ROWS:
        notes.append(
            f"Listing {MAX_ROWS} of {values.size} rows: passing ones first, "
            "then those nearest the threshold."
        )
        keep = keep[:MAX_ROWS]

    check.columns = {name: column[keep] for name, column in columns.items()}
    check.values = values[keep]
    check.passes = passes[keep]
    check.note = "\n".join(notes)
