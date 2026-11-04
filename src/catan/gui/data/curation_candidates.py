"""Decision paths of a neuron's candidates through a curator preset.

Built by ``curation_check.check_neuron`` from its evaluation. A candidate
is what the preset's root group matches: the neuron (neuron level), each
of its footprints (footprint level), or, for source → target presets,
each of its footprints as target with its best source candidates
(including near misses). Every candidate lists its steps (the root's
conditions and subgroups) in order with values and verdicts; the first
failing step is where it drops out, later steps show what would follow.
"""

from dataclasses import dataclass, field, replace
from itertools import combinations

import numpy as np

from catan.gui.data.curation_check import ConditionCheck, GroupCheck
from catan.gui.data.curation_filter import (
    NeuronComponent,
    RelationshipMatch,
)
from catan.gui.data.statistics.queries import ReductionSpec

# Source candidates listed per target besides those passing all pair checks.
MAX_SOURCES = 5

# Largest source set searched exhaustively for the most separated subset.
MAX_EXHAUSTIVE_SOURCES = 10

COMPARISON_SYMBOLS = {"eq": "=", "ge": "≥", "le": "≤"}


@dataclass
class Step:
    label: str
    value: str
    passed: bool | None  # None: not evaluated / unknown
    detail: str = ""
    skipped: bool = False  # optional condition without its statistic


@dataclass
class Candidate:
    label: str
    passed: bool | None
    summary: str = ""
    steps: list = field(default_factory=list)
    children: list = field(default_factory=list)  # Candidate
    components: tuple = ()  # footprints to highlight
    detail: str = ""

    def first_failure(self):
        return next((step for step in self.steps if step.passed is False), None)


def footprint_label(component):
    if component.session_id is None:
        return f"neuron {component.neuron_id}"
    return f"N{component.neuron_id} S{component.session_id}"


def format_value(value):
    if value is None or not np.isfinite(value):
        return "–"
    return f"{value:.3g}"


# ----------------------------------------------------------------------
# Single values
# ----------------------------------------------------------------------


def evaluate_at(engine, query, components):
    """Values of ``query`` at one footprint or an ordered footprint pair
    (source / i side first); pair filters are dropped. Empty if the
    statistic has no matching axes."""
    definition = engine.registry.get(query.statistic_key)
    if definition is None:
        return np.empty(0)

    dims = set(definition.dims)
    fixed = {}

    if len(components) == 2 and "neuron_i" in dims and "neuron_j" in dims:
        first, second = components
        fixed.update(neuron_i=first.neuron_id, neuron_j=second.neuron_id)
        if "session_i" in dims and "session_j" in dims:
            fixed.update(session_i=first.session_id, session_j=second.session_id)
        elif "session" in dims:
            fixed["session"] = first.session_id
        if None in (first.session_id, second.session_id) and dims != {"neuron_i", "neuron_j"}:
            return np.empty(0)
    elif len(components) == 1 and "neuron" in dims:
        if components[0].session_id is None and "session" in dims:
            return np.empty(0)
        fixed["neuron"] = components[0].neuron_id
        if "session" in dims:
            fixed["session"] = components[0].session_id
    else:
        return np.empty(0)

    reductions = query.reduction_dict()
    reductions.update(
        {dim: ReductionSpec("single", index=int(index)) for dim, index in fixed.items()}
    )
    single = replace(query, reductions=tuple(sorted(reductions.items())), filters=())

    table = engine.evaluate_table(single, cache=False)
    if table is None:
        return np.empty(0)
    values = np.asarray(table.values, dtype=np.float64).ravel()
    return values[np.isfinite(values)]


def _same(key_component, wanted):
    """A row's footprint is ``wanted`` (a neuron: any of its footprints)."""
    if wanted.session_id is None:
        return key_component.neuron_id == wanted.neuron_id
    return key_component == wanted


def _pick(values, threshold):
    """One value for an endpoint covering several rows (a neuron
    endpoint): a passing one if any (as the curator), else the nearest."""
    if len(values) == 1:
        return values[0]
    values = np.asarray(values, dtype=np.float64)
    mask = threshold.mask(values)
    if mask is not None and np.any(mask):
        return float(values[np.flatnonzero(mask)[0]])
    distance = np.abs(values - threshold.value)
    if threshold.direction == "between" and threshold.upper_value is not None:
        distance = np.minimum(distance, np.abs(values - threshold.upper_value))
    return float(values[int(np.argmin(distance))])


def _value(engine, condition_check, components):
    """A condition's value at ``components`` (rows of the check first)."""
    value = condition_check.lookup.get(tuple(components))
    if value is not None:
        return value

    if any(c.session_id is None for c in components):
        values = [
            v
            for key, v in condition_check.lookup.items()
            if len(key) == len(components)
            and all(_same(k, c) for k, c in zip(key, components))
        ]
        if values:
            return _pick(values, condition_check.condition.threshold)

    try:
        values = evaluate_at(engine, condition_check.condition.query, components)
    except Exception:
        return None
    if values.size == 1:
        return float(values[0])
    if values.size > 1:
        return _pick(list(values), condition_check.condition.threshold)
    return None


def _passes(threshold, value):
    if value is None or not np.isfinite(value):
        return False
    mask = threshold.mask(np.asarray([value]))
    return bool(mask[0]) if mask is not None else None


def _condition_label(engine, condition):
    definition = engine.registry.get(condition.query.statistic_key)
    title = condition.query.statistic_key if definition is None else definition.title
    try:
        return condition.threshold.expression(title, precision=3)
    except ValueError:
        return title


def _condition_step(engine, condition_check, components):
    condition = condition_check.condition
    if condition_check.skipped:
        return Step(
            _condition_label(engine, condition),
            "skipped (statistic unavailable)",
            None,
            condition_check.note,
            skipped=True,
        )
    value = _value(engine, condition_check, components)
    return Step(
        _condition_label(engine, condition),
        format_value(value),
        _passes(condition.threshold, value) if value is not None else None,
        "" if value is not None else "No value for these footprints.",
    )


def _shortfall(step, threshold, value_text):
    """How far a failed value is from passing (relative)."""
    try:
        value = float(value_text)
    except ValueError:
        return 1.0
    bound = threshold.value
    if threshold.direction == "between" and threshold.upper_value is not None:
        bound = threshold.upper_value if value > threshold.upper_value else threshold.value
    return abs(value - bound) / max(abs(bound), 1e-9)


def _describe_group(group, length=48):
    text = "All of" if group.operator == "and" else "Any of"
    if group.comment:
        comment = group.comment.splitlines()[0]
        if len(comment) > length:
            comment = comment[: length - 1].rstrip() + "…"
        text += f" — {comment}"
    return text


def _count_text(spec, count):
    return f"found {count} (needs {COMPARISON_SYMBOLS[spec.comparison]} {spec.value})"


def _finish(candidate):
    """Summary and hover text from the steps."""
    failure = candidate.first_failure()
    if candidate.passed is None and candidate.steps:
        candidate.passed = (
            None
            if failure is None
            and any(s.passed is None and not s.skipped for s in candidate.steps)
            else failure is None
        )

    if not candidate.summary:
        if failure is not None:
            candidate.summary = f"✗ at {failure.label}: {failure.value}"
        elif candidate.passed:
            candidate.summary = "passes all checks"
        else:
            candidate.summary = "–" if not candidate.steps else "not evaluated"

    lines = []
    failed = False
    for step in candidate.steps:
        mark = "⊘" if step.skipped else {True: "✓", False: "✗", None: "–"}[step.passed]
        prefix = "   (downstream) " if failed else "   "
        lines.append(f"{prefix}{mark} {step.label}: {step.value}")
        failed = failed or step.passed is False
    candidate.detail = "\n".join([candidate.label, *lines])
    return candidate


# ----------------------------------------------------------------------
# Candidates by root match level
# ----------------------------------------------------------------------


def build_candidates(engine, evaluator, result, check, neuron_id):
    root_check = check.root
    if root_check is None:
        return []

    level = root_check.group.match_level

    if level == "relationship":
        return _relationship_candidates(engine, evaluator, result, check, neuron_id)
    if level == "footprint":
        return _footprint_candidates(engine, evaluator, result, check, neuron_id)
    return [_neuron_candidate(engine, result, check, neuron_id)]


def _neuron_footprints(engine, neuron_id):
    ids = np.asarray(engine.state.assignments)[int(neuron_id)]
    return [NeuronComponent(int(neuron_id), int(s)) for s in np.flatnonzero(ids >= 0)]


def _group_step(result, group_check, matched):
    group = group_check.group
    if group.id in result.skipped_nodes:
        return Step(
            _describe_group(group),
            "skipped (only unavailable optional conditions)",
            None,
            skipped=True,
        )
    group_result = result.group_results.get(group.id)
    passed = None if group_result is None else matched(group_result)
    value = group_check.count or {True: "matches", False: "no match", None: "–"}[passed]
    return Step(_describe_group(group), value, passed, group_check.note)


def _neuron_candidate(engine, result, check, neuron_id):
    root_check = check.root
    steps = []

    for child in root_check.children:
        if isinstance(child, ConditionCheck):
            verdict = root_check.condition_verdicts.get(child.condition.id)
            if child.skipped:
                steps.append(_condition_step(engine, child, ()))
                continue
            if child.n_rows == 1:
                value = format_value(float(child.values[0]))
            elif child.n_rows:
                value = f"{child.n_passing} of {child.n_rows} rows pass"
            else:
                value = "no values"
            steps.append(Step(_condition_label(engine, child.condition), value, verdict, child.note))
        else:
            value = child.count or {True: "matches", False: "no match", None: "–"}[child.passed]
            steps.append(Step(_describe_group(child.group), value, child.passed, child.note))

    return _finish(
        Candidate(
            f"Neuron {neuron_id}",
            check.passed,
            steps=steps,
            components=(NeuronComponent(int(neuron_id), None),),
        )
    )


def _footprint_candidates(engine, evaluator, result, check, neuron_id):
    root_check = check.root
    candidates = []

    for footprint in _neuron_footprints(engine, neuron_id):
        steps = []

        for child in root_check.children:
            if isinstance(child, ConditionCheck):
                condition_result = result.condition_results.get(child.condition.id)
                passed = None
                if condition_result is not None:
                    try:
                        level = evaluator._condition_result_for_level(
                            condition_result, "footprint"
                        )
                        passed = footprint in level.matches
                    except Exception:
                        passed = None
                values = [
                    value for key, value in child.lookup.items() if footprint in key
                ]
                value = (
                    format_value(values[0]) if len(values) == 1
                    else f"{len(values)} values" if values else "no values"
                )
                steps.append(
                    Step(_condition_label(engine, child.condition), value, passed, child.note)
                )
            else:
                steps.append(
                    _group_step(
                        result,
                        child,
                        lambda r, f=footprint: f in r.matches
                        or f.neuron_id in {
                            m for m in r.matches if isinstance(m, int)
                        },
                    )
                )

        passed = footprint in result.root_result.matches if result else None
        candidates.append(
            _finish(
                Candidate(
                    f"Footprint {footprint_label(footprint)}",
                    passed,
                    steps=steps,
                    components=(footprint,),
                )
            )
        )

    return _sorted(candidates)


def _sorted(candidates):
    return sorted(
        candidates,
        key=lambda c: ({True: 0, None: 1, False: 2}[c.passed], c.label),
    )


# ----------------------------------------------------------------------
# Source -> target presets
# ----------------------------------------------------------------------


def _relationship_candidates(engine, evaluator, result, check, neuron_id):
    root_check = check.root
    root = root_check.group

    pair_checks = [c for c in root_check.children if isinstance(c, ConditionCheck)]
    same_groups = [
        c for c in root_check.children
        if isinstance(c, GroupCheck) and c.group.apply_to == "same"
    ]
    source_groups = [
        c for c in root_check.children
        if isinstance(c, GroupCheck) and c.group.apply_to == "source"
    ]
    target_groups = [
        c for c in root_check.children
        if isinstance(c, GroupCheck) and c.group.apply_to == "target"
    ]
    between_groups = [
        c for c in root_check.children
        if isinstance(c, GroupCheck) and c.group.apply_to.startswith("between")
    ]

    root_result = result.group_results.get(root.id)
    counts = {} if root_result is None else root_result.counts
    source_kind, target_kind = _endpoint_kinds(evaluator, result, root_result, pair_checks)
    targets = []
    pair_sources = {}  # target -> sources passing the pair conditions

    target_list = (
        [NeuronComponent(int(neuron_id), None)]
        if target_kind == "neuron"
        else _neuron_footprints(engine, neuron_id)
    )

    def as_source(component):
        if source_kind == "neuron":
            return NeuronComponent(component.neuron_id, None)
        return component

    for target in target_list:
        # Sources with a value for any pair condition.
        sources = {
            as_source(key[0])
            for pair_check in pair_checks
            for key in pair_check.lookup
            if len(key) == 2 and _same(key[1], target)
        }
        sources.discard(target)
        if target_kind == "neuron":
            sources = {c for c in sources if c.neuron_id != target.neuron_id}

        source_candidates = [
            _source_candidate(
                engine, result, source, target, pair_checks, same_groups, source_groups
            )
            for source in sources
        ]

        # Passing all pair checks first, then the nearest misses.
        def rank(candidate):
            failed = [
                (step, check_.condition.threshold)
                for step, check_ in zip(candidate.steps, pair_checks)
                if step.passed is False
            ]
            return (
                len(failed),
                sum(_shortfall(step, threshold, step.value) for step, threshold in failed),
            )

        source_candidates.sort(key=rank)
        passing_pairs = [c for c in source_candidates if rank(c)[0] == 0]
        shown = passing_pairs + [
            c for c in source_candidates if rank(c)[0] > 0
        ][:MAX_SOURCES]

        qualifying = [c for c in shown if c.passed is not False]
        target_candidate = Candidate(
            f"Target {footprint_label(target)}"
            if target.session_id is not None
            else f"Neuron {target.neuron_id} (as target)",
            None,
        )
        target_candidate.components = (target,)

        for group_check in target_groups:
            target_candidate.steps.extend(
                _bound_steps(engine, result, group_check, target)
            )

        pair_sources[target] = [c.components[0] for c in passing_pairs]

        # Between checks only among the sources passing the pair conditions
        # and the source checks (the curator applies source subgroups first).
        compatible = qualifying
        between_sources = [c for c in passing_pairs if c in qualifying][:MAX_EXHAUSTIVE_SOURCES]
        support_groups = [
            g for g in between_groups
            if g.group.apply_to == "between_sources" and g.group.summary is None
        ]
        if support_groups and len(between_sources) > 1:
            between, compatible = _between_node(
                engine, support_groups, between_sources, qualifying
            )
            target_candidate.children.append(between)

        # The verdict is the curator's; the counts explain it.
        count = counts.get(target)
        spec = root.count
        matched = root_result is not None and any(
            _match_has_target(m, target) for m in root_result.matches
        )
        target_candidate.passed = matched if root_result is not None else None
        mark = {True: "✓ ", False: "✗ ", None: ""}[target_candidate.passed]
        if spec is not None:
            found = count if count is not None else len(compatible)
            target_candidate.summary = mark + _count_text(spec, found)
        else:
            target_candidate.summary = mark + (
                f"{len(qualifying)} qualifying source{'s' if len(qualifying) != 1 else ''}"
            )
            if qualifying and not matched:
                target_candidate.summary += " (no pair passes all groups)"

        vote = _vote_step(engine, result, root, target, compatible)
        if vote is not None:
            target_candidate.steps.append(vote)
            if target_candidate.passed is False and qualifying:
                target_candidate.summary += f" ({vote.value})"

        explanation = [
            f"{len(passing_pairs)} source footprint(s) pass the pair conditions, "
            f"{len(qualifying)} also the source checks"
            + (
                f", largest separated set: {len(compatible)} "
                f"({', '.join(footprint_label(c.components[0]) for c in compatible)})"
                if support_groups and qualifying
                else ""
            )
            + "."
        ]
        if count is not None:
            explanation.append(f"Curator count: {count}.")

        target_candidate.children[:0] = shown
        target_candidate.detail = "\n".join(
            [target_candidate.label, *explanation]
            + [step.label + ": " + step.value for step in target_candidate.steps]
        )
        targets.append(target_candidate)

    targets = sorted(
        targets,
        key=lambda c: (
            {True: 0, None: 1, False: 2}[c.passed],
            -sum(child.passed is not False for child in c.children),
            c.label,
        ),
    )

    summary_groups = [
        g for g in between_groups
        if g.group.summary is not None or g.group.apply_to != "between_sources"
    ]
    if summary_groups:
        partner_node = _partner_pairs_candidate(
            engine, result, summary_groups, pair_sources, neuron_id
        )
        if partner_node.passed is False:
            # Neuron-level checks: they decide for all targets.
            for target in targets:
                if target.passed is False and "(no pair passes all groups)" in target.summary:
                    target.summary = target.summary.replace(
                        "(no pair passes all groups)", "(fails the partner-pair checks)"
                    )
            targets.insert(0, partner_node)
        else:
            targets.append(partner_node)

    return targets


def _vote_step(engine, result, root, target, compatible):
    """Split / merge: how the other sessions see the region (see
    manipulation_votes). The curator's chosen pair if any, else the best
    compatible pair of a split target."""
    kind = None if root.manipulation is None else root.manipulation.kind
    if kind not in ("split", "merge") or target.session_id is None:
        return None

    label = f"Other sessions ({'separate' if kind == 'split' else 'joint'} > "
    label += f"{'joint' if kind == 'split' else 'separate'})"

    chosen = getattr(result, "manipulation_votes", {}) or {}
    for whole, (pair, votes) in chosen.items():
        # Split: the target is the whole; merge: one of the pieces.
        if (whole == target) if kind == "split" else (target in pair):
            parts = " + ".join(footprint_label(c) for c in pair)
            return Step(
                label,
                f"{votes.describe()} for {parts}",
                votes.supports(kind),
                detail=votes.votes,
            )

    if kind != "split":
        return None

    from catan.gui.data.manipulation_votes import SessionVoter

    sources = [c.components[0] for c in compatible if c.components]
    if len(sources) < 2:
        return None

    from catan.gui.data.shape_decisions import decisions_of

    voter = SessionVoter(
        engine.data.sessions,
        engine.state.assignments,
        decisions_of(engine.data.assignments).values(),
    )
    best = None
    for a, b in combinations(sources, 2):
        if a.neuron_id == b.neuron_id:
            continue
        votes = voter.votes(target, (a, b))
        rank = (votes.margin(kind), votes.separate + votes.joint)
        if best is None or rank > best[0]:
            best = (rank, (a, b), votes)

    if best is None:
        return Step(label, "no source pair from two different neurons", False)

    _, pair, votes = best
    parts = " + ".join(footprint_label(c) for c in pair)
    return Step(
        label,
        f"{votes.describe()} for {parts}",
        votes.supports(kind),
        detail=votes.votes,
    )


# Partner pairs evaluated per summary condition.
MAX_SUMMARY_PAIRS = 60

SUMMARIES = {
    "median": np.median,
    "mean": np.mean,
    "min": np.min,
    "max": np.max,
}


def _partner_pairs(binding, pair_sources):
    """Partner pairs of a target neuron for a "between" binding."""
    pairs = set()
    for target, sources in pair_sources.items():
        if binding == "between_sources":
            pairs.update(combinations(sorted(set(sources), key=_sort_key), 2))
        elif binding == "between_sources_targets":
            pairs.update((source, target) for source in sources)
    if binding == "between_targets":
        by_source = {}
        for target, sources in pair_sources.items():
            for source in sources:
                by_source.setdefault(source, set()).add(target)
        for targets in by_source.values():
            pairs.update(combinations(sorted(targets, key=_sort_key), 2))
    return sorted(pairs, key=lambda pair: tuple(map(_sort_key, pair)))


def _sort_key(component):
    return (component.neuron_id, -1 if component.session_id is None else component.session_id)


def _partner_pairs_candidate(engine, result, groups, pair_sources, neuron_id):
    """Checks over pairs of partners (between sources / targets): the
    combined value (summary) or the share of passing pairs."""
    node = Candidate("Partner-pair checks", None)
    verdicts = []

    for group_check in groups:
        group = group_check.group
        pairs = _partner_pairs(group.apply_to, pair_sources)[:MAX_SUMMARY_PAIRS]
        binding = group.apply_to.replace("_", " ")

        group_result = result.group_results.get(group.id)
        verdict = None
        if group_result is not None:
            verdict = any(
                neuron_id in {m.target.neuron_id}
                for m in group_result.matches
                if isinstance(m, RelationshipMatch)
            )
        verdicts.append(verdict)

        for child in group_check.children:
            if not isinstance(child, ConditionCheck):
                continue
            threshold = child.condition.threshold
            values = []
            pair_nodes = []
            for first, second in pairs:
                value = _value(engine, child, (first, second))
                if value is None:
                    value = _value(engine, child, (second, first))
                if value is None:
                    continue
                values.append(value)
                pair_nodes.append(
                    Candidate(
                        f"{footprint_label(first)} ↔ {footprint_label(second)}",
                        _passes(threshold, value),
                        summary=format_value(value),
                        components=(first, second),
                    )
                )

            label = _condition_label(engine, child.condition)
            if group.summary is not None:
                combined = float(SUMMARIES[group.summary](values)) if values else None
                text = (
                    f"{group.summary} {format_value(combined)} over {len(values)} pairs"
                    if values else "no pairs"
                )
                passed = _passes(threshold, combined) if combined is not None else None
            else:
                passing = sum(_passes(threshold, v) for v in values)
                text = f"{passing} of {len(values)} pairs pass"
                passed = passing == len(values) if values else None

            step = Step(f"{binding}: {label}", text, passed)
            node.steps.append(step)
            child_node = Candidate(
                f"{binding}: {label}",
                passed,
                summary=text,
                children=pair_nodes,
            )
            node.children.append(_finish(child_node))

    # The curator's verdict where it has one, else the computed one.
    node.passed = None if any(v is None for v in verdicts) else all(verdicts)
    return _finish(node)


def _match_has_target(match, target):
    """A root match (pair, or footprint / neuron after a count) of ``target``."""
    if isinstance(match, RelationshipMatch):
        return _same(match.target, target)
    if isinstance(match, NeuronComponent):
        return _same(match, target)
    return isinstance(match, (int, np.integer)) and int(match) == target.neuron_id


def _endpoint_kinds(evaluator, result, root_result, pair_checks):
    kinds = None if root_result is None else root_result.endpoint_kinds
    for pair_check in pair_checks:
        if kinds is not None:
            break
        condition_result = result.condition_results.get(pair_check.condition.id)
        if condition_result is None:
            continue
        try:
            kinds = evaluator._condition_result_for_level(
                condition_result, "relationship"
            ).endpoint_kinds
        except Exception:
            kinds = None
    return kinds or ("footprint", "footprint")


def _source_candidate(engine, result, source, target, pair_checks, same_groups, source_groups):
    steps = [_condition_step(engine, pair_check, (source, target)) for pair_check in pair_checks]

    def pair_matched(group_result):
        return any(
            isinstance(m, RelationshipMatch) and _same(m.source, source) and _same(m.target, target)
            for m in group_result.matches
        )

    for group_check in same_groups:
        steps.append(_group_step(result, group_check, pair_matched))

    for group_check in source_groups:
        steps.extend(_bound_steps(engine, result, group_check, source))

    return _finish(
        Candidate(
            f"{footprint_label(source)} → {footprint_label(target)}",
            None,
            steps=steps,
            components=(source, target),
        )
    )


def _bound_steps(engine, result, group_check, footprint):
    """Steps of a source / target subgroup for one footprint."""
    steps = []
    who = footprint_label(footprint)

    for child in group_check.children:
        if isinstance(child, ConditionCheck):
            step = _condition_step(engine, child, (footprint,))
            step.label = f"{who}: {step.label}"
            steps.append(step)
            continue

        group = child.group
        group_result = result.group_results.get(group.id)
        count = None
        if group_result is not None and group.count is not None:
            count = group_result.counts.get(footprint)
            if count is None:
                count = group_result.counts.get(footprint.neuron_id)

        if count is not None:
            steps.append(
                Step(
                    f"{who}: {_describe_group(group)}",
                    _count_text(group.count, count),
                    group.count.accepts(count),
                )
            )
        elif group_result is not None and footprint in group_result.matches:
            steps.append(Step(f"{who}: {_describe_group(group)}", "matches", True))
        else:
            steps.append(
                Step(
                    f"{who}: {_describe_group(group)}",
                    "–",
                    None,
                    "Not evaluated by the curator for this footprint.",
                )
            )

    return steps


def _between_node(engine, between_groups, candidates, qualifying):
    """Pairwise checks between the ``candidates``' sources (those passing
    the pair and source checks) and the largest subset passing all."""
    sources = [c.components[0] for c in candidates]
    node = Candidate("Between the sources", None)
    ok = {}

    for first, second in combinations(range(len(sources)), 2):
        a, b = sources[first], sources[second]
        steps = []
        for group_check in between_groups:
            for child in group_check.children:
                if not isinstance(child, ConditionCheck):
                    continue
                value = child.lookup.get((a, b), child.lookup.get((b, a)))
                if value is None:
                    try:
                        values = evaluate_at(engine, child.condition.query, (a, b))
                        value = float(values[0]) if values.size == 1 else None
                    except Exception:
                        value = None
                steps.append(
                    Step(
                        _condition_label(engine, child.condition),
                        format_value(value),
                        _passes(child.condition.threshold, value) if value is not None else None,
                    )
                )

        pair = _finish(
            Candidate(f"{footprint_label(a)} ↔ {footprint_label(b)}", None, steps=steps, components=(a, b))
        )
        ok[(first, second)] = pair.passed is not False
        node.children.append(pair)

    eligible = list(range(len(candidates)))
    index_of = {i: k for k, i in enumerate(eligible)}
    sub_ok = {
        (index_of[a], index_of[b]): passed
        for (a, b), passed in ok.items()
        if a in index_of and b in index_of
    }
    best = [eligible[k] for k in _largest_compatible(len(eligible), sub_ok)]
    compatible = [candidates[i] for i in best]
    node.passed = all(ok.values())
    node.summary = (
        "all pairs separated"
        if node.passed
        else f"{sum(not v for v in ok.values())} of {len(ok)} pairs not separated; "
        f"usable: {', '.join(footprint_label(sources[i]) for i in best) or '–'}"
    )
    node.components = tuple(sources)
    node.detail = "\n".join(
        ["Between the sources: only sources passing the pair and source "
         "checks (as the curator); the largest set of sources whose pairs "
         "all pass counts"]
        + [f"   {'✓' if c.passed else '✗'} {c.label}: " + ", ".join(
            f"{s.label} {s.value}" for s in c.steps) for c in node.children]
    )
    return node, compatible


def _largest_compatible(n, ok):
    if n == 0:
        return []
    if n > MAX_EXHAUSTIVE_SOURCES:
        chosen = []
        for index in range(n):
            if all(ok[(min(i, index), max(i, index))] for i in chosen):
                chosen.append(index)
        return chosen
    for size in range(n, 0, -1):
        for subset in combinations(range(n), size):
            if all(ok[pair] for pair in combinations(subset, 2)):
                return list(subset)
    return []
