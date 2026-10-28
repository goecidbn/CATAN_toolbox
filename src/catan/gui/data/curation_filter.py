from catan.gui.background_tasks.runtime import (
    current_task_context, progress_scope, progress_step, progress_iter, reports_progress,
)
from dataclasses import dataclass, field, replace, asdict
from typing import List, Literal, ClassVar
from uuid import uuid4

from catan.gui.structures.state import NeuronComponent
import numpy as np

from catan.gui.data.statistics import (
    PickTable,
    StatisticQuery,
)
from catan.gui.data.statistics.engine import StatisticEngine
from catan.gui.panels.helper.Threshold import ThresholdSpec
from catan.gui.data.threshold_references import (
    ThresholdReferenceError,
    reference_from_dict,
    resolve_reference,
)
from catan.gui.data.curation_actions import ManipulationSpec

FilterOperator = Literal["and", "or"]

# Subgroup bindings comparing partners of the parent's relationships.
BETWEEN_BINDINGS = ("between_sources", "between_targets", "between_sources_targets")
PAIR_SUMMARIES = ("median", "mean", "min", "max")
FilterMatchLevel = Literal[
    "neuron", "footprint", "reassignment", "neuron_pair", "relationship"
]


class CurationFilterError(RuntimeError):

    def __init__(
        self,
        message: str,
        *,
        node_id: str | None = None,
        node_type: str | None = None,
    ):
        super().__init__(message)

        self.node_id = node_id
        self.node_type = node_type

    def attach_node(
        self,
        node_id: str,
        node_type: str,
    ):
        """
        Attach the closest filter node if the error does not
        already have a more specific origin.
        """
        if self.node_id is None:
            self.node_id = node_id
            self.node_type = node_type


@dataclass(slots=True)
class CurationFilterCondition:
    node_type: ClassVar[Literal["condition"]] = "condition"

    query: StatisticQuery
    threshold: ThresholdSpec

    id: str = field(default_factory=lambda: uuid4().hex)


@dataclass(frozen=True, slots=True)
class CountSpec:
    target: Literal["neuron", "footprint"] = "neuron"
    comparison: Literal["eq", "ge", "le"] = "ge"
    value: int = 1

    unit: Literal["neuron", "footprint"] = "footprint"
    anchor: Literal["source", "target"] = "target"
    separate_partner_sessions: bool = False

    def __post_init__(self):

        if self.anchor not in ("source", "target"):
            raise ValueError("Invalid count anchor.")

        if type(self.separate_partner_sessions) is not bool:
            raise ValueError("Invalid session grouping.")

        if self.target not in ("neuron", "footprint"):
            raise ValueError("Invalid count target.")

        if self.comparison not in ("eq", "ge", "le"):
            raise ValueError("Invalid count comparison.")

        if type(self.value) is not int or self.value < 0:
            raise ValueError("The count must be a non-negative integer.")

        if self.unit not in ("neuron", "footprint"):
            raise ValueError("Invalid count unit.")

    def accepts(self, count):
        if self.comparison == "eq":
            return count == self.value

        if self.comparison == "ge":
            return count >= self.value

        return count <= self.value


@dataclass(slots=True)
class CurationFilterGroup:
    node_type: ClassVar[Literal["group"]] = "group"

    operator: FilterOperator = "and"
    match_level: FilterMatchLevel = "footprint"

    children: "list[CurationFilterCondition | CurationFilterGroup]" = field(
        default_factory=list
    )

    id: str = field(default_factory=lambda: uuid4().hex)

    count: CountSpec | None = None

    comment: str = ""
    apply_to: Literal[
        "same",
        "source",
        "target",
        "between_sources",
        "between_targets",
        "between_sources_targets",
    ] = "same"
    manipulation: ManipulationSpec | None = None

    # Between-partner subgroups: combine each condition's values over all
    # partner pairs of a target neuron (median / mean / min / max) and
    # threshold the result. None: largest mutually compatible partner set
    # (between sources / between targets only).
    summary: Literal["median", "mean", "min", "max"] | None = None

    # None means all statuses; an empty tuple means none.
    review_statuses: tuple[int, ...] | None = None


@dataclass(slots=True)
class CurationFilterConditionResult:
    condition: CurationFilterCondition
    table: PickTable
    matching_rows: np.ndarray

    neurons: set[int]
    components: set[NeuronComponent]

    rows_by_neuron: dict[int, np.ndarray]
    rows_by_component: dict[NeuronComponent, np.ndarray]

    projected_result: "CurationFilterGroupResult | None" = None

    def components_for_neuron(
        self,
        neuron_id: int,
    ) -> set[NeuronComponent]:

        rows = self.rows_by_neuron.get(int(neuron_id))

        if rows is None:
            return set()

        return components_for_table_rows(self.table, rows)

    def components_for_component(
        self,
        component: NeuronComponent,
    ) -> set[NeuronComponent]:

        rows = self.rows_by_component.get(component)

        if rows is None:
            return set()

        return components_for_table_rows(self.table, rows)


@dataclass(frozen=True, slots=True)
class ReassignmentMatch:
    source_neuron: int
    source_session: int
    destination_neuron: int


@dataclass(frozen=True, slots=True)
class NeuronPairMatch:
    first: int
    second: int


@dataclass(frozen=True, slots=True)
class RelationshipMatch:
    source: NeuronComponent
    target: NeuronComponent


def match_neurons(match):

    if isinstance(match, RelationshipMatch):
        return {match.target.neuron_id}

    if isinstance(match, ReassignmentMatch):
        return {match.destination_neuron}

    if isinstance(match, NeuronPairMatch):
        return {match.first, match.second}

    if isinstance(match, NeuronComponent):
        return {int(match.neuron_id)}

    return {int(match)}


FilterMatch = (
    int | NeuronComponent | ReassignmentMatch | NeuronPairMatch | RelationshipMatch
)


def pair_target_evidence(matches):
    result = {}

    for match in matches:
        result.setdefault(match.target, set()).add(match.source)

    return result


def restrict_targets(mapping, matches, level):
    if mapping is None:
        return None

    if level == "relationship":
        return pair_target_evidence(matches)

    if level == "footprint":
        return {target: set(mapping.get(target, ())) for target in matches}

    if level == "neuron":
        result = {
            target: set(sources)
            for target, sources in mapping.items()
            if target.neuron_id in matches
        }

        # A matching neuron remains a valid target even when it has
        # no supporting sources, e.g. "exactly zero confusing footprints".
        represented_neurons = {target.neuron_id for target in result}

        for neuron_id in set(matches) - represented_neurons:
            result[NeuronComponent(int(neuron_id), None)] = set()

        return result

    return None


def combined_targets(children, matches, level):
    mappings = [
        result.target_evidence
        for result in children
        if result.target_evidence is not None
    ]

    if not mappings:
        return None

    merged = {}

    for mapping in mappings:
        for target, sources in mapping.items():
            merged.setdefault(target, set()).update(sources)

    return restrict_targets(merged, matches, level)


@dataclass(slots=True)
class CurationFilterGroupResult:
    group_id: str
    match_level: FilterMatchLevel

    matches: set[FilterMatch]

    # Evidence supporting each surviving identity.
    evidence_by_match: dict[
        FilterMatch,
        set[NeuronComponent],
    ]

    neuron_domain: set[int] = field(default_factory=set)
    footprint_domain: set[NeuronComponent] = field(default_factory=set)

    counts: dict[FilterMatch, int] = field(default_factory=dict)
    target_evidence: dict[NeuronComponent, set[NeuronComponent]] | None = None
    endpoint_kinds: tuple[str, str] | None = None

    def components_for_neuron(self, neuron_id):
        evidence = set()

        for match in self.matches:
            if int(neuron_id) in match_neurons(match):
                evidence.update(self.evidence_by_match.get(match, set()))

        return evidence


@dataclass(slots=True)
class CurationFilterResult:
    neurons: set[int]

    condition_results: dict[str, CurationFilterConditionResult]

    group_results: dict[str, CurationFilterGroupResult]

    root_result: CurationFilterGroupResult

    bound_inspections: dict = field(default_factory=dict)

    # Threshold recipes resolved for this evaluation:
    # {condition id: (value, upper_value)}.
    resolved_thresholds: dict = field(default_factory=dict)

    def evidence_for_neuron(
        self,
        neuron_id: int,
    ) -> dict[str, PickTable]:

        evidence = {}

        for condition_id, result in self.condition_results.items():

            rows = result.rows_by_neuron.get(int(neuron_id))

            if rows is None or rows.size == 0:
                continue

            evidence[condition_id] = result.table.subset_rows(rows)

        return evidence

    def components_for_condition(
        self,
        neuron_id: int,
        condition_id: str,
    ) -> set[NeuronComponent]:

        result = self.condition_results.get(condition_id)

        if result is None:
            return set()

        if result.projected_result is None:
            return set()

        return result.projected_result.components_for_neuron(
            neuron_id
        ) & self.root_result.components_for_neuron(neuron_id)

    def components_for_neuron(
        self,
        neuron_id: int,
    ) -> set[NeuronComponent]:

        return self.root_result.components_for_neuron(neuron_id)

    def components_for_group(
        self,
        neuron_id: int,
        group_id: str,
    ) -> set[NeuronComponent]:

        result = self.group_results.get(group_id)

        if result is None:
            return set()

        return result.components_for_neuron(neuron_id)

    def inspection_targets(self, neuron_id, source_type, source_id):
        if source_type == "group":
            node = self.group_results.get(source_id)
        else:
            condition = self.condition_results.get(source_id)
            node = None if condition is None else condition.projected_result

        mapping = self.bound_inspections.get(source_id)

        if mapping is None:
            if node is None or node.target_evidence is None:
                return None
            mapping = node.target_evidence

        root = self.root_result.target_evidence

        return {
            target: (
                set(sources) if root is None else set(sources) & root.get(target, set())
            )
            for target, sources in mapping.items()
            if target.neuron_id == neuron_id
            and (root is None or target in root)
            and neuron_id in self.neurons
        }


class CurationFilterEvaluator:

    def __init__(
        self,
        engine: StatisticEngine,
    ):
        self.engine = engine

    @reports_progress("Evaluating curator preset")
    def evaluate(
        self,
        root: CurationFilterGroup,
    ) -> CurationFilterResult:

        condition_results = {}
        group_results = {}

        self._bound_inspections = {}

        if root.apply_to != "same":
            raise CurationFilterError("The root group has no parent source or target.")

        if invalid_groups := empty_filter_group_ids(root):
            raise CurationFilterError(
                "The curation filter contains "
                f"{len(invalid_groups)} empty group"
                f"{'s' if len(invalid_groups) != 1 else ''}."
            )

        self._review_root = root
        self._review_allowed = None

        if root.review_statuses is not None:
            statuses = np.asarray(self.engine.data.assignments.review_status)

            if statuses.shape != (self.engine.state.assignments.shape[0],):
                raise CurationFilterError(
                    "Review status does not match the assignment table."
                )

            self._review_allowed = set(
                np.flatnonzero(np.isin(statuses, root.review_statuses))
            )

        with progress_scope(0, 2, "Resolving thresholds"):
            resolved = self.resolve_thresholds(root)
        with progress_scope(2, 3, "Preparing curator domains"):
            self._prepare_count_domains(root)
        with progress_scope(3, 97):
            root_result = self._evaluate_group(root, condition_results, group_results)

        neurons = set()

        for match in progress_iter(root_result.matches, start=97, stop=99):
            neurons.update(match_neurons(match))

        return CurationFilterResult(
            neurons=neurons,
            condition_results=condition_results,
            group_results=group_results,
            root_result=root_result,
            bound_inspections=self._bound_inspections,
            resolved_thresholds=resolved,
        )

    def resolve_thresholds(self, root):
        """Resolve the threshold recipes (threshold_references) of all
        conditions in ``root``: their values are set in place.

        Returns {condition id: (value, upper_value)} of the conditions
        with recipes.
        """
        resolved = {}

        def visit(node):
            if is_filter_group(node):
                for child in node.children:
                    visit(child)
                return

            threshold = node.threshold

            if threshold.reference is None and (
                threshold.direction != "between" or threshold.upper_reference is None
            ):
                return

            try:
                if threshold.reference is not None:
                    threshold.value = resolve_reference(
                        threshold.reference, self.engine, node.query
                    )
                if threshold.direction == "between" and threshold.upper_reference is not None:
                    threshold.upper_value = resolve_reference(
                        threshold.upper_reference, self.engine, node.query
                    )
            except ThresholdReferenceError as exc:
                raise CurationFilterError(
                    str(exc), node_id=node.id, node_type="condition"
                ) from exc

            if not np.isfinite(threshold.value) or (
                threshold.direction == "between" and not np.isfinite(threshold.upper_value)
            ):
                raise CurationFilterError(
                    "The threshold recipe gave no finite value.",
                    node_id=node.id,
                    node_type="condition",
                )

            resolved[node.id] = (threshold.value, threshold.upper_value)

        visit(root)
        return resolved

    @reports_progress(None)
    def _evaluate_condition(
        self,
        condition: CurationFilterCondition,
        *,
        requested_pairs=None,
    ) -> CurationFilterConditionResult:
        if not condition.threshold.active:
            raise CurationFilterError(
                "Curation filter condition has no active threshold."
            )

        try:
            condition.threshold.validate()
        except ValueError as exc:
            raise CurationFilterError(str(exc)) from exc

        with progress_scope(0, 85, f"Statistic: {condition.query.statistic_key}"):
            table = self.engine.evaluate_matching_table(
                condition.query,
                condition.threshold.mask,
                requested_pairs=requested_pairs,
            )

        if table is None:
            raise CurationFilterError("No statistic is available for this condition.")

        # The returned table already contains only passing rows.
        matching_rows = np.arange(
            table.n_rows,
            dtype=np.intp,
        )
        with progress_scope(85, 100, "Projecting matching rows"):
            (
                neurons,
                components,
                rows_by_neuron,
                rows_by_component,
            ) = self._project_rows(table, matching_rows)

        return CurationFilterConditionResult(
            condition=condition,
            table=table,
            matching_rows=matching_rows,
            neurons=neurons,
            components=components,
            rows_by_neuron=rows_by_neuron,
            rows_by_component=rows_by_component,
        )

    @staticmethod
    def _endpoint_matches(result, component):
        if result.match_level == "footprint":
            return component in result.matches

        if result.match_level == "neuron":
            return component.neuron_id in result.matches

        if result.match_level == "relationship":
            return component in (result.target_evidence or {})

        return False

    def _apply_endpoint_group(self, parent_result, child_result, binding):
        if child_result.match_level not in ("footprint", "neuron"):
            raise CurationFilterError(
                "A source/target subgroup must return footprints or neurons. "
                "For a pairwise repeatability check, count per target footprint."
            )

        kind = parent_result.endpoint_kinds[0 if binding == "source" else 1]

        if kind == "neuron" and child_result.match_level == "footprint":
            raise CurationFilterError(
                "A neuron endpoint needs a neuron-level subgroup. "
                "Reduce or count its footprint requirements explicitly."
            )
        matches = {
            pair
            for pair in parent_result.matches
            if self._endpoint_matches(
                child_result,
                getattr(pair, binding),
            )
        }

        footprint_domain = set(parent_result.footprint_domain)
        neuron_domain = set(parent_result.neuron_domain)

        # A target requirement restricts eligible targets, including when
        # counting zero supports. A source requirement only filters supports.
        if binding == "target":
            footprint_domain = {
                component
                for component in footprint_domain
                if self._endpoint_matches(child_result, component)
            }
            if kind == "footprint":
                neuron_domain &= {component.neuron_id for component in footprint_domain}
            else:
                neuron_domain = {
                    neuron
                    for neuron in neuron_domain
                    if self._endpoint_matches(
                        child_result,
                        NeuronComponent(neuron, None),
                    )
                }

        return replace(
            parent_result,
            matches=matches,
            evidence_by_match={
                pair: parent_result.evidence_by_match[pair] for pair in matches
            },
            target_evidence=pair_target_evidence(matches),
            footprint_domain=footprint_domain,
            neuron_domain=neuron_domain,
        )

    def _record_bound_inspection(
        self,
        group,
        binding,
        pairs,
        condition_results,
        group_results,
    ):
        def visit(node):
            if is_filter_group(node):
                native = group_results[node.id]
            else:
                native = condition_results[node.id].projected_result

            previous = self._bound_inspections.get(node.id)
            selected = set()

            for pair in pairs:
                endpoint = getattr(pair, binding)

                if previous is not None:
                    accepted = endpoint in previous
                else:
                    accepted = native is not None and self._endpoint_matches(
                        native, endpoint
                    )

                if accepted:
                    selected.add(pair)

            self._bound_inspections[node.id] = pair_target_evidence(selected)

            if is_filter_group(node):
                for child in node.children:
                    visit(child)

        visit(group)

    @reports_progress('Evaluating partner requirements')
    def _apply_support_groups(
        self,
        group,
        result,
        support_groups,
        condition_results,
        group_results,
    ):
        spec = group.count

        if spec is None:
            raise CurationFilterError(
                "Between-partner comparisons require counting on the parent."
            )

        expected = "between_targets" if spec.anchor == "source" else "between_sources"

        if any(child.apply_to != expected for child in support_groups):
            raise CurationFilterError(
                f"Count anchored to {spec.anchor}: use Apply to: "
                f"{expected.replace('_', ' ')}."
            )

        anchor_kind = result.endpoint_kinds[0 if spec.anchor == "source" else 1]

        if spec.target != anchor_kind:
            raise CurationFilterError(
                "Between-partner comparisons require counting "
                "per actual anchor type."
            )

        with progress_scope(0, 5):
            buckets = self._relationship_buckets(result, spec)

        partner_sets = [
            {pair.target if spec.anchor == "source" else pair.source for pair in pairs}
            for pairs in buckets.values()
        ]

        from catan.gui.background_tasks.runtime import current_task_context

        ctx = current_task_context()

        # Only compare supports belonging to a common outer target.
        candidates = set()

        for sources in progress_iter(partner_sets, start=5, stop=15, every=1):
            if ctx is not None:
                ctx.check_cancelled()

            for a in sources:
                for b in sources:
                    if a == b:
                        continue

                    if group.count.unit == "neuron" and a.neuron_id == b.neuron_id:
                        continue

                    candidates.add(RelationshipMatch(a, b))

        compatible = set(candidates)

        for progress_index, (child) in enumerate(support_groups):
            with progress_step(
                progress_index, len(support_groups), start=15, stop=65,
            ):
                if (
                    child.operator != "and"
                    or child.match_level != "relationship"
                    or child.count is not None
                    or any(not is_filter_condition(c) for c in child.children)
                ):
                    raise CurationFilterError(
                        "Use an AND source-target group with conditions "
                        "and no count for comparisons between partners.",
                        node_id=child.id,
                        node_type="group",
                    )

                for condition in child.children:
                    if not self.engine.supports_requested_pairs(condition.query):
                        raise CurationFilterError(
                            "This between-partner condition needs a statistic "
                            "supporting requested footprint pairs, with all "
                            "identities kept or fixed.",
                            node_id=condition.id,
                            node_type="condition",
                        )

                child_result = self._evaluate_group(
                    child,
                    condition_results,
                    group_results,
                    initial_pairs=compatible,
                    initial_kinds=(
                        result.endpoint_kinds[1 if spec.anchor == "source" else 0],
                    )
                    * 2,
                )

                compatible.intersection_update(child_result.matches)

        selected_pairs = set()

        for pairs, sources in progress_iter(zip(buckets.values(), partner_sets), total=len(partner_sets), start=65, stop=95, every=1):
            if ctx is not None:
                ctx.check_cancelled()

            vertices = sorted(
                sources,
                key=lambda c: (
                    c.neuron_id,
                    -1 if c.session_id is None else c.session_id,
                ),
            )

            adjacent = {index: set() for index in range(len(vertices))}

            for i, a in enumerate(vertices):
                for j in range(i + 1, len(vertices)):
                    b = vertices[j]

                    # Compatibility must hold in both directions.
                    if (
                        RelationshipMatch(a, b) in compatible
                        and RelationshipMatch(b, a) in compatible
                    ):
                        adjacent[i].add(j)
                        adjacent[j].add(i)

            # Find one largest mutually compatible set.
            # A-B and B-C alone must not qualify the set A-B-C.
            best = []

            def expand(chosen, available):
                nonlocal best

                if ctx is not None:
                    ctx.check_cancelled()

                if len(chosen) > len(best):
                    best = list(chosen)

                while available:
                    if len(chosen) + len(available) <= len(best):
                        return

                    vertex = available.pop()

                    expand(
                        chosen + [vertex],
                        [other for other in available if other in adjacent[vertex]],
                    )

            expand([], list(range(len(vertices))))

            chosen = {vertices[index] for index in best}

            selected_pairs.update(
                pair
                for pair in pairs
                if (pair.target if spec.anchor == "source" else pair.source) in chosen
            )

        return replace(
            result,
            matches=selected_pairs,
            evidence_by_match={
                pair: result.evidence_by_match[pair] for pair in selected_pairs
            },
            target_evidence=pair_target_evidence(selected_pairs),
        )

    @staticmethod
    def _summary_pairs(binding, sources, targets):
        """Partner pairs of one target neuron compared by a summary group."""
        if binding == "between_sources_targets":
            return [
                RelationshipMatch(source, target)
                for source in sources
                for target in targets
                if source != target
            ]

        members = sources if binding == "between_sources" else targets

        # Both orientations: directional statistics stay well defined.
        return [RelationshipMatch(a, b) for a in members for b in members if a != b]

    def _pair_values(self, condition, pairs):
        """Statistic values of explicit pairs: {RelationshipMatch: value}.

        Pairs without a value (e.g. not computable) are missing.
        """
        if not condition.threshold.active:
            raise CurationFilterError(
                "Curation filter condition has no active threshold."
            )

        if not self.engine.supports_requested_pairs(condition.query):
            raise CurationFilterError(
                "Pair summaries need a statistic supporting requested pairs, "
                "with all identities kept or fixed."
            )

        if not pairs:
            return {}

        rows = np.array(
            [
                (
                    pair.source.neuron_id,
                    pair.target.neuron_id,
                    -1 if pair.source.session_id is None else pair.source.session_id,
                    -1 if pair.target.session_id is None else pair.target.session_id,
                )
                for pair in pairs
            ],
            dtype=np.int64,
        )

        table = self.engine.evaluate_matching_table(
            condition.query,
            lambda values: np.ones(np.shape(values), dtype=bool),
            requested_pairs=rows,
        )

        if table is None:
            raise CurationFilterError("No statistic is available for this condition.")

        endpoints = self._row_endpoints(table, np.arange(table.n_rows, dtype=np.intp))
        values = np.asarray(table.values, dtype=np.float64)

        return {
            RelationshipMatch(source, target): float(value)
            for (source, target), value in zip(endpoints, values)
            if np.isfinite(value)
        }

    def _apply_summary_groups(self, result, groups):
        """Keep target neurons whose partner pairs pass every summary group.

        Per target neuron of the surviving relationships, each condition is
        evaluated on all pairs of its scope (between sources / between
        targets / between sources and targets); its values are combined
        by the group's summary and the threshold applies to that value.
        Scopes without any value fail.
        """
        if result.match_level != "relationship":
            raise CurationFilterError(
                "Between-partner summaries need a source-target parent."
            )

        reduce = {
            "median": np.median,
            "mean": np.mean,
            "min": np.min,
            "max": np.max,
        }

        members = {}

        for pair in result.matches:
            sources, targets = members.setdefault(pair.target.neuron_id, (set(), set()))
            sources.add(pair.source)
            targets.add(pair.target)

        passed = set(members)

        for group in groups:
            if (
                group.operator != "and"
                or group.match_level != "relationship"
                or group.count is not None
                or any(not is_filter_condition(c) for c in group.children)
            ):
                raise CurationFilterError(
                    "Use an AND source-target group with conditions and no "
                    "count for between-partner summaries.",
                    node_id=group.id,
                    node_type="group",
                )

            scopes = {
                neuron: self._summary_pairs(group.apply_to, *members[neuron])
                for neuron in passed
            }
            pairs = {pair for scope in scopes.values() for pair in scope}

            for condition in group.children:
                try:
                    values = self._pair_values(condition, pairs)
                except CurationFilterError as exc:
                    exc.attach_node(condition.id, "condition")
                    raise

                for neuron in list(passed):
                    scope = [values[p] for p in scopes[neuron] if p in values]

                    if not scope or not bool(
                        condition.threshold.mask(
                            np.array([reduce[group.summary](scope)])
                        )[0]
                    ):
                        passed.discard(neuron)

        selected = {pair for pair in result.matches if pair.target.neuron_id in passed}

        return replace(
            result,
            matches=selected,
            evidence_by_match={pair: result.evidence_by_match[pair] for pair in selected},
            target_evidence=pair_target_evidence(selected),
        )

    def _record_support_inspection(self, groups, pairs):
        mapping = pair_target_evidence(pairs)

        for group in groups:
            self._bound_inspections[group.id] = mapping

            for child in group.children:
                self._bound_inspections[child.id] = mapping

    @reports_progress("Evaluating curator group")
    def _evaluate_group(
        self,
        group: CurationFilterGroup,
        condition_results: dict[str, CurationFilterConditionResult],
        group_results: dict[str, CurationFilterGroupResult],
        *,
        initial_pairs=None,
        initial_kinds=None,
        endpoint_domain=None,
    ) -> CurationFilterGroupResult:
        """``endpoint_domain``: footprints for which this group's result must
        be exact (a source subgroup only checks the parent's sources). Other
        footprints may be missing from the result."""

        if initial_pairs is None and endpoint_domain is not None:
            initial_pairs = self._domain_pairs(group, endpoint_domain)

            if initial_pairs is not None and self._existence_count(group):
                initial_pairs = self._existence_pairs(group, initial_pairs)

        bound_children = [
            child
            for child in group.children
            if is_filter_group(child) and child.apply_to != "same"
        ]

        ordinary_children = [
            child
            for child in group.children
            if not (is_filter_group(child) and child.apply_to != "same")
        ]

        for child in bound_children:
            if child.apply_to not in (
                "source",
                "target",
                *BETWEEN_BINDINGS,
            ):
                raise CurationFilterError(
                    "Unknown subgroup binding.",
                    node_id=child.id,
                    node_type="group",
                )

            if child.summary is not None and child.summary not in PAIR_SUMMARIES:
                raise CurationFilterError(
                    f"Unknown pair summary: {child.summary!r}",
                    node_id=child.id,
                    node_type="group",
                )

            if child.apply_to == "between_sources_targets" and child.summary is None:
                raise CurationFilterError(
                    "Between sources and targets: choose how pairs are "
                    "combined (median, mean, min or max).",
                    node_id=child.id,
                    node_type="group",
                )

        if bound_children and (
            group.operator != "and"
            or group.match_level != "relationship"
            or not ordinary_children
        ):
            raise CurationFilterError(
                "Source/target subgroups require an AND source-target parent "
                "with at least one ordinary relationship condition or group.",
                node_id=group.id,
                node_type="group",
            )

        # Largest compatible partner set (parent count) ...
        support_groups = [
            child
            for child in bound_children
            if child.apply_to in BETWEEN_BINDINGS and child.summary is None
        ]

        # ... or pair values combined per target neuron.
        summary_groups = [
            child
            for child in bound_children
            if child.apply_to in BETWEEN_BINDINGS and child.summary is not None
        ]

        endpoint_groups = [
            child
            for child in bound_children
            if child.apply_to not in BETWEEN_BINDINGS
        ]

        child_results = []
        endpoint_kinds = initial_kinds
        remaining_pairs = None if initial_pairs is None else set(initial_pairs)

        progressive = group.operator == "and" and group.match_level == "relationship"

        for progress_index, (child) in enumerate(ordinary_children):
            with progress_step(
                progress_index, len(ordinary_children), start=0, stop=70,
            ):

                # -------------------------------------
                # Condition
                # -------------------------------------

                if is_filter_condition(child):

                    try:
                        requested_pairs = None

                        if (
                            progressive
                            and remaining_pairs is not None
                            and self.engine.supports_requested_pairs(child.query)
                        ):
                            requested_pairs = np.fromiter(
                                (
                                    value
                                    for pair in remaining_pairs
                                    for value in (
                                        pair.source.neuron_id,
                                        pair.target.neuron_id,
                                        (
                                            -1
                                            if pair.source.session_id is None
                                            else pair.source.session_id
                                        ),
                                        (
                                            -1
                                            if pair.target.session_id is None
                                            else pair.target.session_id
                                        ),
                                    )
                                ),
                                dtype=np.int64,
                                count=4 * len(remaining_pairs),
                            ).reshape(-1, 4)

                        with progress_scope(0, 85):
                            condition_result = self._evaluate_condition(
                                child,
                                requested_pairs=requested_pairs,
                            )

                        condition_results[child.id] = condition_result

                        with progress_scope(85, 95, "Constructing condition matches"):
                            child_result = self._condition_result_for_level(
                                condition_result, group.match_level
                            )
                        if group.match_level == "relationship":
                            kinds = child_result.endpoint_kinds

                            if endpoint_kinds is not None and kinds != endpoint_kinds:
                                raise CurationFilterError(
                                    "Pair types differ after reduction: "
                                    f"expected {endpoint_kinds}, received {kinds}. "
                                    "Keep/fix a session for a footprint endpoint, "
                                    "or reduce it for a neuron endpoint."
                                )

                            endpoint_kinds = kinds

                        with progress_scope(95, 100, "Preparing condition domains"):
                            self._set_count_domains(
                                child_result,
                                condition_result.table,
                            )

                    except CurationFilterError as exc:
                        exc.attach_node(child.id, "condition")
                        raise

                # -------------------------------------
                # Nested group
                # -------------------------------------

                elif is_filter_group(child):

                    try:
                        with progress_scope(0, 85):
                            child_result = self._evaluate_group(
                                child,
                                condition_results,
                                group_results,
                                endpoint_domain=(
                                    endpoint_domain
                                    if group.match_level == "footprint"
                                    else None
                                ),
                            )

                        with progress_scope(85, 100, "Projecting subgroup matches"):
                            child_result = self._project_group_result(
                                child_result, group.match_level
                            )

                    except CurationFilterError as exc:
                        exc.attach_node(child.id, "group")
                        raise

                else:
                    raise TypeError(f"Unknown curation filter node: " f"{type(child)!r}")

                if group.match_level == "relationship":
                    if (
                        endpoint_kinds is not None
                        and child_result.endpoint_kinds != endpoint_kinds
                    ):
                        raise CurationFilterError(
                            "Combined relationship groups must "
                            "have the same endpoint types.",
                            node_id=child.id,
                            node_type=child.node_type,
                        )

                    endpoint_kinds = child_result.endpoint_kinds

                child_results.append(child_result)

                if progressive:
                    if remaining_pairs is None:
                        remaining_pairs = set(child_result.matches)
                    else:
                        remaining_pairs.intersection_update(child_result.matches)

        # -----------------------------------------
        # Empty group
        # -----------------------------------------

        if not child_results:

            result = CurationFilterGroupResult(
                group_id=group.id,
                match_level=group.match_level,
                matches=set(),
                evidence_by_match={},
            )

            group_results[group.id] = result

            return result

        # -----------------------------------------
        # Boolean operation
        # -----------------------------------------

        with progress_scope(70, 73, "Combining curator conditions"):
            if group.operator == "and":

                matches = set(child_results[0].matches)

                for child_result in child_results[1:]:
                    matches.intersection_update(child_result.matches)

            elif group.operator == "or":

                matches = set()

                for child_result in child_results:
                    matches.update(child_result.matches)

            else:
                raise ValueError(group.operator)

        # -----------------------------------------
        # Evidence only for identities which
        # survived this group's Boolean operation.
        # -----------------------------------------

        with progress_scope(73, 80, "Combining match evidence"):
            evidence_by_match = {}

            for match in progress_iter(matches):

                evidence = set()

                for child_result in child_results:

                    # With OR, only contributing children matter.
                    # With AND, every child contains the match anyway.
                    if match not in child_result.matches:
                        continue

                    evidence.update(child_result.evidence_by_match.get(match, set()))

                evidence_by_match[match] = evidence

            result = CurationFilterGroupResult(
                group_id=group.id,
                match_level=group.match_level,
                matches=matches,
                evidence_by_match=evidence_by_match,
                target_evidence=combined_targets(
                    child_results,
                    matches,
                    group.match_level,
                ),
            )

            result.endpoint_kinds = endpoint_kinds
            combine = set.intersection if group.operator == "and" else set.union

            result.neuron_domain = combine(*(r.neuron_domain for r in child_results))
            result.footprint_domain = combine(*(r.footprint_domain for r in child_results))

        if group is self._review_root and self._review_allowed is not None:
            allowed = self._review_allowed

            kept = {
                match for match in result.matches if match_neurons(match) <= allowed
            }

            result = replace(
                result,
                matches=kept,
                evidence_by_match={
                    match: result.evidence_by_match[match] for match in kept
                },
                target_evidence=restrict_targets(
                    result.target_evidence,
                    kept,
                    result.match_level,
                ),
                neuron_domain=result.neuron_domain & allowed,
                footprint_domain={
                    component
                    for component in result.footprint_domain
                    if component.neuron_id in allowed
                },
            )

        for progress_index, (child) in enumerate(endpoint_groups):
            with progress_step(
                progress_index, len(endpoint_groups), start=80, stop=86,
            ):
                try:
                    # A source requirement only filters existing pairs, so
                    # only their sources need a result. A target requirement
                    # also restricts zero-support targets: evaluate fully.
                    domain = None

                    if (
                        child.apply_to == "source"
                        and child.match_level == "footprint"
                        and result.match_level == "relationship"
                    ):
                        domain = {pair.source for pair in result.matches}

                    child_result = self._evaluate_group(
                        child,
                        condition_results,
                        group_results,
                        endpoint_domain=domain,
                    )

                    result = self._apply_endpoint_group(
                        result,
                        child_result,
                        child.apply_to,
                    )

                except CurationFilterError as exc:
                    exc.attach_node(child.id, "group")
                    raise

        with progress_scope(86, 89, "Evaluating partner summaries"):
            if summary_groups:
                result = self._apply_summary_groups(result, summary_groups)

        with progress_scope(89, 92, "Evaluating partner requirements"):
            if support_groups:
                result = self._apply_support_groups(
                    group,
                    result,
                    support_groups,
                    condition_results,
                    group_results,
                )

        # Subsequent counting and condition projection must use the
        # relationships that survived all source/target requirements.
        matches = result.matches

        with progress_scope(92, 95, "Counting surviving matches"):
            if group.count is not None:
                try:
                    result = self._apply_count(group, result)
                except CurationFilterError as exc:
                    exc.attach_node(group.id, "group")
                    raise

        for progress_index, (child, child_result) in enumerate(zip(ordinary_children, child_results)):
            with progress_step(
                progress_index, len(ordinary_children), start=95, stop=98, message="Projecting surviving conditions",
            ):
                if not is_filter_condition(child):
                    continue

                surviving = child_result.matches & matches

                projected = replace(
                    child_result,
                    matches=surviving,
                    evidence_by_match={
                        match: child_result.evidence_by_match[match] for match in surviving
                    },
                    target_evidence=restrict_targets(
                        child_result.target_evidence,
                        surviving,
                        child_result.match_level,
                    ),
                )

                if group.count is not None and result.match_level == "relationship":
                    kept = projected.matches & result.matches

                    projected = replace(
                        projected,
                        matches=kept,
                        evidence_by_match={
                            pair: projected.evidence_by_match[pair] for pair in kept
                        },
                        target_evidence=pair_target_evidence(kept),
                    )

                elif group.count is not None:
                    _, evidence = self._count_buckets(
                        projected,
                        group.count.target,
                        group.count.unit,
                    )

                    projected = replace(
                        result,
                        evidence_by_match={
                            subject: evidence.get(subject, set())
                            for subject in result.matches
                        },
                        target_evidence=restrict_targets(
                            projected.target_evidence,
                            result.matches,
                            result.match_level,
                        ),
                    )

                condition_results[child.id].projected_result = projected

        with progress_scope(98, 99, "Constructing bound inspections"):
            if bound_children:
                surviving_pairs = {
                    pair
                    for pair in matches
                    if (
                        pair in result.matches
                        if result.match_level == "relationship"
                        else self._endpoint_matches(result, pair.target)
                    )
                }

                self._record_support_inspection(
                    support_groups + summary_groups,
                    surviving_pairs,
                )

                for child in endpoint_groups:
                    self._record_bound_inspection(
                        child,
                        child.apply_to,
                        surviving_pairs,
                        condition_results,
                        group_results,
                    )
        group_results[group.id] = result
        return result

    def _domain_pairs(self, group, domain):
        """Candidate pairs of a relationship group needed for ``domain``.

        Applies to AND groups counting per target footprint where a condition
        restricts partners to the same neuron: all same-neuron partners of
        each domain footprint then contain every pair that can match, so
        counts of domain footprints are exact. None if not applicable.
        """
        spec = group.count

        if (
            group.operator != "and"
            or group.match_level != "relationship"
            or spec is None
            or spec.anchor != "target"
            or spec.target != "footprint"
            or spec.separate_partner_sessions
            or any(component.session_id is None for component in domain)
        ):
            return None

        if not any(
            is_filter_condition(child)
            and self.engine.supports_requested_pairs(child.query)
            and any(
                f.target == "neuron" and f.relation == "same"
                for f in child.query.filters
            )
            for child in group.children
        ):
            return None

        ids = _active_ids(self.engine)
        pairs = set()

        for anchor in domain:
            for session_id in np.flatnonzero(ids[anchor.neuron_id] >= 0):
                if session_id != anchor.session_id:
                    pairs.add(
                        RelationshipMatch(
                            NeuronComponent(anchor.neuron_id, int(session_id)),
                            anchor,
                        )
                    )

        return pairs

    @staticmethod
    def _existence_count(group):
        """Count "≥ 1 partner" over conditions only: one match suffices."""
        spec = group.count

        return (
            spec is not None
            and spec.comparison == "ge"
            and spec.value == 1
            and all(is_filter_condition(child) for child in group.children)
        )

    @reports_progress("Searching matching partners")
    def _existence_pairs(self, group, candidates):
        """Matching pairs found by evaluating partners nearest-session first.

        Rounds evaluate 1, 2, 4, ... further partners of anchors that have
        no match yet. Each anchor ends with at least one matching pair iff
        any candidate matches, so the group's count "≥ 1" selects the same
        anchors as with all candidates. Counts and inspection evidence of
        the group then show only the partners found.
        """
        by_anchor = {}

        for pair in candidates:
            by_anchor.setdefault(pair.target, []).append(pair)

        for pairs in by_anchor.values():
            pairs.sort(
                key=lambda pair: (
                    abs(pair.source.session_id - pair.target.session_id),
                    pair.source.session_id,
                )
            )

        # Conditions only; the count is applied by the final evaluation.
        probe = replace(group, count=None)

        matched = set()
        pending = by_anchor
        offset, width = 0, 1

        while pending:
            batch = {
                pair
                for pairs in pending.values()
                for pair in pairs[offset : offset + width]
            }

            result = self._evaluate_group(probe, {}, {}, initial_pairs=batch)
            matched.update(result.matches)

            found = {pair.target for pair in result.matches}
            offset += width
            width *= 2

            pending = {
                anchor: pairs
                for anchor, pairs in pending.items()
                if anchor not in found and len(pairs) > offset
            }

        return matched

    @reports_progress('Preparing count domains')
    def _prepare_count_domains(self, root):
        def enabled(group):
            return group.count is not None or any(
                enabled(child) for child in group.children if is_filter_group(child)
            )

        self._counting = enabled(root)
        self._count_components = set()
        self._count_neurons = set()

        if not self._counting:
            return

        # Footprints of inactive sessions are not part of the domain.
        ids = np.asarray(_active_ids(self.engine), dtype=int)
        included = np.ones(ids.shape[0], dtype=bool)

        union = self.engine.data.assignments.union

        if union is not None and union.included is not None:
            included = np.asarray(union.included, dtype=bool)

            if included.shape != (ids.shape[0],):
                raise CurationFilterError("Union inclusion mask has the wrong length.")

        for sid, session in enumerate(progress_iter(self.engine.data.sessions[: ids.shape[1]], stop=95, every=1)):
            if session is None:
                continue

            size = int(session.n_neurons)
            keep = np.ones(size, dtype=bool)

            if session.included is not None:
                keep = np.asarray(session.included, dtype=bool)

                if keep.shape != (size,):
                    raise CurationFilterError(
                        f"Session {sid}: inclusion mask has the wrong length."
                    )

            valid = included & (ids[:, sid] >= 0) & (ids[:, sid] < size)

            rows = np.flatnonzero(valid)
            rows = rows[keep[ids[rows, sid]]]

            self._count_components.update(NeuronComponent(int(n), sid) for n in rows)

        self._count_neurons = {c.neuron_id for c in self._count_components}

    @reports_progress('Preparing condition domains')
    def _set_count_domains(self, result, table):
        if not self._counting:
            return

        def coords(name):
            info = table.stat.dimensions.get(name)

            if info is not None:
                if info.mode == "fixed":
                    return {int(info.parameter)}

                if info.mode == "remaining":
                    return set(map(int, info.coords))

                return None

            binding = table.stat.reference_aliases.get(name)

            if binding is not None:
                base, offset = binding
                values = coords(base)

                return None if values is None else {v + offset for v in values}

            if name.endswith(("_i", "_j")):
                return coords(name.rsplit("_", 1)[0])

            return None

        axes = table.stat.component_axes

        if axes is None:
            axes = []

            for neuron, sessions in (
                ("neuron", ("session", "session_i", "session_j")),
                ("neuron_i", ("session_i", "session")),
                ("neuron_j", ("session_j", "session")),
            ):
                for session in sessions:
                    if coords(neuron) is not None and coords(session) is not None:
                        axes.append((neuron, session))

                        if neuron != "neuron":
                            break

        available = [
            (n, s) for n, s in axes if coords(n) is not None and coords(s) is not None
        ]

        def components(neuron, session):
            ns = coords(neuron)
            ss = coords(session)

            return {
                c
                for c in self._count_components
                if c.neuron_id in ns and c.session_id in ss
            }

        if result.match_level == "relationship":
            neuron, session = self._relationship_axes_for_stat(table.stat)[1]

            result.neuron_domain = self._count_neurons & coords(neuron)

            result.footprint_domain = (
                set() if session is None else components(neuron, session)
            )

        elif result.match_level == "reassignment":
            source, session = available[0]

            destination = "neuron_j" if source == "neuron_i" else "neuron_i"

            result.neuron_domain = self._count_neurons & coords(destination)

            # An existing destination footprint in the candidate's session.
            result.footprint_domain = components(destination, session)

        elif result.match_level == "footprint":
            result.footprint_domain = set().union(
                *(components(n, s) for n, s in available)
            )

            result.neuron_domain = {c.neuron_id for c in result.footprint_domain}

        else:
            result.neuron_domain = self._count_neurons & set().union(
                *(coords(n) or set() for n in ("neuron", "neuron_i", "neuron_j"))
            )

    @reports_progress('Grouping counts')
    def _count_buckets(self, result, target, unit="footprint"):
        if target == "footprint" and result.match_level not in (
            "footprint",
            "reassignment",
            "relationship",
        ):
            raise CurationFilterError(
                "Counting per footprint requires footprint "
                "or reassignment identities."
            )

        if result.match_level == "relationship":
            source_kind, target_kind = result.endpoint_kinds

            if target == "footprint" and target_kind != "footprint":
                raise CurationFilterError("This target is a neuron. Count per neuron.")

            if unit == "footprint" and source_kind != "footprint":
                raise CurationFilterError(
                    "These sources are neurons. Count source neurons."
                )
        buckets = {}
        evidence = {}

        for match in progress_iter(result.matches, stop=95):
            if isinstance(match, RelationshipMatch):
                subject = (
                    match.target if target == "footprint" else match.target.neuron_id
                )

                occurrence = (
                    match.source if unit == "footprint" else match.source.neuron_id
                )

                entries = [(subject, occurrence)]

            elif isinstance(match, ReassignmentMatch):
                # Keep the existing reassignment branch here.
                source = NeuronComponent(
                    match.source_neuron,
                    match.source_session,
                )

                if target == "neuron":
                    subject = match.destination_neuron
                else:
                    subject = NeuronComponent(
                        match.destination_neuron,
                        match.source_session,
                    )

                entries = [(subject, source)]

            elif isinstance(match, NeuronPairMatch):
                entries = [
                    (match.first, match.second),
                    (match.second, match.first),
                ]

            elif isinstance(match, NeuronComponent):
                subject = match.neuron_id if target == "neuron" else match

                entries = [(subject, match)]

            else:
                entries = [(int(match), int(match))]

            for subject, occurrence in entries:
                # Sets prevent duplicate counting across conditions.
                buckets.setdefault(subject, set()).add(occurrence)

                evidence.setdefault(subject, set()).update(
                    result.evidence_by_match.get(match, set())
                )

        return buckets, evidence

    @reports_progress('Grouping relationships')
    def _relationship_buckets(self, result, spec):
        if result.match_level != "relationship":
            raise CurationFilterError(
                "Source/target counting requires relationship matching."
            )

        source_kind, target_kind = result.endpoint_kinds

        anchor_kind, partner_kind = (
            (source_kind, target_kind)
            if spec.anchor == "source"
            else (target_kind, source_kind)
        )

        if spec.target == "footprint" and anchor_kind != "footprint":
            raise CurationFilterError("The count anchor is a neuron. Count per neuron.")

        if spec.unit == "footprint" and partner_kind != "footprint":
            raise CurationFilterError(
                "The partners are neurons. Count distinct neurons."
            )

        if spec.separate_partner_sessions and partner_kind != "footprint":
            raise CurationFilterError("Session grouping requires footprint partners.")

        buckets = {}

        for pair in progress_iter(result.matches, stop=95):
            anchor, partner = (
                (pair.source, pair.target)
                if spec.anchor == "source"
                else (pair.target, pair.source)
            )

            subject = anchor if spec.target == "footprint" else anchor.neuron_id

            key = (
                (subject, partner.session_id)
                if spec.separate_partner_sessions
                else subject
            )

            buckets.setdefault(key, set()).add(pair)

        return buckets

    @reports_progress("Counting relationship partners")
    def _filter_relationship_count(self, result, spec):
        if spec.accepts(0):
            raise CurationFilterError(
                "This count keeps source–target relationships, so zero "
                "partners cannot supply a target to select. Use a positive "
                "count here. Ordinary per-target counting still supports "
                "zero matches."
            )

        with progress_scope(0, 30):
            buckets = self._relationship_buckets(result, spec)
        counts = {}
        kept = set()

        for key, pairs in progress_iter(buckets.items(), start=30, stop=90):
            partners = {
                pair.target if spec.anchor == "source" else pair.source
                for pair in pairs
            }

            counted = (
                partners
                if spec.unit == "footprint"
                else {component.neuron_id for component in partners}
            )

            counts[key] = len(counted)

            if spec.accepts(counts[key]):
                kept.update(pairs)

        return replace(
            result,
            matches=kept,
            counts=counts,
            evidence_by_match={pair: result.evidence_by_match[pair] for pair in kept},
            target_evidence=pair_target_evidence(kept),
        )

    @reports_progress('Counting matches')
    def _apply_count(self, group, result):
        spec = group.count
        if spec.anchor == "source" or spec.separate_partner_sessions:
            return self._filter_relationship_count(result, spec)

        if result.match_level == spec.target and spec.value > 1:
            raise CurationFilterError(
                "These identities have already been collapsed to one match "
                "each. Use source-target matching to count distinct partners."
            )

        with progress_scope(0, 65):
            buckets, evidence = self._count_buckets(
                result,
                spec.target,
                spec.unit,
            )

        domain = (
            result.neuron_domain if spec.target == "neuron" else result.footprint_domain
        )

        with progress_scope(65, 85, "Counting matched subjects"):
            counts = {subject: len(buckets.get(subject, ())) for subject in domain}

        with progress_scope(85, 95, "Applying count threshold"):
            matches = {subject for subject, value in counts.items() if spec.accepts(value)}

        return CurationFilterGroupResult(
            group_id=result.group_id,
            match_level=spec.target,
            matches=matches,
            evidence_by_match={
                subject: evidence.get(subject, set()) for subject in matches
            },
            neuron_domain=(
                set(domain)
                if spec.target == "neuron"
                else {c.neuron_id for c in domain}
            ),
            footprint_domain=(set(domain) if spec.target == "footprint" else set()),
            counts=counts,
            target_evidence=restrict_targets(
                result.target_evidence,
                matches,
                spec.target,
            ),
        )

    @reports_progress("Constructing candidate matches")
    def _candidate_result(self, result, match_level):
        table = result.table
        refs = table.refs_for_rows(
            result.matching_rows,
            include_fixed=True,
        )

        missing = {"neuron_i", "neuron_j"} - set(refs)

        if missing:
            raise CurationFilterError(
                "Keep or fix both neuron dimensions to preserve "
                "candidate identity: " + ", ".join(sorted(missing))
            )

        if match_level == "reassignment":
            if table.stat.curation_kind not in (None, "reassignment"):
                raise CurationFilterError(
                    "This statistic does not describe " "reassignment candidates."
                )

            axes = table.stat.component_axes

            if axes is None:
                same_session = any(
                    f.target == "session" and f.relation == "same"
                    for f in result.condition.query.filters
                )

                if same_session:
                    # Both footprints belong to the same session.
                    # Interpret i as the candidate and j as the target.
                    session_axis = "session" if "session" in refs else "session_i"

                    axes = (("neuron_i", session_axis),)

                else:
                    axes = (
                        ("neuron_i", "session_i"),
                        ("neuron_j", "session_j"),
                    )

            available = [
                (neuron, session)
                for neuron, session in axes
                if neuron in refs and session in refs
            ]

            if len(available) != 1:
                raise CurationFilterError(
                    "A reassignment candidate needs exactly one concrete "
                    "source footprint and both neuron identities. "
                    "Keep or fix the source session, and reduce the "
                    "destination session dimension."
                )

            source_axis, session_axis = available[0]

            if source_axis not in ("neuron_i", "neuron_j"):
                raise CurationFilterError("The source must belong to one neuron axis.")

            destination_axis = "neuron_j" if source_axis == "neuron_i" else "neuron_i"

        elif table.stat.curation_kind != "neuron_pair":
            raise CurationFilterError(
                "Choose a neuron-pair statistic for " "this group's matching mode."
            )

        evidence = {}

        for index in progress_iter(range(len(result.matching_rows)), start=5, stop=90):
            a = int(refs["neuron_i"][index])
            b = int(refs["neuron_j"][index])

            if a == b:
                continue

            if match_level == "reassignment":
                source = int(refs[source_axis][index])
                session = int(refs[session_axis][index])
                destination = int(refs[destination_axis][index])

                match = ReassignmentMatch(
                    source_neuron=source,
                    source_session=session,
                    destination_neuron=destination,
                )

                components = {
                    NeuronComponent(
                        neuron_id=source,
                        session_id=session,
                    )
                }

            else:
                match = NeuronPairMatch(
                    first=min(a, b),
                    second=max(a, b),
                )
                components = set()

            evidence.setdefault(match, set()).update(components)

        return CurationFilterGroupResult(
            group_id=result.condition.id,
            match_level=match_level,
            matches=set(evidence),
            evidence_by_match=evidence,
        )

    @staticmethod
    def _relationship_axes_for_stat(stat):
        axes = stat.relationship_axes

        if axes is None:
            raise CurationFilterError(
                "This statistic does not declare pair identities."
            )

        def dimension(name):
            info = stat.dimensions.get(name)

            if info is None:
                base = stat.reference_aliases.get(name, (name, 0))[0]
                base = stat.reduction_aliases.get(base, base)
                info = stat.dimensions.get(base)

            if info is None:
                raise CurationFilterError(f"Missing dimension metadata for {name}.")

            return info

        effective = []

        for neuron, session in axes:
            if dimension(neuron).mode not in ("remaining", "fixed"):
                raise CurationFilterError(
                    "Pair matching needs a neuron identity: "
                    f"keep or fix {neuron}. "
                    "Session dimensions may be reduced."
                )

            if session is not None:
                mode = dimension(session).mode

                if mode == "reduced":
                    session = None

                elif mode not in ("remaining", "fixed"):
                    raise CurationFilterError(
                        f"Unsupported dimension state for {session}: {mode}"
                    )

            effective.append((neuron, session))

        return tuple(effective)

    def _row_endpoints(self, table, rows):
        """(source, target) components of table rows (relationship axes)."""
        axes = self._relationship_axes_for_stat(table.stat)

        refs = table.refs_for_rows(
            rows,
            include_fixed=True,
        )

        required = {name for endpoint in axes for name in endpoint if name is not None}

        for name in required - refs.keys():
            binding = table.stat.reference_aliases.get(name)

            if binding is not None and binding[0] in refs:
                refs[name] = refs[binding[0]] + binding[1]

        missing = required - refs.keys()

        if missing:
            raise CurationFilterError(
                "Source–target matching requires concrete identities. "
                "Keep or fix: "
                + ", ".join(sorted(missing))
                + ". Reduced session dimensions are represented as neuron endpoints."
            )

        def endpoint(spec, index):
            neuron, session = spec

            return NeuronComponent(
                int(refs[neuron][index]),
                (None if session is None else int(refs[session][index])),
            )

        return [
            (endpoint(axes[0], index), endpoint(axes[1], index))
            for index in range(len(rows))
        ]

    @reports_progress('Constructing relationships')
    def _relationship_result(self, result):
        table = result.table
        axes = self._relationship_axes_for_stat(table.stat)

        endpoints = self._row_endpoints(table, result.matching_rows)

        evidence = {}

        for index in progress_iter(range(len(endpoints)), start=5, stop=90):
            source, target = endpoints[index]

            if source != target:
                evidence[RelationshipMatch(source, target)] = {source}

        return CurationFilterGroupResult(
            group_id=result.condition.id,
            match_level="relationship",
            matches=set(evidence),
            evidence_by_match=evidence,
            target_evidence=pair_target_evidence(evidence),
            endpoint_kinds=tuple(
                "neuron" if session is None else "footprint" for _, session in axes
            ),
        )

    @reports_progress('Projecting matching rows')
    def _project_rows(self, table, rows):
        rows = np.asarray(rows, dtype=int)

        if rows.size == 0:
            return set(), set(), {}, {}

        refs = table.refs_for_rows(rows, include_fixed=True)

        rows_by_neuron = {}
        rows_by_component = {}

        for index, table_row in enumerate(progress_iter(rows, start=5, stop=90)):
            neurons = {
                int(refs[key][index])
                for key in ("neuron", "neuron_i", "neuron_j")
                if key in refs
            }

            components = _component_refs_for_row(
                refs,
                index,
                table.stat.component_axes,
            )

            for neuron_id in neurons:
                rows_by_neuron.setdefault(neuron_id, []).append(int(table_row))

            for component in components:
                rows_by_component.setdefault(component, []).append(int(table_row))

        rows_by_neuron = {
            key: np.asarray(value, dtype=int) for key, value in rows_by_neuron.items()
        }

        rows_by_component = {
            key: np.asarray(value, dtype=int)
            for key, value in rows_by_component.items()
        }

        return (
            set(rows_by_neuron),
            set(rows_by_component),
            rows_by_neuron,
            rows_by_component,
        )

    @reports_progress("Constructing condition evidence")
    def _condition_result_for_level(
        self,
        result: CurationFilterConditionResult,
        match_level: FilterMatchLevel,
    ) -> CurationFilterGroupResult:
        if match_level == "relationship":
            return self._relationship_result(result)
        if match_level in ("reassignment", "neuron_pair"):
            return self._candidate_result(result, match_level)

        if match_level == "neuron":

            matches = set(result.neurons)

            evidence_by_match = {
                neuron_id: result.components_for_neuron(neuron_id)
                for neuron_id in progress_iter(matches, stop=95)
            }

        elif match_level == "footprint":

            if not table_supports_component_identity(result.table):
                raise CurationFilterError(
                    f"{result.table.stat.display_title!r} "
                    "cannot be evaluated 'by footprint' because "
                    "the query does not retain a session/footprint identity. "
                    "Keep a session dimension, or use 'by neuron'."
                )

            # A structurally footprint-capable table with matching rows
            # should actually have produced components.
            if result.matching_rows.size > 0 and not result.components:
                raise RuntimeError(
                    "Footprint-capable curation condition produced "
                    "matching rows but no NeuronComponents."
                )

            matches = set(result.components)

            evidence_by_match = {
                component: result.components_for_component(component)
                for component in progress_iter(matches, stop=95)
            }

        else:
            raise ValueError(match_level)

        return CurationFilterGroupResult(
            group_id=result.condition.id,
            match_level=match_level,
            matches=matches,
            evidence_by_match=evidence_by_match,
        )

    @reports_progress("Projecting subgroup evidence")
    def _project_group_result(self, result, match_level):
        if result.match_level == match_level:
            return result

        if match_level == "neuron":
            evidence = {}

            for match in progress_iter(result.matches, stop=95):
                for neuron_id in match_neurons(match):
                    evidence.setdefault(neuron_id, set()).update(
                        result.evidence_by_match.get(match, set())
                    )

            return CurationFilterGroupResult(
                group_id=result.group_id,
                match_level="neuron",
                matches=set(evidence),
                evidence_by_match=evidence,
                neuron_domain=set(result.neuron_domain),
                target_evidence=restrict_targets(
                    result.target_evidence,
                    set(evidence),
                    "neuron",
                ),
            )

        raise CurationFilterError(
            f"Cannot convert a {result.match_level!r} group "
            f"to {match_level!r}. "
            "Use the same candidate mode for conditions "
            "that must match the same pair."
        )


def _component_refs_for_row(refs, index, component_axes=None):
    if component_axes is not None:
        return {
            NeuronComponent(
                neuron_id=int(refs[neuron_axis][index]),
                session_id=int(refs[session_axis][index]),
            )
            for neuron_axis, session_axis in component_axes
            if neuron_axis in refs and session_axis in refs
        }

    components = set()

    # A single neuron observed in one or two sessions.
    if "neuron" in refs:
        session_keys = [key for key in ("session_i", "session_j") if key in refs]

        if not session_keys and "session" in refs:
            session_keys = ["session"]

        for key in session_keys:
            components.add(
                NeuronComponent(
                    neuron_id=int(refs["neuron"][index]),
                    session_id=int(refs[key][index]),
                )
            )

    # Each pair side belongs to its corresponding session.
    for suffix in ("i", "j"):
        neuron_key = f"neuron_{suffix}"

        if neuron_key not in refs:
            continue

        session_key = f"session_{suffix}"

        if session_key not in refs:
            session_key = "session"

        if session_key in refs:
            components.add(
                NeuronComponent(
                    neuron_id=int(refs[neuron_key][index]),
                    session_id=int(refs[session_key][index]),
                )
            )

    return components


def components_for_table_rows(
    table: PickTable,
    rows,
) -> set[NeuronComponent]:
    rows = np.atleast_1d(np.asarray(rows, dtype=int))

    if rows.size == 0:
        return set()

    refs = table.refs_for_rows(rows, include_fixed=True)

    components = set()

    for index in range(rows.size):
        components.update(
            _component_refs_for_row(
                refs,
                index,
                table.stat.component_axes,
            )
        )

    return components


def table_supports_component_identity(
    table: PickTable,
) -> bool:
    """
    Return whether the evaluated table retains enough semantic
    information to identify at least one concrete footprint.

    Fixed dimensions count as available because
    refs_for_rows(..., include_fixed=True) restores them.
    """

    available = set(table.refs)

    available.update(
        dim_name
        for dim_name, dim in table.stat.dimensions.items()
        if dim.mode == "fixed"
    )

    for original, (compact, _) in table.stat.reference_aliases.items():
        if compact in available:
            available.add(original)

    if table.stat.component_axes is not None:
        return any(
            neuron_axis in available and session_axis in available
            for neuron_axis, session_axis in table.stat.component_axes
        )

    # Ordinary neuron/session statistic.
    if "neuron" in available:
        if "session" in available:
            return True

        # Same neuron represented across a session pair.
        if "session_i" in available or "session_j" in available:
            return True

    # Pairwise neuron dimensions with either their corresponding
    # session dimension or one shared/collapsed session.
    if "neuron_i" in available:
        if "session_i" in available or "session" in available:
            return True

    if "neuron_j" in available:
        if "session_j" in available or "session" in available:
            return True

    return False


def is_filter_condition(node) -> bool:
    return getattr(node, "node_type", None) == "condition"


def is_filter_group(node) -> bool:
    return getattr(node, "node_type", None) == "group"


def empty_filter_group_ids(
    group: CurationFilterGroup,
) -> set[str]:

    empty = set()

    if not group.children:
        empty.add(group.id)

    for child in group.children:

        if is_filter_group(child):
            empty.update(empty_filter_group_ids(child))

    return empty


CURATION_FILTER_SCHEMA_VERSION = 2


def curation_filter_node_to_dict(
    node,
) -> dict:

    if is_filter_condition(node):

        threshold = node.threshold
        threshold.validate()

        threshold_data = {
            "direction": threshold.direction,
            "active": bool(threshold.active),
        }

        # A recipe replaces the value: it is resolved on evaluation.
        if threshold.reference is None:
            threshold_data["value"] = float(threshold.value)
        else:
            threshold_data["reference"] = threshold.reference.to_dict()

        if threshold.direction == "between":
            if threshold.upper_reference is None:
                threshold_data["upper_value"] = float(threshold.upper_value)
            else:
                threshold_data["upper_reference"] = threshold.upper_reference.to_dict()

        # Curator thresholds are currently axis-independent,
        # but retain this if ThresholdSpec has an axis field.
        axis = getattr(
            threshold,
            "axis",
            None,
        )

        if axis is not None:
            threshold_data["axis"] = axis

        return {
            "type": "condition",
            "id": node.id,
            "query": node.query.to_dict(),
            "threshold": threshold_data,
        }

    if is_filter_group(node):

        return {
            "type": "group",
            "id": node.id,
            "operator": node.operator,
            "match_level": node.match_level,
            "children": [
                curation_filter_node_to_dict(child) for child in node.children
            ],
            "count": (None if node.count is None else asdict(node.count)),
            "comment": node.comment,
            "apply_to": node.apply_to,
            "summary": node.summary,
            "manipulation": (
                None if node.manipulation is None else asdict(node.manipulation)
            ),
            "review_statuses": (
                None if node.review_statuses is None else list(node.review_statuses)
            ),
        }

    raise TypeError("Unknown curation filter node: " f"{type(node)!r}")


def curation_filter_node_from_dict(data: dict):

    node_type = data["type"]

    # ============================================================
    # Condition
    # ============================================================

    if node_type == "condition":

        threshold_data = data["threshold"]

        threshold_kwargs = {
            "value": float(threshold_data.get("value", 0.0)),
            "direction": threshold_data["direction"],
            "active": bool(
                threshold_data.get(
                    "active",
                    True,
                )
            ),
            "reference": reference_from_dict(threshold_data.get("reference")),
            "upper_reference": reference_from_dict(
                threshold_data.get("upper_reference")
            ),
        }
        if threshold_data.get("upper_value") is not None:
            threshold_kwargs["upper_value"] = float(threshold_data["upper_value"])
        elif threshold_kwargs["upper_reference"] is not None:
            # Resolved on evaluation.
            threshold_kwargs["upper_value"] = threshold_kwargs["value"]

        if threshold_data.get("axis") is not None:
            threshold_kwargs["axis"] = threshold_data["axis"]

        threshold = ThresholdSpec(**threshold_kwargs)
        threshold.validate()

        condition_kwargs = {
            "query": StatisticQuery.from_dict(data["query"]),
            "threshold": threshold,
        }

        if "id" in data:
            condition_kwargs["id"] = data["id"]

        return CurationFilterCondition(**condition_kwargs)

    # ============================================================
    # Group
    # ============================================================

    if node_type == "group":

        group_kwargs = {
            "operator": data.get(
                "operator",
                "and",
            ),
            "match_level": data.get(
                "match_level",
                "footprint",
            ),
            "children": [
                curation_filter_node_from_dict(child)
                for child in data.get("children", [])
            ],
            "count": (
                None if data.get("count") is None else CountSpec(**data["count"])
            ),
            "comment": data.get("comment", ""),
            "apply_to": data.get("apply_to", "same"),
            "summary": data.get("summary"),
            "manipulation": (
                None
                if data.get("manipulation") is None
                else ManipulationSpec(**data["manipulation"])
            ),
            "review_statuses": (
                None
                if data.get("review_statuses") is None
                else tuple(int(value) for value in data["review_statuses"])
            ),
        }

        if "id" in data:
            group_kwargs["id"] = data["id"]

        if group_kwargs["match_level"] == "footprint_pair":
            group_kwargs["match_level"] = "relationship"
        return CurationFilterGroup(**group_kwargs)

    raise ValueError("Unknown curation filter node type: " f"{node_type!r}")


def curation_filter_to_dict(
    root: CurationFilterGroup,
    *,
    name: str,
    category: str | None = None,
) -> dict:

    document = {
        "schema_version": CURATION_FILTER_SCHEMA_VERSION,
        "name": name,
    }

    # Curation stage (see curation_filter_presets.PRESET_CATEGORIES).
    if category is not None:
        document["category"] = category

    document["root"] = curation_filter_node_to_dict(root)
    return document


def curation_filter_from_dict(
    data: dict,
) -> CurationFilterGroup:

    version = int(
        data.get(
            "schema_version",
            1,
        )
    )

    if version not in (1, CURATION_FILTER_SCHEMA_VERSION):
        raise ValueError(f"Unsupported curation filter schema version {version}.")

    root = curation_filter_node_from_dict(data["root"])

    if not is_filter_group(root):
        raise ValueError("Curation filter root " "must be a group.")

    return root


def _active_ids(engine):
    """Assignment ids in which inactive sessions have no footprints."""
    from catan.gui.data.statistics.engine import active_session_view

    _, state = active_session_view(engine.data, engine.state)
    return np.asarray(state.assignments)
