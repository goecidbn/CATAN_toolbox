from dataclasses import dataclass, field
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

FilterOperator = Literal["and", "or"]
FilterMatchLevel = Literal["neuron", "footprint"]


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


@dataclass(slots=True)
class CurationFilterGroup:
    node_type: ClassVar[Literal["group"]] = "group"

    operator: FilterOperator = "and"
    match_level: FilterMatchLevel = "footprint"

    children: "list[CurationFilterCondition | CurationFilterGroup]" = field(
        default_factory=list
    )

    id: str = field(default_factory=lambda: uuid4().hex)


@dataclass(slots=True)
class CurationFilterConditionResult:
    condition: CurationFilterCondition
    table: PickTable
    matching_rows: np.ndarray

    neurons: set[int]
    components: set[NeuronComponent]

    rows_by_neuron: dict[int, np.ndarray]
    rows_by_component: dict[NeuronComponent, np.ndarray]

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


FilterMatch = int | NeuronComponent


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

    def components_for_neuron(
        self,
        neuron_id: int,
    ) -> set[NeuronComponent]:

        neuron_id = int(neuron_id)

        if self.match_level == "neuron":
            return set(self.evidence_by_match.get(neuron_id, set()))

        if self.match_level == "footprint":

            components = set()

            for component in self.matches:

                if component.neuron_id != neuron_id:
                    continue

                components.update(self.evidence_by_match.get(component, set()))

            return components

        raise ValueError(self.match_level)


@dataclass(slots=True)
class CurationFilterResult:
    neurons: set[int]

    condition_results: dict[str, CurationFilterConditionResult]

    group_results: dict[str, CurationFilterGroupResult]

    root_result: CurationFilterGroupResult

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

        return result.components_for_neuron(neuron_id)

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


class CurationFilterEvaluator:

    def __init__(
        self,
        engine: StatisticEngine,
    ):
        self.engine = engine

    def evaluate(
        self,
        root: CurationFilterGroup,
    ) -> CurationFilterResult:

        condition_results = {}
        group_results = {}

        if invalid_groups := empty_filter_group_ids(root):
            raise CurationFilterError(
                "The curation filter contains "
                f"{len(invalid_groups)} empty group"
                f"{'s' if len(invalid_groups) != 1 else ''}."
            )

        root_result = self._evaluate_group(root, condition_results, group_results)

        if root_result.match_level == "neuron":

            neurons = {int(neuron_id) for neuron_id in root_result.matches}

        elif root_result.match_level == "footprint":

            neurons = {int(component.neuron_id) for component in root_result.matches}

        else:
            raise ValueError(root_result.match_level)

        return CurationFilterResult(
            neurons=neurons,
            condition_results=condition_results,
            group_results=group_results,
            root_result=root_result,
        )

    def _evaluate_condition(
        self,
        condition: CurationFilterCondition,
    ) -> CurationFilterConditionResult:

        if not condition.threshold.active:
            raise CurationFilterError(
                "Curation filter condition has no active threshold."
            )

        table = self.engine.evaluate_table(condition.query)

        mask = condition.threshold.mask(table.values)

        if mask is None:
            raise CurationFilterError(
                "Curation filter condition has no active threshold."
            )

        mask = np.asarray(mask, dtype=bool)

        if mask.shape != table.values.shape:
            raise RuntimeError(
                "Threshold mask shape does not match "
                f"PickTable values: "
                f"{mask.shape} != {table.values.shape}."
            )

        matching_rows = np.flatnonzero(mask)

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

    def _evaluate_group(
        self,
        group: CurationFilterGroup,
        condition_results: dict[
            str,
            CurationFilterConditionResult,
        ],
        group_results: dict[
            str,
            CurationFilterGroupResult,
        ],
    ) -> CurationFilterGroupResult:

        child_results = []

        for child in group.children:

            # -------------------------------------
            # Condition
            # -------------------------------------

            if is_filter_condition(child):

                try:
                    condition_result = self._evaluate_condition(child)

                    condition_results[child.id] = condition_result

                    child_result = self._condition_result_for_level(
                        condition_result, group.match_level
                    )

                except CurationFilterError as exc:
                    exc.attach_node(child.id, "condition")
                    raise

            # -------------------------------------
            # Nested group
            # -------------------------------------

            elif is_filter_group(child):

                try:
                    child_result = self._evaluate_group(
                        child, condition_results, group_results
                    )

                    child_result = self._project_group_result(
                        child_result, group.match_level
                    )

                except CurationFilterError as exc:
                    exc.attach_node(child.id, "group")
                    raise

            else:
                raise TypeError(f"Unknown curation filter node: " f"{type(child)!r}")

            child_results.append(child_result)

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

        evidence_by_match = {}

        for match in matches:

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
        )

        group_results[group.id] = result

        return result

    def _project_rows(
        self,
        table: PickTable,
        rows: np.ndarray,
    ):
        rows = np.asarray(rows, dtype=int)

        if rows.size == 0:
            return (
                set(),
                set(),
                {},
                {},
            )

        refs = table.refs_for_rows(
            rows,
            include_fixed=True,
        )

        rows_by_neuron = {}
        rows_by_component = {}

        for local_row, table_row in enumerate(rows):

            row_neurons = set()
            row_components = set()

            # ------------------------------------------
            # ordinary neuron dimension
            # ------------------------------------------

            if "neuron" in refs:

                neuron_id = int(refs["neuron"][local_row])

                row_neurons.add(neuron_id)

                if "session" in refs:
                    row_components.add(
                        NeuronComponent(
                            neuron_id=neuron_id,
                            session_id=int(refs["session"][local_row]),
                        )
                    )

            # ------------------------------------------
            # pair neuron dimensions
            # ------------------------------------------

            for suffix in ("i", "j"):

                neuron_key = f"neuron_{suffix}"

                if neuron_key not in refs:
                    continue

                neuron_id = int(refs[neuron_key][local_row])

                row_neurons.add(neuron_id)

                session_key = f"session_{suffix}"

                if session_key in refs:

                    session_id = int(refs[session_key][local_row])

                    row_components.add(
                        NeuronComponent(
                            neuron_id=neuron_id,
                            session_id=session_id,
                        )
                    )

                elif "session" in refs:

                    # Important for queries such as:
                    #
                    # min_n_j(distances(n_i, s))
                    #
                    # neuron_j was reduced, but neuron_i and the
                    # shared session dimension survive.
                    session_id = int(refs["session"][local_row])

                    row_components.add(
                        NeuronComponent(
                            neuron_id=neuron_id,
                            session_id=session_id,
                        )
                    )

            for neuron_id in row_neurons:
                rows_by_neuron.setdefault(neuron_id, []).append(int(table_row))

            for component in row_components:
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

    def _condition_result_for_level(
        self,
        result: CurationFilterConditionResult,
        match_level: FilterMatchLevel,
    ) -> CurationFilterGroupResult:

        if match_level == "neuron":

            matches = set(result.neurons)

            evidence_by_match = {
                neuron_id: result.components_for_neuron(neuron_id)
                for neuron_id in matches
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
                for component in matches
            }

        else:
            raise ValueError(match_level)

        return CurationFilterGroupResult(
            group_id=result.condition.id,
            match_level=match_level,
            matches=matches,
            evidence_by_match=evidence_by_match,
        )

    def _project_group_result(
        self,
        result: CurationFilterGroupResult,
        match_level: FilterMatchLevel,
    ) -> CurationFilterGroupResult:

        if result.match_level == match_level:
            return result

        # -----------------------------------------
        # footprint -> neuron
        # -----------------------------------------

        if result.match_level == "footprint" and match_level == "neuron":

            evidence_by_neuron = {}

            for component in result.matches:

                neuron_id = int(component.neuron_id)

                evidence_by_neuron.setdefault(neuron_id, set()).update(
                    result.evidence_by_match.get(component, set())
                )

            return CurationFilterGroupResult(
                group_id=result.group_id,
                match_level="neuron",
                matches=set(evidence_by_neuron),
                evidence_by_match=evidence_by_neuron,
            )

        # -----------------------------------------
        # neuron -> footprint cannot be reconstructed
        # -----------------------------------------

        if result.match_level == "neuron" and match_level == "footprint":
            raise CurationFilterError(
                "A neuron-level filter group cannot be "
                "combined inside a footprint-level group."
            )

        raise ValueError((result.match_level, match_level))


def components_for_table_rows(
    table: PickTable,
    rows,
) -> set[NeuronComponent]:

    rows = np.atleast_1d(np.asarray(rows, dtype=int))

    if rows.size == 0:
        return set()

    refs = table.refs_for_rows(
        rows,
        include_fixed=True,
    )

    components = set()

    for i in range(rows.size):

        # -----------------------------------------
        # Ordinary neuron
        # -----------------------------------------

        if "neuron" in refs:

            neuron_id = int(refs["neuron"][i])

            if "session" in refs:

                components.add(
                    NeuronComponent(
                        neuron_id=neuron_id,
                        session_id=int(refs["session"][i]),
                    )
                )

            else:

                for session_key in (
                    "session_i",
                    "session_j",
                ):

                    if session_key in refs:
                        components.add(
                            NeuronComponent(
                                neuron_id=neuron_id,
                                session_id=int(refs[session_key][i]),
                            )
                        )

        # -----------------------------------------
        # Pairwise neuron dimensions
        # -----------------------------------------

        for suffix in ("i", "j"):

            neuron_key = f"neuron_{suffix}"

            if neuron_key not in refs:
                continue

            neuron_id = int(refs[neuron_key][i])

            session_key = f"session_{suffix}"

            if session_key in refs:

                session_id = int(refs[session_key][i])

            elif "session" in refs:

                session_id = int(refs["session"][i])

            else:
                # Session was reduced away:
                # no concrete footprint can be reconstructed.
                continue

            components.add(
                NeuronComponent(
                    neuron_id=neuron_id,
                    session_id=session_id,
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


CURATION_FILTER_SCHEMA_VERSION = 1


def curation_filter_node_to_dict(
    node,
) -> dict:

    if is_filter_condition(node):

        threshold = node.threshold

        threshold_data = {
            "value": float(threshold.value),
            "direction": threshold.direction,
            "active": bool(threshold.active),
        }

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
        }

    raise TypeError("Unknown curation filter node: " f"{type(node)!r}")


def curation_filter_node_from_dict(
    data: dict,
):

    node_type = data["type"]

    # ============================================================
    # Condition
    # ============================================================

    if node_type == "condition":

        threshold_data = data["threshold"]

        threshold_kwargs = {
            "value": float(threshold_data["value"]),
            "direction": threshold_data["direction"],
            "active": bool(
                threshold_data.get(
                    "active",
                    True,
                )
            ),
        }

        if threshold_data.get("axis") is not None:
            threshold_kwargs["axis"] = threshold_data["axis"]

        condition_kwargs = {
            "query": StatisticQuery.from_dict(data["query"]),
            "threshold": ThresholdSpec(**threshold_kwargs),
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
        }

        if "id" in data:
            group_kwargs["id"] = data["id"]

        return CurationFilterGroup(**group_kwargs)

    raise ValueError("Unknown curation filter node type: " f"{node_type!r}")


def curation_filter_to_dict(
    root: CurationFilterGroup,
    *,
    name: str,
) -> dict:

    return {
        "schema_version": CURATION_FILTER_SCHEMA_VERSION,
        "name": name,
        "root": curation_filter_node_to_dict(root),
    }


def curation_filter_from_dict(
    data: dict,
) -> CurationFilterGroup:

    version = int(
        data.get(
            "schema_version",
            1,
        )
    )

    if version != CURATION_FILTER_SCHEMA_VERSION:
        raise ValueError("Unsupported curation filter " f"schema version {version}.")

    root = curation_filter_node_from_dict(data["root"])

    if not is_filter_group(root):
        raise ValueError("Curation filter root " "must be a group.")

    return root
