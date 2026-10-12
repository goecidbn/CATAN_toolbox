from dataclasses import replace

from .queries import (
    StatisticQuery,
    ReductionSpec,
    normalize_neuron_bound_reductions,
    normalize_footprint_pair_reductions,
    normalize_component_bound_reductions,
    normalize_neuron_pair_bound_reductions,
    normalize_component_pair_bound_reductions,
)
from .dimensions import neuron_bound_dim, component_bound_dims


def prepare_neuron_table_query(
    query: StatisticQuery,
    registry,
) -> StatisticQuery | None:

    if query is None or query.statistic_key == "none":
        return None

    stat_def = registry[query.statistic_key]

    reductions = normalize_neuron_bound_reductions(
        stat_def,
        query.reduction_dict(),
        query.filters,
    )

    return StatisticQuery(
        statistic_key=query.statistic_key,
        reductions=tuple(sorted(reductions.items())),
        reduction_order=query.reduction_order,
        filters=query.filters,
        context="neuron_bound",
        parameters=query.parameters,
    )


def prepare_component_table_query(
    query: StatisticQuery,
    registry,
) -> StatisticQuery | None:

    if query is None:
        return None

    stat_def = registry[query.statistic_key]

    reductions = normalize_component_bound_reductions(
        stat_def,
        query.reduction_dict(),
        query.filters,
    )

    return StatisticQuery(
        statistic_key=query.statistic_key,
        reductions=tuple(sorted(reductions.items())),
        reduction_order=query.reduction_order,
        filters=query.filters,
        context="component_bound",
        parameters=query.parameters,
    )


def prepare_neuron_pair_table_query(
    query: StatisticQuery,
    registry,
) -> StatisticQuery | None:

    if query is None:
        return None

    stat_def = registry[query.statistic_key]

    reductions = normalize_neuron_pair_bound_reductions(
        stat_def,
        query.reduction_dict(),
    )

    return StatisticQuery(
        statistic_key=query.statistic_key,
        reductions=tuple(sorted(reductions.items())),
        reduction_order=query.reduction_order,
        filters=query.filters,
        context="neuron_pair_bound",
        parameters=query.parameters,
    )


def prepare_component_pair_table_query(
    query: StatisticQuery,
    registry,
) -> StatisticQuery | None:

    if query is None:
        return None

    stat_def = registry[query.statistic_key]

    reductions = normalize_component_pair_bound_reductions(
        stat_def,
        query.reduction_dict(),
    )

    return StatisticQuery(
        statistic_key=query.statistic_key,
        reductions=tuple(sorted(reductions.items())),
        reduction_order=query.reduction_order,
        filters=query.filters,
        context="component_pair_bound",
        parameters=query.parameters,
    )


def prepare_tooltip_query(query, registry, component, entity_mode):
    """
    Bind the displayed entity before calculation.

    Other axes retain their configured reductions and comparison domain.
    """
    if entity_mode == "neuron":
        prepared = prepare_neuron_table_query(query, registry)
    else:
        prepared = prepare_component_table_query(query, registry)

    if prepared is None:
        return None

    definition = registry[prepared.statistic_key]
    reductions = prepared.reduction_dict()

    if entity_mode == "neuron":
        neuron_dim = neuron_bound_dim(definition.dims)
        session_dim = None
    else:
        bound = component_bound_dims(definition.dims)

        if bound is None:
            raise ValueError("Statistic has no component binding.")

        neuron_dim, session_dim = bound

    reductions[neuron_dim] = ReductionSpec(
        "single",
        index=int(component.neuron_id),
    )

    # Bind the session only when the displayed statistic retains it.
    # A user-selected session reduction must remain unchanged.
    if (
        session_dim is not None
        and reductions.get(session_dim, ReductionSpec("keep")).method == "keep"
    ):
        if component.session_id is None:
            return None

        session_id = int(component.session_id)

        # Session zero has no predecessor. Do not let the general
        # normalizer turn this into the pair (1, 0).
        with_previous = any(
            item.target == "session"
            and item.relation == "with previous"
            and item.collapse_same
            for item in prepared.filters
        )

        if (
            definition.independent_pairs
            and session_dim == "session_i"
            and with_previous
            and session_id == 0
        ):
            return None

        reductions[session_dim] = ReductionSpec(
            "single",
            index=session_id,
        )

    if definition.independent_pairs:
        reductions = normalize_footprint_pair_reductions(
            definition,
            reductions,
            prepared.filters,
        )

    return replace(
        prepared,
        reductions=tuple(sorted(reductions.items())),
    )
