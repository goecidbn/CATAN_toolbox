from .queries import (
    StatisticQuery,
    normalize_neuron_bound_reductions,
    normalize_component_bound_reductions,
    normalize_neuron_pair_bound_reductions,
    normalize_component_pair_bound_reductions,
)


def prepare_neuron_table_query(
    query: StatisticQuery,
    registry,
) -> StatisticQuery | None:

    if query is None or query.statistic_key == "none":
        return None

    stat_def = registry[query.statistic_key]

    reductions = normalize_neuron_bound_reductions(stat_def, query.reduction_dict())

    return StatisticQuery(
        statistic_key=query.statistic_key,
        reductions=tuple(sorted(reductions.items())),
        reduction_order=query.reduction_order,
        filters=query.filters,
        context="neuron_bound",
    )


def prepare_component_table_query(
    query: StatisticQuery,
    registry,
) -> StatisticQuery | None:

    if query is None:
        return None

    stat_def = registry[query.statistic_key]

    reductions = normalize_component_bound_reductions(stat_def, query.reduction_dict())

    return StatisticQuery(
        statistic_key=query.statistic_key,
        reductions=tuple(sorted(reductions.items())),
        reduction_order=query.reduction_order,
        filters=query.filters,
        context="component_bound",
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
    )
