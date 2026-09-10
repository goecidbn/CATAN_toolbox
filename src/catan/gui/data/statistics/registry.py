import numpy as np
from functools import partial

from .definitions import CALCULATED_STATISTICS
from .types import StatisticDefinition
from .dimensions import (
    get_match_stat_dims,
    make_match_coord_getter,
)
from . import calculations


def make_session_stat_definition(
    name: str,
) -> StatisticDefinition:

    return StatisticDefinition(
        key=f"session:{name}",
        title=name,
        description=f"Loaded session statistic: {name}",
        category="session_loaded",
        dims=("neuron", "session"),
        getter=partial(
            calculations.get_quality_metric,
            key=name,
        ),
    )


def make_match_stat_definition(
    name: str,
    values: np.ndarray,
) -> StatisticDefinition:

    values = np.asarray(values)
    dims = get_match_stat_dims(name, values)

    return StatisticDefinition(
        key=f"match:{name}",
        title=name,
        description=f"Loaded match statistic: {name}",
        category="match_loaded",
        dims=dims,
        getter=partial(
            calculations.get_match_metric,
            key=name,
            dims=dims,
        ),
        coord_getter=make_match_coord_getter(
            dims,
            values.shape,
        ),
    )


def get_loaded_session_stat_names(data) -> list[str]:
    names = set()

    for session in data.sessions:
        if session is None:
            continue

        names.update(session.quality.keys())

    return sorted(names)


def build_statistics_registry(
    data,
    state,
) -> dict[str, StatisticDefinition]:

    registry = dict(CALCULATED_STATISTICS)

    for name in get_loaded_session_stat_names(data):
        definition = make_session_stat_definition(name)
        registry[definition.key] = definition

    assignments = getattr(
        data,
        "assignments",
        None,
    )

    if assignments is not None:
        for name, values in assignments.stats.items():
            definition = make_match_stat_definition(
                name,
                values,
            )
            registry[definition.key] = definition

    return registry
