from typing import Literal

from dataclasses import dataclass
import numpy as np

from .dimensions import SESSION_DIMS, NEURON_DIMS
from .queries import (
    ReductionSpec,
    StatisticQuery,
    allowed_error_methods,
    allowed_reduction_methods,
    normalize_session_series_reductions,
)
from .prepared_data import PickTable

ErrorMode = Literal["none", "std", "sem", "iqr"]


@dataclass
class SessionSeries:
    title: str
    table: PickTable
    session_dim: str
    session_ids: np.ndarray
    values: np.ndarray
    errors_low: np.ndarray | None = None
    errors_high: np.ndarray | None = None
    n: np.ndarray | None = None

    @property
    def has_errors(self) -> bool:
        return self.errors_low is not None and self.errors_high is not None


@dataclass
class PlotData:
    first_series: SessionSeries
    second_series: SessionSeries | None = None
    correlation: float | None = None
    correlation_n: int = 0

    @property
    def is_dual(self) -> bool:
        return self.second_series is not None


def prepare_query(query: StatisticQuery, registry):
    engine_query = make_engine_query_for_session_series(
        query,
        registry,
    )

    if engine_query is None:
        return None

    validate_session_series_engine_query(
        engine_query,
        registry,
    )
    return engine_query


def make_engine_query_for_session_series(
    query: StatisticQuery,
    registry,
) -> StatisticQuery | None:

    if query is None or query.statistic_key == "none":
        return None

    stat_def = registry[query.statistic_key]

    session_dims = [dim for dim in stat_def.dims if dim in SESSION_DIMS]

    if not session_dims:
        raise ValueError(
            f"Statistic {query.statistic_key!r} cannot be used "
            "as a session series because it has no session dimension."
        )

    reductions = normalize_session_series_reductions(
        stat_def,
        query.reduction_dict(),
        query.filters,
    )

    # Used only for determining reduction order.
    if "session" in session_dims:
        keep_session_dim = "session"
    elif "session_i" in session_dims:
        keep_session_dim = "session_i"
    else:
        keep_session_dim = session_dims[0]

    reduction_order = _session_series_reduction_order(
        stat_def.dims,
        reductions,
        keep_session_dim=keep_session_dim,
    )

    return StatisticQuery(
        statistic_key=query.statistic_key,
        reductions=tuple(sorted(reductions.items())),
        reduction_order=reduction_order,
        filters=query.filters,
        context=query.context,
    )


def validate_session_series_engine_query(query: StatisticQuery, registry):
    stat_def = registry[query.statistic_key]
    reductions = query.reduction_dict()

    kept_sessions = [
        dim
        for dim in stat_def.dims
        if dim in SESSION_DIMS
        and reductions.get(dim, ReductionSpec("keep")).method == "keep"
    ]

    session_filter = next(
        (f for f in query.filters if f.target == "session"),
        None,
    )

    relation = session_filter.relation if session_filter is not None else "all"

    if relation in ("same", "with previous"):
        expected = {"session_i", "session_j"}

        if set(kept_sessions) != expected:
            raise ValueError(
                f"Session relation {relation!r} requires both "
                f"session dimensions to remain until filtering; "
                f"got {kept_sessions}."
            )

    else:
        if len(kept_sessions) != 1:
            raise ValueError(
                "Session series needs exactly one kept session dimension, "
                f"got {kept_sessions}."
            )

    for dim in stat_def.dims:
        spec = reductions.get(dim, ReductionSpec("keep"))

        if dim in NEURON_DIMS:
            if spec.method in ("keep", "single"):
                raise ValueError(
                    f"Session series cannot leave neuron dimension {dim!r} "
                    "unreduced."
                )

            if spec.error_method == "none":
                # You may allow this, but for shaded error bars probably warn.
                pass


def _session_series_reduction_order(
    dims: tuple[str, ...],
    reductions: dict[str, ReductionSpec],
    *,
    keep_session_dim: str,
) -> tuple[str, ...]:
    first_pass = []
    neuron_pass = []

    for dim in dims:
        spec = reductions.get(dim)

        if spec is None:
            continue

        if spec.method in ("keep", "single"):
            continue

        if dim == keep_session_dim:
            continue

        if dim in NEURON_DIMS:
            neuron_pass.append(dim)
        else:
            first_pass.append(dim)

    return tuple(first_pass + neuron_pass)


def validate_session_series_reductions(stat_def, reductions):
    kept_sessions = []
    error_dims = []

    for dim in stat_def.dims:
        spec = reductions.get(dim, ReductionSpec("mean"))

        if dim in SESSION_DIMS and spec.method == "keep":
            kept_sessions.append(dim)

        if dim in NEURON_DIMS and spec.error_method != "none":
            error_dims.append(dim)

    if len(kept_sessions) != 1:
        raise ValueError(
            "Session-series mode requires exactly one kept session dimension, "
            f"got {kept_sessions}."
        )

    if len(error_dims) > 1:
        raise ValueError(
            "Session-series mode currently supports error bars over only one "
            f"neuron dimension, got {error_dims}. For pairwise statistics, use "
            "a joint grouped reduction later."
        )


def _get_single_session_dim(table: PickTable) -> str:
    available = [dim for dim in SESSION_DIMS if dim in table.refs]

    if len(available) != 1:
        raise ValueError(
            "Session-series plot requires exactly one remaining session "
            f"dimension, got {available}. Table refs are {tuple(table.refs)}."
        )

    return available[0]


def build_session_series_from_table(table: PickTable) -> SessionSeries:
    session_dim = _get_single_session_dim(table)

    session_ids = np.asarray(table.refs[session_dim])
    values = np.asarray(table.values, dtype=float)

    # Usually the table should already have one row per session after
    # session-series reductions. But this makes the function robust.
    unique_sessions = np.asarray(sorted(np.unique(session_ids)))

    out_values = np.full(unique_sessions.shape, np.nan, dtype=float)
    out_err_low = None
    out_err_high = None
    out_n = None

    if table.has_errors:
        out_err_low = np.full(unique_sessions.shape, np.nan, dtype=float)
        out_err_high = np.full(unique_sessions.shape, np.nan, dtype=float)

    if table.n is not None:
        out_n = np.zeros(unique_sessions.shape, dtype=int)

    for i, sid in enumerate(unique_sessions):
        rows = np.flatnonzero(session_ids == sid)

        # Expected case: one row per session.
        if rows.size == 1:
            row = rows[0]
            out_values[i] = values[row]

            if table.has_errors:
                out_err_low[i] = table.errors_low[row]
                out_err_high[i] = table.errors_high[row]

            if table.n is not None:
                out_n[i] = table.n[row]

        # Fallback case: multiple rows per session.
        # This can happen if some dimension was not reduced.
        else:
            row_values = values[rows]
            out_values[i] = np.nanmean(row_values)

            if table.has_errors:
                # Conservative fallback: average existing error estimates.
                out_err_low[i] = np.nanmean(table.errors_low[rows])
                out_err_high[i] = np.nanmean(table.errors_high[rows])

            if table.n is not None:
                out_n[i] = int(np.nansum(table.n[rows]))
            else:
                out_n = None

    # label = getattr(table.stat, "title", "") or "statistic"

    return SessionSeries(
        title={"first": table.stat.title},
        table=table,
        session_dim=session_dim,
        session_ids=unique_sessions,
        values=out_values,
        errors_low=out_err_low,
        errors_high=out_err_high,
        n=out_n,
    )


def build_plot_data(
    *,
    first_table: PickTable,
    second_table: PickTable | None = None,
) -> PlotData:
    first_series = build_session_series_from_table(first_table)

    if second_table is None:
        return PlotData(first_series=first_series)

    second_series = build_session_series_from_table(second_table)

    r, n = session_series_correlation(first_series, second_series)

    return PlotData(
        first_series=first_series,
        second_series=second_series,
        correlation=r,
        correlation_n=n,
    )


def _align_session_series(a: SessionSeries, b: SessionSeries):
    a_map = {session_id: i for i, session_id in enumerate(a.session_ids)}
    b_map = {session_id: i for i, session_id in enumerate(b.session_ids)}

    common_sessions = np.asarray(sorted(set(a_map).intersection(b_map)))

    a_values = np.asarray(
        [a.values[a_map[session_id]] for session_id in common_sessions],
        dtype=float,
    )
    b_values = np.asarray(
        [b.values[b_map[session_id]] for session_id in common_sessions],
        dtype=float,
    )

    return common_sessions, a_values, b_values


def session_series_correlation(a: SessionSeries, b: SessionSeries):
    _, a_values, b_values = _align_session_series(a, b)

    mask = np.isfinite(a_values) & np.isfinite(b_values)
    n = int(np.count_nonzero(mask))

    if n < 2:
        return None, n

    r = float(np.corrcoef(a_values[mask], b_values[mask])[0, 1])
    return r, n
