from typing import Literal

from dataclasses import dataclass, replace
import numpy as np

from .dimensions import SESSION_DIMS, NEURON_DIMS, canonical_dim
from .queries import (
    ReductionSpec,
    StatisticQuery,
    allowed_error_methods,
    allowed_reduction_methods,
    normalize_session_series_reductions,
)
from .table import PickTable

ErrorMode = Literal["none", "std", "sem", "iqr"]


@dataclass
class SessionSeries:
    title: str
    table: PickTable | None
    session_dim: str
    session_ids: np.ndarray
    values: np.ndarray
    errors_low: np.ndarray | None = None
    errors_high: np.ndarray | None = None
    n: np.ndarray | None = None

    @property
    def has_errors(self) -> bool:
        return self.errors_low is not None and self.errors_high is not None

    def tooltip_value_for_index(
        self,
        index: int,
    ) -> str:

        value = self.values[index]

        text = f"{self.title}: " f"{value:.4g}"

        if self.has_errors:

            low = self.errors_low[index]
            high = self.errors_high[index]

            if np.isfinite(low) and np.isfinite(high):

                if np.isclose(low, high):
                    text += f" ± {low:.4g}"

                else:
                    text += f" +{high:.4g}" f" / -{low:.4g}"

        if self.n is not None and np.isfinite(self.n[index]):
            text += f"    n={int(self.n[index])}"

        return text


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

    # Follow the actually retained axis (including a user-selected j axis).
    keep_session_dim = next(
        (
            d
            for d in session_dims
            if reductions.get(d, ReductionSpec("keep")).method == "keep"
        ),
        session_dims[0],
    )

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
        parameters=query.parameters,
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
            if spec.method == "keep" or (
                spec.method == "single" and stat_def.key != "footprint_similarity"
            ):
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
    # Sparse tables omit unobserved NaN rows. Keep the full session domain
    # so missing sessions remain gaps rather than disappearing from a line.
    from .sparse_values import SparseStatisticArray

    if isinstance(table.stat, SparseStatisticArray):
        unique_sessions = np.asarray(table.stat.dimensions[session_dim].coords)
    else:
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
        if rows.size == 0:
            continue

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
        title=table.stat.title,
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


# ============================================================
# Selection subsets
# ============================================================


def make_selection_query(engine_query: StatisticQuery | None) -> StatisticQuery | None:
    """
    The session-series engine query with its neuron reductions undone, so
    that the rows of selected neurons can be reduced separately
    (selected_rows, pair_selection_queries). Other dimensions are reduced as before.
    """
    if engine_query is None:
        return None

    reductions = engine_query.reduction_dict()
    neuron_order = tuple(d for d in engine_query.reduction_order if d in NEURON_DIMS)

    if not neuron_order:
        return None

    for dim in neuron_order:
        reductions[dim] = ReductionSpec("keep")

    # The session-series context would turn kept neuron dims back into
    # reductions (normalize_footprint_pair_reductions).
    return replace(
        engine_query,
        reductions=tuple(sorted(reductions.items())),
        reduction_order=tuple(
            d for d in engine_query.reduction_order if d not in NEURON_DIMS
        ),
        context="generic",
    )


def neuron_reductions(
    engine_query: StatisticQuery,
) -> tuple[tuple[str, ReductionSpec], ...]:
    """Neuron reductions of a session-series engine query, in order."""
    reductions = engine_query.reduction_dict()

    return tuple(
        (dim, reductions[dim])
        for dim in engine_query.reduction_order
        if dim in NEURON_DIMS
    )


def uses_pair_selection(engine_query: StatisticQuery) -> bool:
    """
    Whether the series reduces over neuron pairs. Keeping both neuron
    dimensions would cost n² values per session, so such selections are
    evaluated per selected neuron (pair_selection_queries) instead of
    from one kept table.
    """
    neuron_order = [d for d in engine_query.reduction_order if d in NEURON_DIMS]

    if len(neuron_order) < 2:
        return False

    # Pairs restricted to the same neuron have one neuron per row.
    return not any(
        f.target == "neuron" and f.relation == "same" and f.collapse_same
        for f in engine_query.filters
    )


def pair_selection_queries(selection_query: StatisticQuery, neurons):
    """
    (fixed dim, neuron, query) evaluating the pair rows of each selected
    neuron. Several neurons: pairs within the selection; one neuron: all
    pairs involving it, in both orientations (as rows_matching_components).
    """
    reductions = selection_query.reduction_dict()
    orientations = ("neuron_i", "neuron_j") if len(neurons) == 1 else ("neuron_i",)

    for dim in orientations:
        for neuron in neurons:
            single = {**reductions, dim: ReductionSpec("single", index=int(neuron))}

            yield dim, int(neuron), replace(
                selection_query,
                reductions=tuple(sorted(single.items())),
            )


def pair_selection_rows(table: PickTable, fixed_dim: str, neuron: int, neurons):
    """Columns and values of one pair_selection_queries result."""
    columns = table.refs_for_rows(np.arange(table.n_rows), include_fixed=True)
    partner = columns["neuron_j" if fixed_dim == "neuron_i" else "neuron_i"]

    if len(neurons) > 1:
        keep = np.isin(partner, neurons)
    elif fixed_dim == "neuron_j":
        # The self pair was part of the first orientation.
        keep = partner != neuron
    else:
        keep = np.ones(table.n_rows, dtype=bool)

    return (
        {name: np.asarray(column)[keep] for name, column in columns.items()},
        np.asarray(table.values, dtype=float)[keep],
    )


def concatenate_rows(parts):
    """Join (columns, values) parts with the same columns."""
    if not parts:
        return {}, np.empty(0)

    names = parts[0][0].keys()

    return (
        {name: np.concatenate([columns[name] for columns, _ in parts]) for name in names},
        np.concatenate([values for _, values in parts]),
    )


def selected_rows(table: PickTable, components):
    """Columns and values of the rows matching the components."""
    rows = table.rows_matching_components(components)

    return (
        {name: np.asarray(table.refs[name][rows]) for name in table.refs},
        np.asarray(table.values[rows], dtype=float),
    )


def remaining_session_dim(table: PickTable) -> str | None:
    return next((d for d in SESSION_DIMS if d in table.refs), None)


def reduce_selected_rows(
    columns: dict[str, np.ndarray],
    values: np.ndarray,
    reductions: tuple[tuple[str, ReductionSpec], ...],
    *,
    session_dim: str,
    aliases: dict[str, str],
    session_ids: np.ndarray,
    title: str,
) -> SessionSeries | None:
    """
    Session series of selected rows, reduced over the neuron dimensions
    like the full series (StatisticArray.reduce_dimension per group).
    """
    if values.size == 0 or session_dim not in columns:
        return None

    errors_low = errors_high = n = None
    reduced = set()

    for dim, spec in reductions:
        target = _column_for_dim(aliases, columns, dim)

        if target is None or target in reduced:
            continue

        reduced.add(target)

        keys = [column for name, column in columns.items() if name != target]
        groups, first_rows = _group_rows(keys, values.size)
        n_groups = first_rows.size

        center = _grouped_center(values, groups, n_groups, spec.method)

        if spec.error_method != "none":
            errors_low, errors_high, n = _grouped_error(
                values, groups, n_groups, center, spec.error_method
            )
        elif errors_low is not None:
            # Errors of an earlier reduction are carried along, as in
            # StatisticArray._reduce_existing_errors.
            errors_low = _grouped_center(errors_low, groups, n_groups, spec.method)
            errors_high = _grouped_center(errors_high, groups, n_groups, spec.method)
            n = np.bincount(groups, weights=n, minlength=n_groups)

        values = center
        columns = {
            name: column[first_rows]
            for name, column in columns.items()
            if name != target
        }

    index = {int(sid): i for i, sid in enumerate(session_ids)}

    out_values = np.full(len(session_ids), np.nan)
    out_low = None if errors_low is None else np.full(len(session_ids), np.nan)
    out_high = None if errors_high is None else np.full(len(session_ids), np.nan)
    out_n = None if n is None else np.zeros(len(session_ids), dtype=int)

    for row, sid in enumerate(columns[session_dim]):
        i = index.get(int(sid))

        if i is None:
            continue

        out_values[i] = values[row]

        if out_low is not None:
            out_low[i] = errors_low[row]
            out_high[i] = errors_high[row]

        if out_n is not None:
            out_n[i] = int(n[row])

    return SessionSeries(
        title=title,
        table=None,
        session_dim=session_dim,
        session_ids=np.asarray(session_ids),
        values=out_values,
        errors_low=out_low,
        errors_high=out_high,
        n=out_n,
    )


def _column_for_dim(aliases, columns, dim: str) -> str | None:
    if dim in columns:
        return dim

    alias = aliases.get(dim)

    if alias in columns:
        return alias

    canonical = canonical_dim(dim)
    return canonical if canonical in columns else None


def _group_rows(keys, size):
    """Group id per row and the first row of each group."""
    if not keys:
        return np.zeros(size, dtype=np.intp), np.zeros(min(size, 1), dtype=np.intp)

    codes = []
    shape = []

    for key in keys:
        uniques, inverse = np.unique(key, return_inverse=True)
        codes.append(inverse.reshape(-1))
        shape.append(uniques.size)

    combined = np.ravel_multi_index(codes, shape)
    _, first_rows, groups = np.unique(
        combined, return_index=True, return_inverse=True
    )
    return groups.reshape(-1), first_rows


def _sorted_groups(values, groups, n_groups):
    """Finite values sorted by group and value, with group starts/counts."""
    finite = np.isfinite(values)
    values = values[finite]
    groups = groups[finite]

    order = np.lexsort((values, groups))
    counts = np.bincount(groups, minlength=n_groups)
    starts = np.cumsum(counts) - counts
    return values[order], groups[order], starts, counts


def _grouped_quantile(sorted_values, starts, counts, q):
    """np.percentile (linear interpolation) per group; NaN if empty."""
    out = np.full(counts.size, np.nan)
    present = counts > 0

    position = q * (counts[present] - 1)
    low = np.floor(position).astype(np.intp)
    high = np.minimum(low + 1, counts[present] - 1)
    fraction = position - low

    base = starts[present]
    out[present] = (
        sorted_values[base + low] * (1.0 - fraction)
        + sorted_values[base + high] * fraction
    )
    return out


def _grouped_center(values, groups, n_groups, method):
    """Grouped version of StatisticArray._compute_center."""
    values, sorted_groups, starts, counts = _sorted_groups(
        np.asarray(values, dtype=float), groups, n_groups
    )
    present = counts > 0

    if method in ("mean", "sum", "std"):
        total = np.bincount(sorted_groups, weights=values, minlength=n_groups)

        if method == "sum":
            return np.where(present, total, np.nan)

        mean = np.full(n_groups, np.nan)
        mean[present] = total[present] / counts[present]

        if method == "mean":
            return mean

        squares = np.bincount(
            sorted_groups,
            weights=(values - mean[sorted_groups]) ** 2,
            minlength=n_groups,
        )
        std = np.full(n_groups, np.nan)
        std[present] = np.sqrt(squares[present] / counts[present])
        return std

    if method == "median":
        return _grouped_quantile(values, starts, counts, 0.5)

    out = np.full(n_groups, np.nan)

    if method == "min":
        out[present] = values[starts[present]]
        return out

    if method == "max":
        out[present] = values[starts[present] + counts[present] - 1]
        return out

    raise ValueError(f"Unknown reduction method: {method!r}")


def _grouped_error(values, groups, n_groups, center, method):
    """Grouped version of StatisticArray._compute_error."""
    values, sorted_groups, starts, counts = _sorted_groups(
        np.asarray(values, dtype=float), groups, n_groups
    )

    if method in ("std", "sem"):
        total = np.bincount(sorted_groups, weights=values, minlength=n_groups)
        mean = total / np.maximum(counts, 1)
        squares = np.bincount(
            sorted_groups,
            weights=(values - mean[sorted_groups]) ** 2,
            minlength=n_groups,
        )

        error = np.sqrt(squares / np.maximum(counts - 1, 1))

        if method == "sem":
            error = error / np.sqrt(np.maximum(counts, 1))

        error = np.where(counts > 1, error, 0.0)
        return error, error, counts

    if method == "iqr":
        q25 = _grouped_quantile(values, starts, counts, 0.25)
        q75 = _grouped_quantile(values, starts, counts, 0.75)
        return center - q25, q75 - center, counts

    raise ValueError(f"Unknown error method: {method!r}")
