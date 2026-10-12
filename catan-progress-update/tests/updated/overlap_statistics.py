"""Overlap of registered footprints, without pairwise shift optimization."""

from collections import OrderedDict
from functools import lru_cache

import numpy as np
from scipy import sparse

from catan.gui.background_tasks.runtime import (
    current_task_context, progress_scope, progress_step, progress_iter, reports_progress,
)

from .blockwise import evaluate_blockwise
from .distance_statistics import pair_plan


@reports_progress("Calculating footprint overlap")
def overlap_query(
    data,
    state,
    query,
    *,
    weighted=False,
    containment=True,
    directional=False,
    normalize_by_target=False,
    support_threshold=0.1,
    select_values=None,
    requested_pairs=None,
):
    if not 0 <= support_threshold < 1:
        raise ValueError("Support threshold must be in [0, 1).")

    ids = np.array(state.assignments, dtype=int, copy=True)
    n, s = ids.shape

    names = (
        "neuron_i",
        "neuron_j",
        "session_i",
        "session_j",
    )

    dims, aliases, references, coord, allowed = pair_plan(names, n, s, query)

    ctx = current_task_context()

    included = np.ones(n, dtype=bool)
    union = data.assignments.union

    if union is not None and union.included is not None:
        included = np.array(
            union.included,
            dtype=bool,
            copy=True,
        )

        if included.shape != (n,):
            raise ValueError("Union inclusion mask does not match assignments.")

    @lru_cache(maxsize=16)
    @reports_progress("Preparing overlap supports")
    def prepare(sid, columns):
        session = data.sessions[sid]

        if (
            session is None
            or session.footprints is None
            or not session.status.get("spatial_loaded", False)
            or not session.status.get("aligned", False)
            or data.alignment_is_stale(sid)
        ):
            return None

        shape = tuple(session.dims)

        source = session.footprints

        if source.shape[0] != int(np.prod(shape)):
            raise ValueError(
                f"Session {sid}: footprint shape does not match dimensions."
            )

        columns = np.asarray(columns, dtype=np.int64)

        if np.any(columns < 0) or np.any(columns >= source.shape[1]):
            raise ValueError("Assignments reference missing footprints.")

        a = sparse.csc_matrix(
            source[:, columns],
            dtype=np.float32,
            copy=True,
        )

        a.sum_duplicates()
        a.sort_indices()

        keep = np.ones(a.shape[1], dtype=bool)

        if session.included is not None:
            session_included = np.asarray(session.included, dtype=bool)

            if session_included.shape != (source.shape[1],):
                raise ValueError(f"Session {sid}: invalid inclusion mask.")

            keep = session_included[columns]

        for col in progress_iter(range(a.shape[1]), start=10, stop=90):
            if col % 128 == 0 and ctx is not None:
                ctx.check_cancelled()

            values = a.data[a.indptr[col] : a.indptr[col + 1]]

            if not keep[col] or not np.all(np.isfinite(values)):
                values[:] = 0
                continue

            np.maximum(values, 0, out=values)
            peak = float(values.max()) if values.size else 0.0

            if peak > 0:
                values /= peak
                values[values < support_threshold] = 0

                if not weighted:
                    values[:] = values > 0

        with progress_scope(90, 100, "Measuring footprint support"):
            a.eliminate_zeros()

            support = a.copy()
            support.data.fill(1)

            mass = np.asarray(a.sum(axis=0)).ravel()

        return a, support, mass, shape

    if requested_pairs is not None:
        from .requested_rows import evaluate_requested_rows

        if any(
            spec.method not in ("keep", "single")
            for spec in query.reduction_dict().values()
        ):
            raise ValueError("Requested pairs require unreduced footprint identities.")

        pairs = np.asarray(requested_pairs, dtype=np.int64)

        if pairs.ndim != 2 or pairs.shape[1] != 4:
            raise ValueError("Requested pairs must have four coordinate columns.")

        @reports_progress("Calculating requested overlap")
        def read_rows(refs):
            ni, nj, si, sj = (refs[name] for name in names)

            out = np.full(len(ni), np.nan, dtype=np.float32)
            permitted = included[ni] & included[nj] & ~((ni == nj) & (si == sj))

            for pair_filter in query.filters:
                left, right = (ni, nj) if pair_filter.target == "neuron" else (si, sj)
                relation = pair_filter.relation

                if relation == "same":
                    permitted &= left == right
                elif relation == "different":
                    permitted &= left != right
                elif relation == "with previous" and pair_filter.target == "session":
                    permitted &= right == left - 1
                elif relation != "all":
                    raise ValueError(f"Unsupported pair filter: {pair_filter}")

            session_pairs = np.unique(
                np.column_stack((si, sj)),
                axis=0,
            )

            for progress_index, (source, target) in enumerate(session_pairs):
                with progress_step(
                    progress_index, len(session_pairs), start=0, stop=100,
                ):
                    rows = np.flatnonzero(permitted & (si == source) & (sj == target))

                    if not rows.size:
                        continue

                    if ctx is not None:
                        ctx.check_cancelled()

                    fi = ids[ni[rows], source]
                    fj = ids[nj[rows], target]

                    present = (fi >= 0) & (fj >= 0)
                    rows = rows[present]
                    fi = fi[present]
                    fj = fj[present]

                    if not rows.size:
                        continue

                    columns_i = np.unique(fi)
                    columns_j = np.unique(fj)

                    with progress_scope(0, 20):
                        left = prepare(
                            int(source),
                            tuple(int(value) for value in columns_i),
                        )
                    with progress_scope(20, 40):
                        right = prepare(
                            int(target),
                            tuple(int(value) for value in columns_j),
                        )

                    if left is None or right is None:
                        continue

                    a, _, ma, shape_a = left
                    b, _, mb, shape_b = right

                    if shape_a != shape_b:
                        raise ValueError("Overlap requires the same registered pixel grid.")

                    fi = np.searchsorted(columns_i, fi)
                    fj = np.searchsorted(columns_j, fj)

                    # Compare corresponding columns, not their Cartesian product.
                    # Binary supports and weighted footprints both use min().
                    with progress_scope(40, 90, "Intersecting requested footprints"):
                        intersection = np.asarray(
                            a[:, fi].minimum(b[:, fj]).sum(axis=0)
                        ).ravel()

                    if directional:
                        denominator = mb[fj] if normalize_by_target else ma[fi]
                    elif containment:
                        denominator = np.minimum(ma[fi], mb[fj])
                    else:
                        denominator = ma[fi] + mb[fj] - intersection

                    valid = (ma[fi] > 0) & (mb[fj] > 0) & (denominator > 0)

                    values = np.full(len(rows), np.nan, dtype=np.float32)
                    np.divide(
                        intersection,
                        denominator,
                        out=values,
                        where=valid,
                    )
                    out[rows] = np.clip(values, 0, 1)

            return out

        return evaluate_requested_rows(
            dims,
            read_rows,
            {name: pairs[:, index] for index, name in enumerate(names)},
            reference_aliases=references,
            reduction_aliases=aliases,
            applied_filters=tuple(f.target for f in query.filters),
            select_values=select_values,
        )

    # Cache physical-footprint pair matrices, rather than the full
    # tracked-neuron × tracked-neuron × session × session tensor.
    cache = OrderedDict()
    cache_bytes = 0
    cache_budget = 64 * 1024**2

    @reports_progress("Calculating overlap pairs")
    def pair_scores(si, sj, columns_i, columns_j):
        nonlocal cache_bytes

        # Both overlap normalizations are symmetric.
        if not directional and si > sj:
            result = pair_scores(sj, si, columns_j, columns_i)
            return None if result is None else result.T

        key = (si, sj, columns_i, columns_j)

        if key in cache:
            result = cache.pop(key)
            cache[key] = result
            return result

        with progress_scope(0, 15):
            left = prepare(si, columns_i)
        with progress_scope(15, 30):
            right = prepare(sj, columns_j)

        if left is None or right is None:
            return None

        a, sa, ma, shape_a = left
        b, sb, mb, shape_b = right

        if shape_a != shape_b:
            raise ValueError("Overlap requires the same registered pixel grid.")

        with progress_scope(30, 85, "Intersecting overlap supports"):
            if a.shape[1] * b.shape[1] <= 64:
                # Small queries: intersect sparse pixel indices directly.
                intersection = np.zeros(
                    (a.shape[1], b.shape[1]),
                    dtype=np.float32,
                )

                for i in progress_iter(range(a.shape[1]), every=1):
                    if ctx is not None:
                        ctx.check_cancelled()

                    ia, ib = a.indptr[i : i + 2]

                    for j in range(b.shape[1]):
                        ja, jb = b.indptr[j : j + 2]

                        shared, ai, bi = np.intersect1d(
                            a.indices[ia:ib],
                            b.indices[ja:jb],
                            assume_unique=True,
                            return_indices=True,
                        )

                        if weighted:
                            intersection[i, j] = np.minimum(
                                a.data[ia:ib][ai],
                                b.data[ja:jb][bi],
                            ).sum(dtype=np.float64)
                        else:
                            intersection[i, j] = shared.size

            else:
                # Larger queries retain the efficient sparse matrix approach.
                intersections = (sa.T @ sb).tocoo()

                if weighted:
                    intersection = np.zeros(
                        (a.shape[1], b.shape[1]),
                        dtype=np.float32,
                    )

                    for k, (i, j) in enumerate(progress_iter(zip(intersections.row, intersections.col), total=intersections.nnz, start=20)):
                        if k % 128 == 0 and ctx is not None:
                            ctx.check_cancelled()

                        ia, ib = a.indptr[i : i + 2]
                        ja, jb = b.indptr[j : j + 2]

                        _, ai, bi = np.intersect1d(
                            a.indices[ia:ib],
                            b.indices[ja:jb],
                            assume_unique=True,
                            return_indices=True,
                        )

                        intersection[i, j] = np.minimum(
                            a.data[ia:ib][ai],
                            b.data[ja:jb][bi],
                        ).sum(dtype=np.float64)

                else:
                    intersection = intersections.toarray()

        with progress_scope(85, 100, "Normalizing overlap scores"):
            if directional:
                # Rows: source footprints (neuron_i, session_i).
                # Columns: target footprints (neuron_j, session_j).
                denominator = mb[None, :] if normalize_by_target else ma[:, None]
            elif containment:
                denominator = np.minimum(ma[:, None], mb[None, :])
            else:
                denominator = ma[:, None] + mb[None, :] - intersection

            valid = (ma[:, None] > 0) & (mb[None, :] > 0) & (denominator > 0)

            scores = np.full(
                intersection.shape,
                np.nan,
                dtype=np.float32,
            )

            np.divide(
                intersection,
                denominator,
                out=scores,
                where=valid,
            )

            np.clip(scores, 0, 1, out=scores)

        if scores.nbytes <= cache_budget:
            while cache and cache_bytes + scores.nbytes > cache_budget:
                _, old = cache.popitem(last=False)
                cache_bytes -= old.nbytes

            cache[key] = scores
            cache_bytes += scores.nbytes

        return scores

    @reports_progress("Calculating overlap block")
    def read(selected):
        ni, nj, si, sj = np.broadcast_arrays(*(coord(name, selected) for name in names))

        out = np.full(
            ni.shape,
            np.nan,
            dtype=np.float32,
        )

        permitted = (
            allowed(selected) & included[ni] & included[nj] & ~((ni == nj) & (si == sj))
        )

        for progress_index, (source) in enumerate(np.unique(si)):
            with progress_step(
                progress_index, len(np.unique(si)), start=0, stop=100,
            ):
                for progress_index, (reference) in enumerate(np.unique(sj)):
                    with progress_step(
                        progress_index, len(np.unique(sj)), start=0, stop=100,
                    ):
                        if ctx is not None:
                            ctx.check_cancelled()

                        mask = permitted & (si == source) & (sj == reference)

                        if not np.any(mask):
                            continue

                        fi = ids[ni[mask], source]
                        fj = ids[nj[mask], reference]

                        present = (fi >= 0) & (fj >= 0)

                        values = np.full(
                            fi.shape,
                            np.nan,
                            dtype=np.float32,
                        )

                        if np.any(present):
                            columns_i = np.unique(fi[present])
                            columns_j = np.unique(fj[present])

                            with progress_scope(0, 95):
                                scores = pair_scores(
                                    int(source),
                                    int(reference),
                                    tuple(int(value) for value in columns_i),
                                    tuple(int(value) for value in columns_j),
                                )

                            if scores is not None:
                                local_i = np.searchsorted(columns_i, fi[present])
                                local_j = np.searchsorted(columns_j, fj[present])

                                values[present] = scores[local_i, local_j]

                        out[mask] = values

        return out

    return evaluate_blockwise(
        dims,
        read,
        query,
        aliases=aliases,
        reference_aliases=references,
        applied_filters=tuple(f.target for f in query.filters),
        select_values=select_values,
    )


overlap_query.supports_value_selection = True
overlap_query.supports_requested_pairs = True
