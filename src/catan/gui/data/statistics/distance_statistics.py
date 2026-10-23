"""Distances in aligned pixel coordinates."""

import numpy as np

from catan.gui.background_tasks.runtime import (
    current_task_context,
    progress_step,
    reports_progress,
)

from .dimensions import Dimension
from .queries import ReductionSpec
from .blockwise import evaluate_blockwise
from .sparse_pairs import SparsePairCollector, only_kept_axes, selection_upper_bound
from .session_relations import (
    NEURON_RELATIONS,
    DetectionReference,
    is_order_relation,
    relation_mask,
)


def geometry(data, state):
    ids = np.array(state.assignments, dtype=int, copy=True)
    n, s = ids.shape

    positions = np.full((n, s, 2), np.nan, dtype=np.float32)
    available = np.zeros(s, dtype=bool)

    for sid, session in enumerate(data.sessions[:s]):
        if (
            session is None
            or session.centroids is None
            or not session.status.get("spatial_loaded", False)
            or not session.status.get("aligned", False)
            or data.alignment_is_stale(sid)
        ):
            continue

        scale = float(session.params.get("pxtomu", 1.0))
        if not np.isfinite(scale) or scale <= 0:
            raise ValueError("pxtomu must be finite and positive.")

        centers = np.asarray(session.centroids, dtype=np.float32) / scale

        present = ids[:, sid] >= 0
        fp = ids[present, sid]

        if np.any(fp >= len(centers)):
            raise ValueError(
                f"Session {sid}: assignments reference missing footprints."
            )

        keep = np.ones(len(centers), dtype=bool)

        if session.included is not None:
            keep = np.asarray(session.included, dtype=bool)

            if keep.shape != (len(centers),):
                raise ValueError(f"Session {sid}: inclusion mask has the wrong length.")

        rows = np.flatnonzero(present)
        rows = rows[keep[fp]]

        positions[rows, sid] = centers[ids[rows, sid]]
        available[sid] = True

    union = data.assignments.union

    if union is not None and union.included is not None:
        included = np.asarray(union.included, dtype=bool)

        if included.shape != (n,):
            raise ValueError("Union inclusion mask does not match assignments.")

        positions[~included] = np.nan

    return positions, available


def union_positions(data, n):
    union = data.assignments.union

    if union is None or union.centroids is None:
        raise ValueError("Build the union before calculating union-centroid distances.")

    scale = float(union.params.get("pxtomu", 1.0))

    if not np.isfinite(scale) or scale <= 0:
        raise ValueError("Union pxtomu must be finite and positive.")

    result = np.array(union.centroids, dtype=np.float32, copy=True) / scale

    if result.shape != (n, 2):
        raise ValueError("Union centroids do not match the current assignments.")

    if union.included is not None:
        keep = np.asarray(union.included, dtype=bool)

        if keep.shape != (n,):
            raise ValueError("Union inclusion mask does not match assignments.")

        result[~keep] = np.nan

    return result


def reference_neurons(names, coordinate):
    """Reference neuron of a pair (j endpoint, or the shared neuron)."""
    for name in ("neuron_j", "neuron"):
        if name in names:
            return coordinate(name)

    return None


def pair_plan(names, n, s, query, ids=None):
    """``ids`` (assignments) is needed for neuron-dependent session relations."""
    reductions = query.reduction_dict()
    requested, fixed = {}, {}

    for name in names:
        size = n if name.startswith("neuron") else s
        spec = reductions.get(name, ReductionSpec("keep"))

        if spec.method == "single":
            if spec.index is None or not 0 <= spec.index < size:
                raise ValueError(f"Invalid index for {name}: {spec.index}")

            fixed[name] = int(spec.index)
            requested[name] = np.array([spec.index], dtype=int)
        else:
            requested[name] = np.arange(size)

    filters = {f.target: f for f in query.filters}
    dimensions, aliases, references = {}, {}, {}

    for name in names:
        if name in aliases:
            continue

        target = name.removesuffix("_i")
        right = target + "_j"
        f = filters.get(target)

        linked = (
            name.endswith("_i")
            and right in names
            and f is not None
            and f.collapse_same
            and (
                f.relation == "same"
                or (target == "session" and f.relation == "with previous")
            )
        )

        if linked:
            offset = int(f.relation == "with previous")
            coords = np.intersect1d(
                requested[name],
                requested[right] + offset,
            )

            is_fixed = (name in fixed or right in fixed) and len(coords) == 1

            dimensions[target] = Dimension(
                name=target,
                coords=coords,
                mode="fixed" if is_fixed else "remaining",
                parameter=int(coords[0]) if is_fixed else None,
            )

            for original, delta in ((name, 0), (right, -offset)):
                aliases[original] = target
                references[original] = (target, delta)

                if original in fixed:
                    dimensions[original] = Dimension(
                        name=original,
                        coords=requested[original],
                        mode="fixed",
                        parameter=fixed[original],
                    )
        else:
            dimensions[name] = Dimension(
                name=name,
                coords=requested[name],
                mode="fixed" if name in fixed else "remaining",
                parameter=fixed.get(name),
            )

    base_dims = tuple(d for d, info in dimensions.items() if info.mode == "remaining")

    def coordinate(name, selected):
        logical, offset = references.get(name, (name, 0))
        info = dimensions[logical]

        if info.mode == "fixed":
            return np.asarray(info.parameter + offset)

        shape = [1] * len(base_dims)
        shape[base_dims.index(logical)] = len(selected[logical])

        return np.asarray(selected[logical]).reshape(shape) + offset

    detection = (
        DetectionReference(ids)
        if ids is not None
        and any(f.relation in NEURON_RELATIONS for f in filters.values())
        else None
    )

    def allowed(selected):
        result = True

        for target, f in filters.items():
            left, right = target + "_i", target + "_j"

            if left not in names or right not in names:
                continue

            a = coordinate(left, selected)
            b = coordinate(right, selected)

            if f.relation == "same":
                result = result & (a == b)
            elif f.relation == "different":
                result = result & (a != b)
            elif f.relation == "with previous" and target == "session":
                result = result & (b == a - 1)
            elif target == "session" and is_order_relation(f.relation):
                neurons = reference_neurons(
                    names, lambda name: coordinate(name, selected)
                )
                result = result & relation_mask(f.relation, a, b, neurons, detection)
            elif f.relation != "all":
                raise ValueError(f"Unsupported {target} relation: {f.relation}")

        return result

    return dimensions, aliases, references, coordinate, allowed


def distance_query(
    data,
    state,
    query,
    *,
    mode="cross",
    exclude_self=True,
    select_values=None,
    requested_pairs=None,
):
    modes = {
        "cross": ("neuron_i", "neuron_j", "session_i", "session_j"),
        "within": ("neuron_i", "neuron_j", "session"),
        "displacement": ("neuron", "session_i", "session_j"),
        "to_union": ("neuron_i", "neuron_j", "session"),
        "union": ("neuron_i", "neuron_j"),
    }

    names = modes[mode]
    n, s = state.assignments.shape

    positions = None if mode == "union" else geometry(data, state)[0]
    union = union_positions(data, n) if mode in ("union", "to_union") else None

    dims, aliases, references, coord, allowed = pair_plan(
        names, n, s, query, ids=state.assignments
    )
    detection = (
        DetectionReference(state.assignments)
        if any(f.relation in NEURON_RELATIONS for f in query.filters)
        else None
    )

    if requested_pairs is not None:
        from .requested_rows import evaluate_requested_rows

        if any(
            spec.method not in ("keep", "single")
            for spec in query.reduction_dict().values()
        ):
            raise ValueError("Requested pairs require unreduced identities.")

        # Columns:
        # source neuron, target neuron, source session, target session.
        #
        # -1 represents an explicitly declared neuron endpoint.
        pairs = np.asarray(requested_pairs, dtype=np.int64)

        if pairs.ndim != 2 or pairs.shape[1] != 4:
            raise ValueError("Requested pairs must have four coordinate columns.")

        ni, nj, si, sj = pairs.T

        # Respect shared coordinates and endpoint types.
        if mode == "within":
            pairs = pairs[si == sj]
        elif mode == "displacement":
            pairs = pairs[ni == nj]
        elif mode == "to_union":
            pairs = pairs[sj == -1]
        elif mode == "union":
            pairs = pairs[(si == -1) & (sj == -1)]

        columns = {
            "cross": (0, 1, 2, 3),
            "within": (0, 1, 2),
            "displacement": (0, 2, 3),
            "to_union": (0, 1, 2),
            "union": (0, 1),
        }[mode]

        def read_rows(refs):
            if mode == "displacement":
                ni = nj = refs["neuron"]
            else:
                ni = refs["neuron_i"]
                nj = refs["neuron_j"]

            if mode in ("cross", "displacement"):
                si = refs["session_i"]
                sj = refs["session_j"]

                a = positions[ni, si]
                b = positions[nj, sj]
                identical = (ni == nj) & (si == sj)

            elif mode == "within":
                sid = refs["session"]

                a = positions[ni, sid]
                b = positions[nj, sid]
                identical = ni == nj

            elif mode == "to_union":
                a = positions[ni, refs["session"]]
                b = union[nj]

                # A footprint versus its own union is meaningful.
                identical = False

            else:
                a = union[ni]
                b = union[nj]
                identical = ni == nj

            delta = a - b
            values = np.hypot(delta[:, 0], delta[:, 1])

            valid = np.ones(len(values), dtype=bool)

            if exclude_self:
                valid &= ~np.asarray(identical)

            for f in query.filters:
                left = f.target + "_i"
                right = f.target + "_j"

                if left not in refs or right not in refs:
                    continue

                a, b = refs[left], refs[right]

                if f.relation == "same":
                    valid &= a == b
                elif f.relation == "different":
                    valid &= a != b
                elif f.relation == "with previous" and f.target == "session":
                    valid &= b == a - 1
                elif f.target == "session" and is_order_relation(f.relation):
                    valid &= relation_mask(f.relation, a, b, nj, detection)
                elif f.relation != "all":
                    raise ValueError(f"Unsupported pair filter: {f}")

            return np.where(valid, values, np.nan)

        return evaluate_requested_rows(
            dims,
            read_rows,
            {name: pairs[:, column] for name, column in zip(names, columns)},
            reference_aliases=references,
            reduction_aliases=aliases,
            applied_filters=tuple(f.target for f in query.filters),
            select_values=select_values,
        )

    if (
        select_values is not None
        and mode in ("cross", "within", "to_union", "union")
        and only_kept_axes(query)
    ):
        result = _sparse_distance_selection(
            mode,
            names,
            n,
            s,
            positions,
            union,
            dims,
            aliases,
            references,
            query,
            select_values,
            exclude_self=exclude_self,
            ids=state.assignments,
        )

        if result is not None:
            return result

    def read(selected):
        if mode == "displacement":
            ni = nj = coord("neuron", selected)
        else:
            ni = coord("neuron_i", selected)
            nj = coord("neuron_j", selected)

        if mode in ("cross", "displacement"):
            si = coord("session_i", selected)
            sj = coord("session_j", selected)
            a, b = positions[ni, si], positions[nj, sj]
            identical = (ni == nj) & (si == sj)

        elif mode == "within":
            sid = coord("session", selected)
            a, b = positions[ni, sid], positions[nj, sid]
            identical = ni == nj

        elif mode == "to_union":
            a = positions[ni, coord("session", selected)]
            b = union[nj]

            # Own footprint versus own union is meaningful.
            identical = False

        else:
            a, b = union[ni], union[nj]
            identical = ni == nj

        delta = a - b
        result = np.hypot(delta[..., 0], delta[..., 1])

        valid = allowed(selected)
        if exclude_self:
            valid = valid & ~np.asarray(identical)

        return np.where(valid, result, np.nan)

    return evaluate_blockwise(
        dims,
        read,
        query,
        aliases=aliases,
        reference_aliases=references,
        applied_filters=tuple(f.target for f in query.filters),
        select_values=select_values,
    )


@reports_progress("Finding nearby footprint pairs")
def _sparse_distance_selection(
    mode,
    names,
    n,
    s,
    positions,
    union,
    dims,
    aliases,
    references,
    query,
    select_values,
    *,
    exclude_self,
    ids=None,
):
    """Selected distances, enumerating only pairs within the selection radius.

    Returns None if the selection has no finite upper bound, in which case
    the blockwise path is used.
    """
    from scipy.spatial import cKDTree

    collector = SparsePairCollector(
        names, dims, aliases, references, n, s, query, select_values, ids=ids
    )

    clouds = [p.reshape(-1, 2) for p in (positions, union) if p is not None]
    points = np.concatenate(clouds, axis=0)
    points = points[np.all(np.isfinite(points), axis=1)]

    if not len(points):
        return collector.result()

    extent = np.ptp(points, axis=0)
    radius = selection_upper_bound(select_values, float(np.hypot(*extent)) + 1.0)

    if radius is None:
        return None

    if radius == 0.0:
        return collector.result()

    # Candidates only; values are recomputed exactly as in the grid path.
    search = radius * (1 + 1e-5) + 1e-4

    ctx = current_task_context()
    source_rows = collector.mask("neuron_i")
    target_rows = collector.mask("neuron_j")
    trees = {}

    def cloud(key, rows_mask):
        if key not in trees:
            coords = union if key[0] == "union" else positions[:, key[1]]
            rows = np.flatnonzero(rows_mask & np.all(np.isfinite(coords), axis=1))
            trees[key] = (rows, cKDTree(coords[rows]) if len(rows) else None)

        return trees[key]

    def nearby(left, right):
        rows_a, tree_a = left
        rows_b, tree_b = right

        if tree_a is None or tree_b is None:
            return None

        found = tree_a.sparse_distance_matrix(tree_b, search, output_type="ndarray")

        if not len(found):
            return None

        return rows_a[found["i"]], rows_b[found["j"]]

    def emit(physical, a, b, identical):
        delta = a - b
        values = np.hypot(delta[..., 0], delta[..., 1])

        if exclude_self:
            values = np.where(identical, np.nan, values)

        collector.emit(physical, values)

    if mode in ("cross", "within"):
        session_names = ("session_i", "session_j") if mode == "cross" else ("session",)
        sources = set(collector.domain(session_names[0]).tolist())
        targets = set(collector.domain(session_names[-1]).tolist())

        if mode == "cross":

            def allowed(si, sj):
                return (
                    si in sources
                    and sj in targets
                    and bool(collector.related("session", si, sj))
                )

            pairs = sorted(
                {
                    (min(i, j), max(i, j))
                    for i in sources
                    for j in targets
                    if allowed(i, j)
                }
            )
        else:
            pairs = [(sid, sid) for sid in sorted(sources)]

            def allowed(si, sj):
                return si == sj

        both = source_rows | target_rows

        for index, (a, b) in enumerate(pairs):
            with progress_step(index, len(pairs)):
                if ctx is not None and index % 16 == 0:
                    ctx.check_cancelled()

                found = nearby(cloud(("session", a), both), cloud(("session", b), both))

                if found is None:
                    continue

                rows_a, rows_b = found

                directions = [(a, b, rows_a, rows_b)]

                if a != b:
                    directions.append((b, a, rows_b, rows_a))

                for si, sj, ni, nj in directions:

                    if not allowed(si, sj):
                        continue

                    keep = source_rows[ni] & target_rows[nj]
                    ni, nj = ni[keep], nj[keep]

                    if mode == "cross":
                        physical = {
                            "neuron_i": ni,
                            "neuron_j": nj,
                            "session_i": si,
                            "session_j": sj,
                        }
                        identical = (ni == nj) & (si == sj)
                    else:
                        physical = {"neuron_i": ni, "neuron_j": nj, "session": si}
                        identical = ni == nj

                    emit(physical, positions[ni, si], positions[nj, sj], identical)

    elif mode == "to_union":
        sessions = collector.domain("session")
        union_cloud = cloud(("union", None), target_rows)

        for index, sid in enumerate(sessions):
            with progress_step(index, len(sessions)):
                if ctx is not None and index % 16 == 0:
                    ctx.check_cancelled()

                found = nearby(cloud(("session", int(sid)), source_rows), union_cloud)

                if found is None:
                    continue

                ni, nj = found
                emit(
                    {"neuron_i": ni, "neuron_j": nj, "session": int(sid)},
                    positions[ni, sid],
                    union[nj],
                    False,
                )

    else:
        found = nearby(
            cloud(("union", "i"), source_rows),
            cloud(("union", "j"), target_rows),
        )

        if found is not None:
            ni, nj = found
            emit({"neuron_i": ni, "neuron_j": nj}, union[ni], union[nj], ni == nj)

    return collector.result()


distance_query.supports_value_selection = True
distance_query.supports_requested_pairs = True
