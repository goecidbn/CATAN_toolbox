"""Neighbour-only footprint comparisons with independent pair constraints."""
import numpy as np
from scipy.spatial import cKDTree

from catan.core.image_correlation import calculate_img_correlation
from catan.gui.background_tasks.runtime import current_task_context

from .dimensions import Dimension
from .sparse_values import SparseStatisticArray


def calculate_footprint_pairs(data, state, indexers=None, filters=(), neighborhood_thr=10):
    indexers = indexers or {}
    n, s = state.assignments.shape
    sizes = {"neuron_i": n, "neuron_j": n, "session_i": s, "session_j": s}
    requested = {}
    for dim, size in sizes.items():
        if dim in indexers:
            index = indexers[dim]
            if index is None or not 0 <= int(index) < size:
                raise ValueError(f"Invalid {dim} index {index!r}")
            requested[dim] = np.array([int(index)], dtype=int)
        else:
            requested[dim] = np.arange(size, dtype=int)

    relations = {f.target: f.relation for f in filters}
    collapse = {f.target: f.collapse_same for f in filters}
    nr = relations.get("neuron", "all")
    sr = relations.get("session", "all")
    if nr not in ("all", "same", "different") or sr not in ("all", "same", "different", "with previous"):
        raise ValueError("Unsupported pair relation")
    compact_n = nr == "same" and collapse.get("neuron", True)
    compact_s = sr in ("same", "with previous") and collapse.get("session", True)

    # Build only legal session pairs; this is at most S*S small integers.
    si_ids, sj_ids = requested["session_i"], requested["session_j"]
    if sr in ("same", "with previous"):
        offset = int(sr == "with previous")
        allowed_j = set(sj_ids.tolist())
        pairs = [(int(si), int(si) - offset) for si in si_ids if int(si) - offset in allowed_j]
    else:
        pairs = [(int(si), int(sj)) for si in si_ids for sj in sj_ids
                 if sr != "different" or si != sj]

    dimensions, aliases = {}, {}
    for target, compact in (("neuron", compact_n), ("session", compact_s)):
        left, right = target + "_i", target + "_j"
        if compact:
            coords = (np.intersect1d(requested[left], requested[right]) if target == "neuron"
                      else np.asarray([si for si, _ in pairs], dtype=int))
            fixed = (left in indexers or right in indexers) and len(coords) == 1
            dimensions[target] = Dimension(name=target, coords=coords,
                mode="fixed" if fixed else "remaining", parameter=int(coords[0]) if fixed else None)
            aliases.update({left: target, right: target})
        else:
            for dim in (left, right):
                dimensions[dim] = Dimension(name=dim, coords=requested[dim],
                    mode="fixed" if dim in indexers else "remaining", parameter=indexers.get(dim))
        # Keep explicitly chosen reference coordinates for picking/tooltips.
        if compact:
            for dim in (left, right):
                if dim in indexers:
                    dimensions[dim] = Dimension(name=dim, coords=requested[dim],
                        mode="fixed", parameter=int(indexers[dim]))

    dims = tuple(d for d, info in dimensions.items() if info.mode == "remaining")
    lookups = {d: {int(value): i for i, value in enumerate(dimensions[d].coords)} for d in dims}
    ni_ids, nj_ids = requested["neuron_i"], requested["neuron_j"]
    if nr == "same":
        ni_ids = nj_ids = np.intersect1d(ni_ids, nj_ids)

    value_blocks, position_blocks, values, positions = [], [], [], []

    def flush():
        if values:
            value_blocks.append(np.asarray(values, dtype=float))
            position_blocks.append(np.asarray(positions, dtype=np.int64).reshape(len(values), len(dims)))
            values.clear()
            positions.clear()

    ctx = current_task_context()
    for step, (si, sj) in enumerate(pairs):
        if ctx is not None:
            ctx.check_cancelled()
            ctx.progress(int(100 * step / max(1, len(pairs))))
        source, reference = data.sessions[si], data.sessions[sj]
        if any(session is None or session.centroids is None or session.footprints is None
               for session in (source, reference)):
            continue
        fi = state.assignments[ni_ids, si]
        fj = state.assignments[nj_ids, sj]

        if nr == "same":
            if si == sj:  # Existing self-comparison policy.
                continue
            valid = np.flatnonzero((fi >= 0) & (fj >= 0))
            delta = source.centroids[fi[valid]] - reference.centroids[fj[valid]]
            distance = np.linalg.norm(delta, axis=1)
            valid = valid[np.isfinite(distance) & (distance < neighborhood_thr)]
            candidates = ((int(k), int(k)) for k in valid)
        else:
            vi, vj = np.flatnonzero(fi >= 0), np.flatnonzero(fj >= 0)
            ci, cj = source.centroids[fi[vi]], reference.centroids[fj[vj]]
            good_i, good_j = np.all(np.isfinite(ci), axis=1), np.all(np.isfinite(cj), axis=1)
            vi, ci, vj, cj = vi[good_i], ci[good_i], vj[good_j], cj[good_j]
            if not len(vj):
                continue
            tree = cKDTree(cj)

            def nearby_pairs():
                for count, (i, center) in enumerate(zip(vi, ci)):
                    if count % 64 == 0 and ctx is not None:
                        ctx.check_cancelled()
                    for local_j in tree.query_ball_point(center, neighborhood_thr):
                        j = vj[local_j]
                        if (nr == "different" or si == sj) and ni_ids[i] == nj_ids[j]:
                            continue
                        if np.linalg.norm(center - cj[local_j]) < neighborhood_thr:
                            yield int(i), int(j)
            candidates = nearby_pairs()

        for count, (i, j) in enumerate(candidates):
            if count % 64 == 0 and ctx is not None:
                ctx.check_cancelled()
            similarity, _, _ = calculate_img_correlation(
                source.footprints[:, fi[i]], reference.footprints[:, fj[j]],
                crop=True, shift=True, mode="cosine_union", gamma=0.1,
                shift_optimized=True,
            )
            if not np.isfinite(similarity):
                continue
            refs = {"neuron_i": int(ni_ids[i]), "neuron_j": int(nj_ids[j]),
                    "session_i": si, "session_j": sj,
                    "neuron": int(ni_ids[i]), "session": si}
            positions.append(tuple(lookups[d][refs[d]] for d in dims))
            values.append(float(similarity))
            if len(values) == 4096:
                flush()
        flush()

    stat = SparseStatisticArray(
        name="footprint_similarity", title="Footprint similarity", category="pair",
        values=np.concatenate(value_blocks) if value_blocks else np.empty(0),
        positions=np.concatenate(position_blocks) if position_blocks else np.empty((0, len(dims)), dtype=np.int64),
        dimensions=dimensions, reduction_aliases=aliases,
        applied_pair_filters=tuple(f.target for f in filters),
        reference_aliases={
            original: (compact, -1 if original == "session_j" and sr == "with previous" else 0)
            for original, compact in aliases.items()
        },
    )
    stat.validate()
    return stat
