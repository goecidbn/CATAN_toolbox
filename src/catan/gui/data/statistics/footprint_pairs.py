"""Neighbour-only footprint comparisons with independent pair constraints."""

import numpy as np
from scipy.spatial import cKDTree

from collections import OrderedDict

from catan.core.image_correlation import (
    compact_positive_footprint,
    footprint_similarity_peaks,
)
from catan.gui.background_tasks.runtime import (
    current_task_context,
    progress_scope,
    progress_step,
    reports_progress,
)

from .dimensions import Dimension
from .sparse_values import SparseStatisticArray

from collections import Counter
from time import perf_counter

TRACE_FOOTPRINT_PAIRS = False


@reports_progress("Calculating footprint similarity")
def calculate_footprint_pairs(
    data,
    state,
    indexers=None,
    filters=(),
    neighborhood_thr=10,
    gamma=0.1,
):
    indexers = indexers or {}

    trace = TRACE_FOOTPRINT_PAIRS

    if trace:
        trace_started = perf_counter()
        trace_count = 0
        trace_score_evaluations = 0
        trace_neurons_i = set()
        trace_neurons_j = set()
        trace_sessions = Counter()
        trace_examples = []
        trace_score_seconds = 0.0
        trace_patch_examples = []
        trace_largest_patch = 0

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
    if nr not in ("all", "same", "different") or sr not in (
        "all",
        "same",
        "different",
        "with previous",
    ):
        raise ValueError("Unsupported pair relation")
    compact_n = nr == "same" and collapse.get("neuron", True)
    compact_s = sr in ("same", "with previous") and collapse.get("session", True)

    # Build only legal session pairs; this is at most S*S small integers.
    si_ids, sj_ids = requested["session_i"], requested["session_j"]
    if sr in ("same", "with previous"):
        offset = int(sr == "with previous")
        allowed_j = set(sj_ids.tolist())
        pairs = [
            (int(si), int(si) - offset)
            for si in si_ids
            if int(si) - offset in allowed_j
        ]
    else:
        pairs = [
            (int(si), int(sj))
            for si in si_ids
            for sj in sj_ids
            if sr != "different" or si != sj
        ]

    dimensions, aliases = {}, {}
    for target, compact in (("neuron", compact_n), ("session", compact_s)):
        left, right = target + "_i", target + "_j"
        if compact:
            coords = (
                np.intersect1d(requested[left], requested[right])
                if target == "neuron"
                else np.asarray([si for si, _ in pairs], dtype=int)
            )
            fixed = (left in indexers or right in indexers) and len(coords) == 1
            dimensions[target] = Dimension(
                name=target,
                coords=coords,
                mode="fixed" if fixed else "remaining",
                parameter=int(coords[0]) if fixed else None,
            )
            aliases.update({left: target, right: target})
        else:
            for dim in (left, right):
                dimensions[dim] = Dimension(
                    name=dim,
                    coords=requested[dim],
                    mode="fixed" if dim in indexers else "remaining",
                    parameter=indexers.get(dim),
                )
        # Keep explicitly chosen reference coordinates for picking/tooltips.
        if compact:
            for dim in (left, right):
                if dim in indexers:
                    dimensions[dim] = Dimension(
                        name=dim,
                        coords=requested[dim],
                        mode="fixed",
                        parameter=int(indexers[dim]),
                    )

    dims = tuple(d for d, info in dimensions.items() if info.mode == "remaining")
    position_maps = {}

    for d in dims:
        coords = np.asarray(dimensions[d].coords, dtype=np.int64)
        position_map = np.full(
            int(coords.max()) + 1 if len(coords) else 0, -1, dtype=np.int64
        )
        position_map[coords] = np.arange(len(coords))
        position_maps[d] = position_map

    ni_ids, nj_ids = requested["neuron_i"], requested["neuron_j"]
    if nr == "same":
        ni_ids = nj_ids = np.intersect1d(ni_ids, nj_ids)

    value_blocks, position_blocks = [], []

    patch_cache = OrderedDict()

    patch_cache_bytes = 0
    patch_cache_budget = 128 * 1024**2

    def prepared_patch(session_id, footprint_id):
        nonlocal patch_cache_bytes, trace_largest_patch

        key = (int(session_id), int(footprint_id))

        if key in patch_cache:
            patch_cache.move_to_end(key)
            return patch_cache[key]

        session = data.sessions[key[0]]
        patch = compact_positive_footprint(
            session.footprints[:, key[1]],
            tuple(session.dims),
        )
        if TRACE_FOOTPRINT_PAIRS and patch is not None:
            trace_largest_patch = max(trace_largest_patch, patch.size)

            if len(trace_patch_examples) < 8:
                trace_patch_examples.append(
                    (
                        key,  # (session, footprint)
                        patch.shape,
                        int(np.count_nonzero(patch)),
                    )
                )

        size = 0 if patch is None else patch.nbytes

        if size <= patch_cache_budget:
            while patch_cache and (
                patch_cache_bytes + size
                > patch_cache_budget  # or len(patch_cache) >= 512
            ):
                _, old = patch_cache.popitem(last=False)
                if old is not None:
                    patch_cache_bytes -= old.nbytes

            patch_cache[key] = patch
            patch_cache_bytes += size

        return patch

    # Keep opposite directions adjacent; retain both requested directions.
    pairs.sort(
        key=lambda pair: (
            min(pair),
            max(pair),
        )
    )

    score_cache = {}
    score_cache_sessions = None

    ctx = current_task_context()
    for progress_index, (step, (si, sj)) in enumerate(enumerate(pairs)):
        with progress_step(
            progress_index,
            len(pairs),
            start=0,
            stop=95,
            message=f"Similarity: sessions {si + 1} / {sj + 1}",
        ):

            session_key = (min(si, sj), max(si, sj))

            if session_key != score_cache_sessions:
                score_cache.clear()
                score_cache_sessions = session_key

            if ctx is not None:
                ctx.check_cancelled()
            source, reference = data.sessions[si], data.sessions[sj]
            if any(
                session is None
                or session.centroids is None
                or session.footprints is None
                for session in (source, reference)
            ):
                continue

            if tuple(source.dims) != tuple(reference.dims):
                raise ValueError(
                    "Footprint similarity requires the same aligned pixel grid; "
                    f"received {source.dims} and {reference.dims}."
                )

            fi = state.assignments[ni_ids, si]
            fj = state.assignments[nj_ids, sj]

            if nr == "same":
                if si == sj:  # Existing self-comparison policy.
                    continue
                valid = np.flatnonzero((fi >= 0) & (fj >= 0))
                delta = source.centroids[fi[valid]] - reference.centroids[fj[valid]]
                distance = np.linalg.norm(delta, axis=1)
                valid = valid[np.isfinite(distance) & (distance < neighborhood_thr)]
                cand_i = cand_j = valid.astype(np.int64)
            else:
                vi, vj = np.flatnonzero(fi >= 0), np.flatnonzero(fj >= 0)
                ci, cj = source.centroids[fi[vi]], reference.centroids[fj[vj]]
                good_i, good_j = np.all(np.isfinite(ci), axis=1), np.all(
                    np.isfinite(cj), axis=1
                )
                vi, ci, vj, cj = vi[good_i], ci[good_i], vj[good_j], cj[good_j]
                if not len(vj) or not len(vi):
                    continue

                # All neighbours in one call; exact distance check below.
                found = cKDTree(cj).query_ball_point(ci, neighborhood_thr)
                lengths = np.fromiter(
                    (len(f) for f in found), dtype=np.int64, count=len(found)
                )
                local_i = np.repeat(np.arange(len(vi)), lengths)
                local_j = (
                    np.concatenate(
                        [np.asarray(f, dtype=np.int64) for f in found if len(f)]
                    )
                    if lengths.sum()
                    else np.empty(0, dtype=np.int64)
                )
                order = np.lexsort((local_j, local_i))
                local_i, local_j = local_i[order], local_j[order]

                keep = (
                    np.linalg.norm(ci[local_i] - cj[local_j], axis=1) < neighborhood_thr
                )
                cand_i, cand_j = vi[local_i[keep]], vj[local_j[keep]]

                if nr == "different" or si == sj:
                    distinct = ni_ids[cand_i] != nj_ids[cand_j]
                    cand_i, cand_j = cand_i[distinct], cand_j[distinct]

            if not len(cand_i):
                continue

            if ctx is not None:
                ctx.check_cancelled()

            if TRACE_FOOTPRINT_PAIRS:
                trace_count += len(cand_i)
                trace_neurons_i.update(ni_ids[cand_i].tolist())
                trace_neurons_j.update(nj_ids[cand_j].tolist())
                trace_sessions[(int(si), int(sj))] += len(cand_i)
                score_started = perf_counter()

            # Symmetric cache: score each footprint pair once, in the
            # orientation of its first appearance.
            pair_keys = []
            pending = {}

            for left_fp, right_fp in zip(fi[cand_i].tolist(), fj[cand_j].tolist()):
                left_key = (int(si), left_fp)
                right_key = (int(sj), right_fp)
                pair_key = (
                    (left_key, right_key)
                    if left_key <= right_key
                    else (right_key, left_key)
                )
                pair_keys.append(pair_key)

                if pair_key not in score_cache and pair_key not in pending:
                    pending[pair_key] = (left_key, right_key)

            if pending:
                if trace:
                    trace_score_evaluations += len(pending)

                lefts = [prepared_patch(*left) for left, _ in pending.values()]
                rights = [prepared_patch(*right) for _, right in pending.values()]

                if ctx is not None:
                    ctx.check_cancelled()

                scores = footprint_similarity_peaks(lefts, rights, gamma=gamma)
                score_cache.update(zip(pending.keys(), scores.tolist()))

            similarity = np.fromiter(
                (score_cache[key] for key in pair_keys),
                dtype=float,
                count=len(pair_keys),
            )

            if TRACE_FOOTPRINT_PAIRS:
                trace_score_seconds += perf_counter() - score_started

            finite = np.isfinite(similarity)

            if not np.any(finite):
                continue

            refs = {
                "neuron_i": ni_ids[cand_i[finite]],
                "neuron_j": nj_ids[cand_j[finite]],
                "session_i": si,
                "session_j": sj,
                "neuron": ni_ids[cand_i[finite]],
                "session": si,
            }

            value_blocks.append(similarity[finite])
            position_blocks.append(
                np.column_stack(
                    [
                        np.broadcast_to(position_maps[d][refs[d]], (int(finite.sum()),))
                        for d in dims
                    ]
                ).astype(np.int64)
                if dims
                else np.empty((int(finite.sum()), 0), dtype=np.int64)
            )

    with progress_scope(95, 99, "Assembling similarity results"):
        stat = SparseStatisticArray(
            name="footprint_similarity",
            title="Footprint similarity",
            category="pair",
            values=np.concatenate(value_blocks) if value_blocks else np.empty(0),
            positions=(
                np.concatenate(position_blocks)
                if position_blocks
                else np.empty((0, len(dims)), dtype=np.int64)
            ),
            dimensions=dimensions,
            reduction_aliases=aliases,
            applied_pair_filters=tuple(f.target for f in filters),
            reference_aliases={
                original: (
                    compact,
                    -1 if original == "session_j" and sr == "with previous" else 0,
                )
                for original, compact in aliases.items()
            },
        )

    if TRACE_FOOTPRINT_PAIRS:

        def brief(values):
            values = sorted(values)
            suffix = " …" if len(values) > 12 else ""
            return f"{values[:12]}{suffix} ({len(values)} distinct)"

        elapsed_ms = 1000 * (perf_counter() - trace_started)

        print(
            "\n[similarity pairs]\n"
            f"  requested indexers: {indexers}\n"
            f"  relations: neuron={nr}, session={sr}\n"
            f"  ordered candidate pairs: {trace_count}\n"
            f"  numerical score evaluations: {trace_score_evaluations}\n"
            f"  symmetric score-cache hits: "
            f"{trace_count - trace_score_evaluations}\n"
            f"  neuron_i: {brief(trace_neurons_i)}\n"
            f"  neuron_j: {brief(trace_neurons_j)}\n"
            f"  session-pair counts, first 12: "
            f"{list(trace_sessions.items())[:12]}\n"
            f"  example pairs ((neuron, session, footprint), ...): "
            f"{trace_examples}\n"
            f"  scoring incl. patch lookup/preparation: "
            f"{1000 * trace_score_seconds:.3f} ms\n"
            f"  pair calculation total: {elapsed_ms:.3f} ms\n"
            f"  patch examples ((session, footprint), shape, positive pixels): "
            f"{trace_patch_examples}\n"
            f"  largest patch: {trace_largest_patch} pixels\n",
            flush=True,
        )

    stat.validate()

    if ctx is not None:
        ctx.check_cancelled()
        current_task_context().progress(100)

    return stat


@reports_progress("Calculating requested footprint similarity")
def footprint_similarity_for_pairs(
    data,
    state,
    pairs,
    *,
    neighborhood_thr=10,
    gamma=0.1,
):
    """Similarity of explicit (neuron_i, neuron_j, session_i, session_j) rows.

    Applies the candidate rules of calculate_footprint_pairs(): both
    footprints present, raw centroids closer than ``neighborhood_thr`` and
    no self-comparison. Other rows are NaN.
    """
    pairs = np.asarray(pairs, dtype=np.int64).reshape(-1, 4)
    values = np.full(len(pairs), np.nan, dtype=np.float64)

    if not len(pairs):
        return values

    ids = state.assignments
    ni, nj, si, sj = pairs.T

    valid = (si >= 0) & (sj >= 0) & ~((ni == nj) & (si == sj))
    rows = np.flatnonzero(valid)
    fi = np.full(len(pairs), -1, dtype=np.int64)
    fj = np.full(len(pairs), -1, dtype=np.int64)
    fi[rows] = ids[ni[rows], si[rows]]
    fj[rows] = ids[nj[rows], sj[rows]]
    valid &= (fi >= 0) & (fj >= 0)

    ctx = current_task_context()
    lefts, rights, targets = [], [], []

    for source_id, reference_id in np.unique(pairs[valid][:, 2:], axis=0):
        if ctx is not None:
            ctx.check_cancelled()

        source = data.sessions[source_id]
        reference = data.sessions[reference_id]

        if any(
            session is None or session.centroids is None or session.footprints is None
            for session in (source, reference)
        ):
            continue

        if tuple(source.dims) != tuple(reference.dims):
            raise ValueError(
                "Footprint similarity requires the same aligned pixel grid; "
                f"received {source.dims} and {reference.dims}."
            )

        rows = np.flatnonzero(valid & (si == source_id) & (sj == reference_id))
        distance = np.linalg.norm(
            source.centroids[fi[rows]] - reference.centroids[fj[rows]], axis=1
        )
        rows = rows[np.isfinite(distance) & (distance < neighborhood_thr)]

        for row in rows:
            lefts.append(
                compact_positive_footprint(
                    source.footprints[:, fi[row]], tuple(source.dims)
                )
            )
            rights.append(
                compact_positive_footprint(
                    reference.footprints[:, fj[row]], tuple(reference.dims)
                )
            )
            targets.append(row)

    if targets:
        values[targets] = footprint_similarity_peaks(lefts, rights, gamma=gamma)

    return values
