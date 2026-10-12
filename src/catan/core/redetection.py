from dataclasses import replace

import numpy as np
from scipy import ndimage, sparse


def original_sampling_maps(remap, dims, ctx):
    """Original output pixel -> aligned input coordinate (y, x)."""
    height, width = map(int, dims)
    yy, xx = np.indices((height, width), dtype=np.float64)

    if remap is None or not remap.success:
        raise ValueError("Session alignment is unavailable. Align the session first.")

    matrix = np.asarray(remap.matrix, dtype=np.float64)

    if matrix.shape != (3, 3) or not np.isfinite(matrix).all():
        raise ValueError("Invalid session transformation.")

    qx = matrix[0, 0] * xx + matrix[0, 1] * yy + matrix[0, 2]
    qy = matrix[1, 0] * xx + matrix[1, 1] * yy + matrix[1, 2]

    x = qx.copy()
    y = qy.copy()

    flow = remap.flow

    if flow is not None:
        flow = np.asarray(flow, dtype=np.float64)

        if flow.shape != (height, width, 2) or not np.isfinite(flow).all():
            raise ValueError("Invalid optical-flow field.")

        # Existing forward sampling:
        #
        # original = inv(matrix) @ (aligned + flow(aligned))
        #
        # Consequently, invert the full deformation by solving:
        #
        # aligned + flow(aligned) = matrix @ original
        #
        # Merely negating flow or shift would be incorrect.
        for _ in range(100):
            ctx.check_cancelled()

            fx = ndimage.map_coordinates(
                flow[..., 0],
                [y, x],
                order=1,
                mode="nearest",
            )
            fy = ndimage.map_coordinates(
                flow[..., 1],
                [y, x],
                order=1,
                mode="nearest",
            )

            rx = x + fx - qx
            ry = y + fy - qy

            if np.max(np.hypot(rx, ry)) < 0.01:
                break

            x -= 0.5 * rx
            y -= 0.5 * ry

        fx = ndimage.map_coordinates(flow[..., 0], [y, x], order=1, mode="nearest")
        fy = ndimage.map_coordinates(flow[..., 1], [y, x], order=1, mode="nearest")

        residual = np.hypot(x + fx - qx, y + fy - qy)
        interior = (x >= 0) & (x <= width - 1) & (y >= 0) & (y <= height - 1)

        if np.any(interior & (residual > 0.05)):
            raise ValueError(
                "Could not invert optical flow reliably. "
                "Inspect the session alignment before exporting."
            )

    # Undo a transpose applied to footprints by the remap.
    # Background-only transposition must NOT be undone here.
    if remap.transpose:
        x = x.T
        y = y.T

    return y, x


def centroid(column, dims):
    """Centroid in pixels, independent of pxtomu."""
    column = sparse.csc_matrix(column)
    weights = column.data.astype(np.float64)

    if not np.isfinite(weights).all() or np.any(weights < 0) or weights.sum() <= 0:
        raise ValueError("A footprint has invalid or empty spatial weights.")

    rows, cols = np.divmod(column.indices, dims[1])

    return (
        np.array(
            [
                np.dot(rows, weights),
                np.dot(cols, weights),
            ]
        )
        / weights.sum()
    )


def infer_aligned_footprint(
    sessions,
    ids,
    union,
    session_id,
    neuron_id,
    *,
    ctx=None,
):
    """Return an inferred sparse footprint in aligned, C-order coordinates."""
    dims = tuple(sessions[session_id].dims)

    if ids.ndim != 2 or not 0 <= neuron_id < ids.shape[0]:
        raise ValueError("Invalid neuron ID for footprint inference.")

    if ids.shape[1] != len(sessions):
        raise ValueError("Assignments do not match the loaded sessions.")

    if ids[neuron_id, session_id] >= 0:
        return None

    if not bool(union.included[neuron_id]):
        return None

    observations = []

    for sid, other in enumerate(sessions):
        if ctx is not None:
            ctx.check_cancelled()

        footprint_id = int(ids[neuron_id, sid])

        if footprint_id < 0:
            continue

        if not other.status["spatial_loaded"]:
            raise ValueError(
                f"Load spatial data for {other.name} " "before inferring footprints."
            )

        if tuple(other.dims) != dims or footprint_id >= other.footprints.shape[1]:
            raise ValueError("Footprint dimensions or assignment IDs are inconsistent.")

        if other.included is not None and not other.included[footprint_id]:
            continue

        synthetic = getattr(other, "synthetic", None)
        if synthetic is not None and synthetic[footprint_id]:
            continue

        observations.append(
            (
                sid,
                centroid(other.footprints[:, footprint_id], dims),
            )
        )

    if not observations:
        return None

    previous = [item for item in observations if item[0] < session_id]
    following = [item for item in observations if item[0] > session_id]

    if previous and following:
        a, ca = previous[-1]
        b, cb = following[0]
        position = ca + (cb - ca) * ((session_id - a) / (b - a))
    else:
        position = (previous[-1] if previous else following[0])[1]

    column = sparse.csc_matrix(union.footprints[:, neuron_id])

    if column.shape != (int(np.prod(dims)), 1):
        raise ValueError("Union footprint does not match the session dimensions.")

    displacement = position - centroid(column, dims)

    # Translate nonzero pixels using bilinear interpolation.
    # This avoids allocating a full image for every nearby neuron.
    row, col = np.divmod(column.indices, dims[1])
    y = row + displacement[0]
    x = col + displacement[1]

    y0 = np.floor(y).astype(np.int64)
    x0 = np.floor(x).astype(np.int64)
    fy = y - y0
    fx = x - x0

    indices = []
    weights = []

    for dy, wy in ((0, 1 - fy), (1, fy)):
        for dx, wx in ((0, 1 - fx), (1, fx)):
            yy = y0 + dy
            xx = x0 + dx
            values = column.data * wy * wx

            keep = (
                (yy >= 0) & (yy < dims[0]) & (xx >= 0) & (xx < dims[1]) & (values > 0)
            )

            indices.append(yy[keep] * dims[1] + xx[keep])
            weights.append(values[keep])

    indices = np.concatenate(indices)
    weights = np.concatenate(weights).astype(np.float32)

    result = sparse.csc_matrix(
        (
            weights,
            (
                indices,
                np.zeros(len(indices), dtype=np.int64),
            ),
        ),
        shape=column.shape,
    )
    result.sum_duplicates()
    result.eliminate_zeros()

    return result if result.nnz else None


def load_c(session, ctx, read_fields):
    """Load C temporarily without changing the GUI session."""
    value = session.traces.get("C")

    if value is None:
        config = session.source_config
        fields = (
            {}
            if config is None
            else config.get_fields_to_load(enabled_only=False).get("traces", {})
        )

        spec = fields.get("C")

        # Native configurations may expose /traces/*.
        # Resolve that to C so we do not load every trace type.
        if spec is None and "*" in fields:
            wildcard = fields["*"]

            if wildcard.path.endswith("/*") and wildcard.path.count("*") == 1:
                spec = replace(
                    wildcard,
                    path=wildcard.path[:-1] + "C",
                )

        if spec is None:
            raise ValueError(
                "Configure the C trace field in this session's "
                "load configuration, then retry export."
            )

        ctx.message(f"Loading C: {session.name}")

        loaded = read_fields(
            session.path,
            {
                "traces": {
                    "C": replace(spec, required=True),
                }
            },
            ctx=ctx,
        )
        value = loaded.get("traces", {}).get("C")

    if value is None:
        raise ValueError("The configured C field was not loaded.")

    values = value.toarray() if sparse.issparse(value) else np.asarray(value)

    if values.ndim != 2 or values.shape[1] < 3:
        raise ValueError(
            "C must have shape (footprints, frames), " "with at least three frames."
        )

    return values


def noise_parameters(traces, session, ctx):
    baselines = []
    sigmas = []

    # Excluded original footprints can still provide real noise
    # measurements, particularly after replacing them by splits.
    for footprint in np.flatnonzero(~session.synthetic):
        ctx.check_cancelled()

        if footprint >= traces.shape[0]:
            raise ValueError(f"C has no row for detected footprint {footprint}.")

        row = np.asarray(traces[footprint], dtype=np.float64)

        if not np.isfinite(row).all():
            continue

        difference = np.diff(row)

        baselines.append(np.median(row))
        sigmas.append(
            np.median(np.abs(difference - np.median(difference)))
            / (0.67448975 * np.sqrt(2.0))
        )

    if not sigmas:
        raise ValueError(
            "No finite detected C traces are available " "to estimate seed noise."
        )

    return (
        float(np.median(baselines)),
        float(np.median(sigmas)),
    )


def build_seeds(
    sessions,
    ids,
    union,
    session_id,
    *,
    include_missing,
    seed,
    ctx,
    read_fields,
):
    session = sessions[session_id]
    dims = tuple(session.dims)

    if min(dims) < 2:
        raise ValueError(
            "Seed export requires two spatial dimensions " "of at least two pixels."
        )

    n_footprints = session.footprints.shape[1]
    owners = np.full(n_footprints, -1, dtype=np.int64)

    if ids.shape[1] > session_id:
        for neuron in np.flatnonzero(ids[:, session_id] >= 0):
            footprint = int(ids[neuron, session_id])

            if footprint >= n_footprints or owners[footprint] >= 0:
                raise ValueError(
                    "Assignments contain an invalid or " "duplicate footprint ID."
                )

            owners[footprint] = neuron

    # Present mode also works for included footprints that have
    # not yet received a tracked neuron identity.
    records = [
        (int(owners[footprint]), int(footprint))
        for footprint in np.flatnonzero(session.included)
    ]

    if include_missing:
        if ids.shape != (
            union.footprints.shape[1],
            len(sessions),
        ):
            raise ValueError(
                "Rebuild the active assignments union before "
                "exporting missing neurons."
            )

        records += [
            (int(neuron), -1)
            for neuron in np.flatnonzero(union.included)
            if ids[neuron, session_id] < 0
        ]

    if not records:
        raise ValueError("No included seeds are available for this session.")

    traces = load_c(session, ctx, read_fields)

    needs_noise = any(
        footprint < 0 or session.synthetic[footprint] for _, footprint in records
    )

    mu, sigma = noise_parameters(traces, session, ctx) if needs_noise else (0.0, 0.0)

    ymap, xmap = original_sampling_maps(session.remap, dims, ctx)
    native_dims = ymap.shape

    columns = []
    kept = []
    omitted = []

    for neuron, footprint in records:
        ctx.check_cancelled()

        if footprint >= 0:
            image = session.footprints[:, footprint].toarray().reshape(dims)

        else:
            column = infer_aligned_footprint(
                sessions,
                ids,
                union,
                session_id,
                neuron,
                ctx=ctx,
            )

            if column is None:
                omitted.append((neuron, footprint))
                continue

            image = column.toarray().reshape(dims)

        if not np.isfinite(image).all() or np.any(image < 0):
            raise ValueError(
                f"Footprint {footprint}, neuron {neuron}, " "contains invalid weights."
            )

        native = ndimage.map_coordinates(
            image.astype(np.float32),
            [ymap, xmap],
            order=1,
            mode="constant",
            cval=0,
            prefilter=False,
        )

        if not np.any(native > 0):
            omitted.append((neuron, footprint))
            continue

        columns.append(sparse.csc_matrix(native.reshape(-1, 1, order="F")))
        kept.append((neuron, footprint))

    if not kept:
        raise ValueError(
            "All seeds were empty or outside the session " "imaging window."
        )

    C = np.empty(
        (len(kept), traces.shape[1]),
        dtype=np.float32,
    )
    generated = np.zeros(len(kept), dtype=bool)

    for row_id, (neuron, footprint) in enumerate(kept):
        ctx.check_cancelled()

        generated[row_id] = footprint < 0 or session.synthetic[footprint]

        if generated[row_id]:
            # Stable for the same session/component, independent
            # of whether other sessions are exported alongside it.
            rng = np.random.default_rng(
                np.random.SeedSequence(
                    [
                        seed,
                        session_id,
                        neuron + 1,
                        footprint + 1,
                    ]
                )
            )
            C[row_id] = rng.normal(mu, sigma, C.shape[1])

        else:
            if footprint >= traces.shape[0] or not np.isfinite(traces[footprint]).all():
                raise ValueError(
                    "Missing or nonfinite C trace for detected "
                    f"footprint {footprint}. Repair its C mapping "
                    "and retry."
                )

            C[row_id] = traces[footprint]

    return {
        "A": sparse.hstack(columns, format="csc"),
        "C": C,
        "dims": np.asarray(native_dims, dtype=np.int64),
        "neuron_ids": np.asarray(
            [neuron for neuron, _ in kept],
            dtype=np.int64,
        ),
        "footprint_ids": np.asarray(
            [footprint for _, footprint in kept],
            dtype=np.int64,
        ),
        "inferred": np.asarray(
            [footprint < 0 for _, footprint in kept],
            dtype=np.uint8,
        ),
        "generated_C": generated.astype(np.uint8),
        "omitted_ids": np.asarray(omitted, dtype=np.int64).reshape(-1, 2),
        "noise_mean": np.asarray(mu),
        "noise_sigma": np.asarray(sigma),
    }
