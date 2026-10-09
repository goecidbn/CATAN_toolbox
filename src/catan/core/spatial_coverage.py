import numpy as np
from scipy.ndimage import distance_transform_edt, map_coordinates


def session_valid_mask(session):
    if session is None:
        raise ValueError("A session is missing.")

    name = session.name or str(session.path)

    if not session.status.get("spatial_loaded", False):
        raise ValueError(
            f"{name}: load spatial data before calculating border distance."
        )

    remap = session.remap

    if not session.status.get("aligned", False) or remap is None or not remap.success:
        raise ValueError(
            f"{name}: apply a successful alignment before "
            "calculating border distance."
        )

    if tuple(remap.dims) != tuple(session.dims):
        raise ValueError(f"{name}: alignment and session dimensions disagree.")

    return remap.valid_mask(use_optical_flow=True)


def signed_border_distance(mask):
    mask = np.asarray(mask, dtype=bool)

    if mask.ndim != 2 or not mask.any():
        raise ValueError("The imaging areas have no shared valid pixels.")

    # Explicitly include the outside of the canvas as invalid.
    padded = np.pad(mask, 1, constant_values=False)

    inside = distance_transform_edt(padded)[1:-1, 1:-1]
    outside = distance_transform_edt(~padded)[1:-1, 1:-1]

    # Raster boundary lies halfway between adjacent pixel centres.
    result = np.where(mask, inside - 0.5, 0.5 - outside)

    return result.astype(np.float32)


def distances_at_centroids(session, distance_map):
    centroids = session.centroids

    if centroids is None:
        return None

    scale = float(session.params.get("pxtomu", 1.0))

    if not np.isfinite(scale) or scale <= 0:
        raise ValueError("pxtomu must be finite and positive.")

    # Stored centroids include pxtomu; the map is indexed in pixels.
    xy = np.asarray(centroids, dtype=float) / scale

    height, width = distance_map.shape

    valid = (
        np.isfinite(xy).all(axis=1)
        & (xy[:, 0] >= 0)
        & (xy[:, 0] <= width - 1)
        & (xy[:, 1] >= 0)
        & (xy[:, 1] <= height - 1)
    )

    result = np.full(len(xy), np.nan)

    result[valid] = map_coordinates(
        distance_map,
        [xy[valid, 1], xy[valid, 0]],
        order=1,
        mode="nearest",
        prefilter=False,
    )

    return result
