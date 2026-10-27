"""Refine session alignments from footprint positions.

The background alignment places a session in the common frame; footprints
(cell bodies) can still be offset by a small rigid amount, e.g. because
the background is dominated by neuropil at a slightly different depth, or
because chained references accumulate small errors ("stairs" across
sessions). Footprint centroids measure that offset directly: matched
centroids of the same cells in two sessions should coincide after
alignment.

Coordinates are (x, y) image pixels in the common frame. A correction is
a 3x3 rigid matrix C acting after the session's alignment (common ->
corrected common), like Remapping.matrix acts on the original image.
"""

from copy import deepcopy

import cv2
import numpy as np
from scipy.ndimage import map_coordinates
from scipy.spatial import cKDTree

DEFAULTS = {
    # Mutual nearest neighbours closer than this are considered the same cell.
    "match_distance": 4.0,
    # A correction needs at least this many matched cells (per reference);
    # sparse sessions (ID28: 43-91 cells, 20-40 % matched) need few.
    "min_matches": 10,
    # Larger corrections are not footprint refinements: rejected.
    "max_correction": 5.0,
    # Also estimate a small rotation.
    "rotation": True,
}


# ------------------------------------------------------------------ points
def sample_flow(flow, points):
    """Bilinear flow (dx, dy) at (x, y) points; edge values outside."""
    coordinates = [points[:, 1], points[:, 0]]
    return np.column_stack(
        [
            map_coordinates(flow[..., channel], coordinates, order=1, mode="nearest")
            for channel in (0, 1)
        ]
    )


def map_points(points, matrix, flow=None, iterations=3):
    """Original-image (x, y) points -> common frame under an alignment.

    The alignment samples the original image at inv(matrix) @ (p + flow(p))
    for output point p; forward by fixed-point iteration (flows are smooth
    and small)."""
    points = np.asarray(points, dtype=np.float64)
    matrix = np.asarray(matrix, dtype=np.float64)
    rigid = points @ matrix[:2, :2].T + matrix[:2, 2]
    if flow is None:
        return rigid
    moved = rigid
    for _ in range(iterations):
        moved = rigid - sample_flow(flow, moved)
    return moved


def move_points(points, current, current_flow, target, target_flow, iterations=3):
    """Common-frame points under one alignment -> under another."""
    points = np.asarray(points, dtype=np.float64)
    if current_flow is not None:
        points = points + sample_flow(current_flow, points)
    inverse = np.linalg.inv(np.asarray(current, dtype=np.float64))
    original = points @ inverse[:2, :2].T + inverse[:2, 2]
    return map_points(original, target, target_flow, iterations)


# --------------------------------------------------------------- matching
class PointSet:
    """Finite points with their k-d tree (built once, queried often)."""

    def __init__(self, points):
        self.points = np.asarray(points, dtype=np.float64).reshape(-1, 2)
        # Empty footprints have no centroid.
        self.ids = np.flatnonzero(np.isfinite(self.points).all(axis=1))
        self.finite = self.points[self.ids]
        self.tree = cKDTree(self.finite) if len(self.ids) else None

    def __len__(self):
        return len(self.ids)


def mutual_matches(moving, reference, max_distance):
    """Index pairs (i_moving, i_reference) of mutual nearest neighbours
    closer than ``max_distance`` (arrays or PointSets)."""
    moving = moving if isinstance(moving, PointSet) else PointSet(moving)
    reference = reference if isinstance(reference, PointSet) else PointSet(reference)
    if not len(moving) or not len(reference):
        return np.empty(0, dtype=int), np.empty(0, dtype=int)

    distance, forward = reference.tree.query(moving.finite)
    _, backward = moving.tree.query(reference.finite)

    i = np.arange(len(moving))
    keep = (distance <= max_distance) & (backward[forward] == i)
    return moving.ids[i[keep]], reference.ids[forward[keep]]


def rigid_matrix(angle_rad, translation, center):
    """Rotation by ``angle_rad`` about ``center`` followed by ``translation``."""
    c, s = np.cos(angle_rad), np.sin(angle_rad)
    R = np.array([[c, -s], [s, c]])
    matrix = np.eye(3)
    matrix[:2, :2] = R
    matrix[:2, 2] = np.asarray(translation) + np.asarray(center) - R @ np.asarray(center)
    return matrix


def robust_rigid_fit(source, target, center, *, rotation=True, iterations=5):
    """Rigid matrix mapping ``source`` onto ``target`` points, with Tukey
    weights (moved or mismatched cells lose influence)."""
    source = np.asarray(source, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)
    center = np.asarray(center, dtype=np.float64)
    weights = np.ones(len(source))
    matrix = np.eye(3)

    for _ in range(iterations):
        w = weights / weights.sum()
        mean_source = (w[:, None] * source).sum(axis=0)
        mean_target = (w[:, None] * target).sum(axis=0)

        angle = 0.0
        if rotation:
            a = source - mean_source
            b = target - mean_target
            cross = np.sum(w * (a[:, 0] * b[:, 1] - a[:, 1] * b[:, 0]))
            dot = np.sum(w * (a[:, 0] * b[:, 0] + a[:, 1] * b[:, 1]))
            angle = float(np.arctan2(cross, dot))

        c, s = np.cos(angle), np.sin(angle)
        R = np.array([[c, -s], [s, c]])
        matrix = np.eye(3)
        matrix[:2, :2] = R
        matrix[:2, 2] = mean_target - R @ mean_source

        difference = source @ R.T + matrix[:2, 2] - target
        residual = np.sqrt(difference[:, 0] ** 2 + difference[:, 1] ** 2)
        middle = len(residual) // 2
        scale = 1.4826 * np.partition(residual, middle)[middle] + 1e-6
        u = residual / (4.685 * scale)
        weights = np.where(u < 1, (1 - u**2) ** 2, 0.0)
        if weights.sum() <= 0:
            break

    return matrix


def correction_parameters(matrix, center):
    """(dx, dy, angle in degrees) of a correction about ``center``."""
    R = matrix[:2, :2]
    angle = float(np.arctan2(R[1, 0], R[0, 0]))
    translation = matrix[:2, 2] - (np.asarray(center) - R @ np.asarray(center))
    return float(translation[0]), float(translation[1]), float(np.degrees(angle))


# --------------------------------------------------------------- estimates
def pair_estimate(points, reference, center, settings):
    """Correction of ``points`` onto one reference session's points, or
    None if too few cells match (arrays or PointSets)."""
    i, j = mutual_matches(points, reference, settings["match_distance"])
    if len(i) < settings["min_matches"]:
        return None
    points = points.points if isinstance(points, PointSet) else np.asarray(points)
    reference = reference.points if isinstance(reference, PointSet) else np.asarray(reference)
    matrix = robust_rigid_fit(points[i], reference[j], center, rotation=settings["rotation"])
    before = float(np.median(np.linalg.norm(points[i] - reference[j], axis=1)))
    moved = points[i] @ matrix[:2, :2].T + matrix[:2, 2]
    after = float(np.median(np.linalg.norm(moved - reference[j], axis=1)))
    return {"matrix": matrix, "matches": int(len(i)), "before": before, "after": after}


def combine(estimates, center):
    """Median of the references' corrections (dx, dy, angle)."""
    parameters = np.array([correction_parameters(e["matrix"], center) for e in estimates])
    dx, dy, angle = np.median(parameters, axis=0)
    return rigid_matrix(np.radians(angle), (dx, dy), center), (float(dx), float(dy), float(angle))


def estimate_correction(points, references, center, settings=None, iterations=8, tolerance=0.02):
    """Correction for a session's ``points`` against reference sessions'
    points (list of arrays): match, fit, apply and match again (ICP; a
    single matching underestimates offsets of a few px). Returns a dict
    with "matrix", "parameters" (dx, dy, angle), "accepted", "reason",
    "references", "matches", median distances of matched cells."""
    settings = {**DEFAULTS, **(settings or {})}
    points = np.asarray(points, dtype=np.float64)
    references = [r if isinstance(r, PointSet) else PointSet(r) for r in references]
    references = [r for r in references if len(r)]

    def estimates_for(current):
        moving = PointSet(current)
        return [
            estimate
            for reference in references
            if (estimate := pair_estimate(moving, reference, center, settings)) is not None
        ]

    total = np.eye(3)
    current = points
    first = None
    for _ in range(iterations):
        estimates = estimates_for(current)
        if not estimates:
            break
        if first is None:
            first = estimates
        step, (dx, dy, angle) = combine(estimates, center)
        total = step @ total
        current = points @ total[:2, :2].T + total[:2, 2]
        corners = np.array(center) * np.array([[0, 0], [2, 0], [0, 2], [2, 2]])
        if np.abs(corners @ step[:2, :2].T + step[:2, 2] - corners).max() < tolerance:
            break

    if first is None:
        return {
            "matrix": np.eye(3),
            "parameters": (0.0, 0.0, 0.0),
            "accepted": False,
            "reason": f"fewer than {settings['min_matches']} matching cells",
            "references": 0,
        }

    final = estimates_for(current) or first
    parameters = correction_parameters(total, center)
    # Distances of the cells matched after the correction, without and with
    # it (the matched set grows with the correction: comparing against the
    # first matching would compare different cells).
    moved = PointSet(current)
    pairs = [mutual_matches(moved, reference, settings["match_distance"]) for reference in references]
    pairs = [(i, j, reference) for (i, j), reference in zip(pairs, references) if len(i) >= settings["min_matches"]]
    if pairs:
        before = float(np.median([np.median(np.linalg.norm(points[i] - r.points[j], axis=1)) for i, j, r in pairs]))
        after = float(np.median([np.median(np.linalg.norm(current[i] - r.points[j], axis=1)) for i, j, r in pairs]))
    else:
        before = float(np.median([e["before"] for e in first]))
        after = float(np.median([e["before"] for e in final]))
    size = float(np.hypot(parameters[0], parameters[1]))

    reason = None
    if size > settings["max_correction"]:
        reason = f"correction {size:.1f} px exceeds {settings['max_correction']:.0f} px"
    elif after >= before:
        reason = "no improvement of matched cell distances"

    return {
        "matrix": total,
        "parameters": parameters,
        "accepted": reason is None,
        "reason": reason,
        "references": len(final),
        "matches": int(np.median([e["matches"] for e in final])),
        "median_before": before,
        "median_after": after,
    }


def global_corrections(points_by_session, center, settings=None, passes=2):
    """Corrections of all sessions against all others: the relative offset
    of every pair of sessions (ICP, estimate_correction) and one robust
    least-squares solution of c_i - c_j = r_ij (dx, dy, rotation; small
    corrections add up), weighted by the matched cells, with the mean
    correction at zero (the sessions as a whole stay where the background
    put them). A second pass refines the remaining offsets.

    points_by_session: list of (n_i, 2) arrays (None: not refined).
    Returns a list of results (see estimate_correction; None if skipped).
    """
    settings = {**DEFAULTS, **(settings or {})}
    sessions = [i for i, p in enumerate(points_by_session) if p is not None and len(p)]
    current = {i: np.asarray(points_by_session[i], dtype=np.float64) for i in sessions}
    total = {i: np.eye(3) for i in sessions}
    pairs_found = {i: 0 for i in sessions}
    matches = {i: [] for i in sessions}

    for _ in range(passes):
        point_sets = {k: PointSet(current[k]) for k in sessions}
        rows = []
        for a_index, i in enumerate(sessions):
            for j in sessions[a_index + 1 :]:
                pair = estimate_correction(
                    current[i], [point_sets[j]], center, settings, iterations=4
                )
                if not pair["references"]:
                    continue
                # Moves i onto j: c_i - c_j = parameters.
                rows.append((i, j, np.array(pair["parameters"]), pair["matches"]))

        if not rows:
            break

        index = {k: n for n, k in enumerate(sessions)}
        A = np.zeros((len(rows) + 1, len(sessions)))
        b = np.zeros((len(rows) + 1, 3))
        weights = np.ones(len(rows) + 1)
        for n, (i, j, parameters, count) in enumerate(rows):
            A[n, index[i]], A[n, index[j]] = 1.0, -1.0
            b[n] = parameters
            weights[n] = np.sqrt(count)
        # Mean correction zero.
        A[-1] = 1.0
        weights[-1] = 1e3

        # Robust: pairs disagreeing with the solution lose weight.
        robust = np.ones(len(rows) + 1)
        for _ in range(5):
            w = (weights * robust)[:, None]
            solution, *_ = np.linalg.lstsq(A * w, b * w, rcond=None)
            residual = np.linalg.norm((A @ solution - b)[:-1, :2], axis=1)
            scale = 1.4826 * np.median(residual) + 1e-6
            u = residual / (4.685 * scale)
            robust[:-1] = np.where(u < 1, (1 - u**2) ** 2, 0.0)

        largest = 0.0
        for k in sessions:
            dx, dy, angle = solution[index[k]]
            matrix = rigid_matrix(np.radians(angle), (dx, dy), center)
            current[k] = current[k] @ matrix[:2, :2].T + matrix[:2, 2]
            total[k] = matrix @ total[k]
            largest = max(largest, float(np.hypot(dx, dy)))

        for i, j, _, count in rows:
            pairs_found[i] += 1
            pairs_found[j] += 1
            matches[i].append(count)
            matches[j].append(count)

        if largest < 0.02:
            break

    out = [None] * len(points_by_session)
    for k in sessions:
        dx, dy, angle = correction_parameters(total[k], center)
        size = float(np.hypot(dx, dy))
        reason = None
        if not pairs_found[k]:
            reason = f"fewer than {settings['min_matches']} matching cells with any session"
        elif size > settings["max_correction"]:
            reason = f"correction {size:.1f} px exceeds {settings['max_correction']:.0f} px"
        out[k] = {
            "matrix": total[k],
            "parameters": (dx, dy, angle),
            "accepted": reason is None,
            "reason": reason,
            "references": pairs_found[k],
            "matches": int(np.median(matches[k])) if matches[k] else 0,
        }
    return out


# ------------------------------------------------------------- application
def compose_correction(remap, correction):
    """A copy of ``remap`` whose result is moved by the rigid ``correction``
    (common -> corrected common): matrix C @ M; the flow is carried along,
    f'(p) = R f(C^-1 p)."""
    remap = deepcopy(remap)
    correction = np.asarray(correction, dtype=np.float64)
    remap.matrix = correction @ np.asarray(remap.matrix, dtype=np.float64)
    remap.shift, remap.rotation = remap._rigid_parameters(remap.dims, remap.matrix)
    remap.shift = np.asarray(remap.shift, dtype=float)

    flow = getattr(remap, "flow", None)
    if flow is not None:
        height, width = remap.dims
        yy, xx = np.indices((height, width), dtype=np.float32)
        inverse = np.linalg.inv(correction)
        source_x = (inverse[0, 0] * xx + inverse[0, 1] * yy + inverse[0, 2]).astype(np.float32)
        source_y = (inverse[1, 0] * xx + inverse[1, 1] * yy + inverse[1, 2]).astype(np.float32)
        moved = np.stack(
            [
                cv2.remap(
                    np.ascontiguousarray(flow[..., channel], dtype=np.float32),
                    source_x,
                    source_y,
                    cv2.INTER_LINEAR,
                    borderMode=cv2.BORDER_REPLICATE,
                )
                for channel in (0, 1)
            ],
            axis=-1,
        )
        R = correction[:2, :2]
        remap.flow = (moved @ R.T).astype(np.float32)
    return remap
