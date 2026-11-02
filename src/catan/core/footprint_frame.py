"""A session's background relative to its own footprints (original frame).

The background places a session in the common frame (its alignment), the
footprints follow it through the same remap. Background and footprints of
a session can disagree (e.g. a correlation image from another motion
correction run). Per session:

- background correction (``session.params["background_correction"]``):
  a rigid transform (shift, rotation) and optional flow field that brings
  the background into the footprints' frame before it is aligned. The
  footprints are never moved (no resampling). Estimated on loading when
  background and footprints agree up to a shift / rotation
  (``estimate_correction``, source "auto"), or edited in the alignment
  editor's single-session view (source "manual");
- alignment source: the session is aligned by its (corrected) background
  or by the projection of its footprints (``session.params["alignment_source"]``;
  unset = footprints only when the background does not resemble them at
  all, ``background_coherence``).
"""

import numpy as np
from scipy import ndimage, sparse

ALIGNMENT_SOURCES = ("background", "footprints")

# Background vs. own footprint projection (high-passed, cross-correlation
# within COHERENCE_MAX_SHIFT px). ID29, CaImAn Cn vs. its footprints, 11
# sessions: peak at zero shift, robust z 9.7–19.9 (correlation 0.25–0.69);
# unrelated images (transposed / flipped background): z ≤ 5.1
# (correlation ≤ 0.09). Matched sessions peak at ≤ 1 px. ID20: five
# sessions whose Cn is offset by 3.6–14.5 px (pure translations), z 12.8–24.1.
COHERENCE_MAX_SHIFT = 30
COHERENCE_MIN_ZSCORE = 7.0
COHERENCE_MAX_OFFSET = 2.0

# Automatic correction: offsets below this are left alone (estimation noise
# of a matching background, ID29 ≤ 1 px); rotations searched within
# ±CORRECTION_MAX_ROTATION and kept only if they raise the peak correlation
# by CORRECTION_MIN_ROTATION_GAIN (relative).
CORRECTION_MIN_OFFSET = 1.5
CORRECTION_MAX_ROTATION = 2.0
CORRECTION_ROTATION_STEP = 0.5
CORRECTION_ROTATION_REFINE = 0.1
CORRECTION_MIN_ROTATION_GAIN = 0.02


def correction_of(params):
    """The session's background correction as a dict {"shift": (dx, dy) px,
    "rotation": degrees, "flow": (h, w, 2) array or None, "source": "auto" |
    "manual"}, or None (no correction).

    Older sessions / recipes stored a footprint offset (dx, dy) that moved
    the footprints onto the background; the same match now moves the
    background by (-dx, -dy)."""
    params = params or {}
    value = params.get("background_correction")
    if isinstance(value, dict):
        shift = np.asarray(value.get("shift", (0.0, 0.0)), dtype=float).reshape(-1)
        rotation = float(value.get("rotation", 0.0) or 0.0)
        flow = value.get("flow")
        flow = None if flow is None else np.asarray(flow, dtype=np.float32)
        if shift.shape != (2,) or not np.isfinite(shift).all() or not np.isfinite(rotation):
            return None
        if flow is not None and (flow.ndim != 3 or flow.shape[2] != 2 or not np.isfinite(flow).all()):
            flow = None
        source = value.get("source", "manual")
        # A manual zero is kept: "no correction" chosen by the user (not
        # estimated again on loading).
        if not np.any(shift) and rotation == 0.0 and flow is None and source == "auto":
            return None
        return {
            "shift": (float(shift[0]), float(shift[1])),
            "rotation": rotation,
            "flow": flow,
            "source": source,
        }
    offset = params.get("footprint_offset")
    if offset is not None:
        offset = np.asarray(offset, dtype=float).reshape(-1)
        if offset.shape == (2,) and np.isfinite(offset).all() and np.any(offset):
            return {
                "shift": (-float(offset[0]), -float(offset[1])),
                "rotation": 0.0,
                "flow": None,
                "source": "manual",
            }
    return None


def correction_params(correction):
    """``correction`` as stored in session params (None = no correction)."""
    if correction is None:
        return None
    flow = correction.get("flow")
    return {
        "shift": [float(value) for value in correction["shift"]],
        "rotation": float(correction.get("rotation", 0.0)),
        "flow": None if flow is None else np.asarray(flow, dtype=np.float32),
        "source": correction.get("source", "manual"),
    }


def set_correction(params, correction):
    """Store ``correction`` (or remove it) in ``params``; drops the older
    footprint offset."""
    params.pop("footprint_offset", None)
    value = correction_params(correction)
    if value is None or (
        not any(value["shift"])
        and value["rotation"] == 0.0
        and value["flow"] is None
        and value["source"] == "auto"
    ):
        params.pop("background_correction", None)
    else:
        params["background_correction"] = value


def is_identity(correction):
    return correction is None or (
        not any(correction["shift"])
        and not correction.get("rotation")
        and correction.get("flow") is None
    )


def correction_remap(dims, correction):
    """The correction as a Remapping (original frame → footprint frame)."""
    from .structures.remap import Remapping

    remap = Remapping.identity(dims)
    remap.method = "background_correction"
    if correction is None:
        return remap
    dx, dy = correction["shift"]
    remap.shift = np.array([dy, dx], dtype=float)
    remap.rotation = float(correction.get("rotation", 0.0))
    remap.matrix = Remapping._rigid_matrix(tuple(dims), remap.shift, remap.rotation)
    flow = correction.get("flow")
    if flow is not None and tuple(np.shape(flow)[:2]) == tuple(dims):
        remap.flow = np.asarray(flow, dtype=np.float32)
    return remap


def apply_correction(background, correction):
    """The background moved into the footprints' frame (unchanged without
    a correction)."""
    if background is None or is_identity(correction):
        return background
    background = np.asarray(background, dtype=np.float32)
    return correction_remap(background.shape, correction).apply_remap(
        background, use_optical_flow=True
    )


def estimate_correction(background, footprint_projection):
    """Bring a background that matches its footprints up to a shift /
    rotation onto them.

    Returns (correction or None, coherence of the corrected background,
    coherence of the uncorrected one). None: no correction needed (offset
    below CORRECTION_MIN_OFFSET) or possible (the background does not
    resemble the footprints: background_coherence)."""
    before = background_coherence(background, footprint_projection)
    if not matched(before):
        return None, before, before
    if float(np.hypot(*before["offset"])) < CORRECTION_MIN_OFFSET:
        return None, before, before

    from .structures.remap import Remapping

    background = np.asarray(background, dtype=np.float32)
    dims = background.shape

    def at(angle):
        if angle == 0.0:
            return before
        rotated = Remapping._warp_dense(background, Remapping._rotation_matrix(dims, angle))
        return background_coherence(rotated, footprint_projection)

    scores = {0.0: before}

    def search(angles):
        for angle in angles:
            angle = round(float(angle), 4)
            if angle not in scores:
                scores[angle] = at(angle)
        return max(scores, key=lambda key: scores[key].get("correlation", -np.inf))

    best = search(
        np.arange(
            -CORRECTION_MAX_ROTATION,
            CORRECTION_MAX_ROTATION + CORRECTION_ROTATION_STEP / 2,
            CORRECTION_ROTATION_STEP,
        )
    )
    if best != 0.0:
        best = search(
            np.arange(
                best - CORRECTION_ROTATION_STEP,
                best + CORRECTION_ROTATION_STEP + CORRECTION_ROTATION_REFINE / 2,
                CORRECTION_ROTATION_REFINE,
            )
        )
    if scores[best]["correlation"] < before["correlation"] * (1 + CORRECTION_MIN_ROTATION_GAIN):
        best = 0.0

    # The offset moves the footprints onto the (rotated) background: the
    # background moves the opposite way.
    dx, dy = scores[best]["offset"]
    correction = {
        "shift": (-float(dx), -float(dy)),
        "rotation": float(best),
        "flow": None,
        "source": "auto",
    }
    after = background_coherence(apply_correction(background, correction), footprint_projection)
    return correction, after, before


def correct_background(params, background, footprint_projection, *, reestimate=False):
    """Set ``params``' background correction for ``background`` and return
    the coherence of the corrected background with the footprints.

    A manual correction is kept (unless ``reestimate``, e.g. for a new
    background); an automatic or missing one is estimated anew
    (estimate_correction). The returned coherence carries the uncorrected
    offset ("uncorrected_offset") when a correction is applied."""
    if "footprint_offset" in params:
        # Older footprint offset: the equivalent background correction.
        set_correction(params, correction_of(params))
    correction = correction_of(params)
    if correction is not None and correction.get("source") != "auto" and not reestimate:
        before = background_coherence(background, footprint_projection)
        after = background_coherence(apply_correction(background, correction), footprint_projection)
    else:
        correction, after, before = estimate_correction(background, footprint_projection)
        set_correction(params, correction)
    after = dict(after)
    if not is_identity(correction) and before.get("offset") is not None:
        after["uncorrected_offset"] = tuple(before["offset"])
        after["correction"] = describe_correction(correction)
    return after


def correction_key(correction):
    """Hashable identity of a correction (cache keys)."""
    if correction is None:
        return None
    flow = correction.get("flow")
    return (
        tuple(map(float, correction["shift"])),
        float(correction.get("rotation", 0.0)),
        None if flow is None else id(flow),
    )


def describe_correction(correction):
    """"background moved by (dx, dy) px, rotated r°, with flow" text."""
    if is_identity(correction):
        return "no background correction"
    dx, dy = correction["shift"]
    parts = [f"shift ({dx:+.1f}, {dy:+.1f}) px"]
    if correction.get("rotation"):
        parts.append(f"rotation {correction['rotation']:+.2f}°")
    if correction.get("flow") is not None:
        parts.append("flow")
    source = "estimated" if correction.get("source") == "auto" else "manual"
    return f"{', '.join(parts)} ({source})"


def matched(coherence):
    """The background resembles the footprints (possibly offset)."""
    return (coherence.get("zscore") or 0.0) >= COHERENCE_MIN_ZSCORE


def projection(footprints, dims, included=None):
    """Sum projection (float32 image) of the (included) footprints."""
    footprints = sparse.csc_matrix(footprints)
    if footprints.shape[0] != int(np.prod(dims)):
        return None
    if included is not None and len(included) == footprints.shape[1]:
        footprints = footprints[:, np.flatnonzero(np.asarray(included, dtype=bool))]
    return np.asarray(footprints.sum(axis=1), dtype=np.float32).reshape(tuple(dims))


def aligned_projection(session):
    """Projection of the session's included footprints in the aligned
    (common) frame; None without footprints."""
    footprints = getattr(session, "footprints", None)
    if footprints is None:
        return None
    return projection(footprints, session.dims, getattr(session, "included", None))


def unaligned_projection(session):
    """aligned_projection in the session's original frame (that of
    background_template), so that it warps like the background.

    The remap samples each output pixel p from the original position
    S(p) (Remapping.sampling_maps): every aligned pixel is splatted
    bilinearly back to S(p) and normalised by the splatted weight."""
    aligned = aligned_projection(session)
    remap = getattr(session, "remap", None)
    if aligned is None or remap is None:
        return aligned

    dims = tuple(session.dims)
    original = aligned
    matrix = np.asarray(remap.matrix, dtype=np.float64)
    if remap.success and (
        remap.flow is not None or not np.allclose(matrix, np.eye(3), rtol=0, atol=1e-8)
    ):
        x, y = remap.sampling_maps(use_optical_flow=True)
        height, width = dims
        x0, y0 = np.floor(x).astype(np.int64), np.floor(y).astype(np.int64)
        fx, fy = x - x0, y - y0

        values = np.zeros(height * width, dtype=np.float64)
        weights = np.zeros(height * width, dtype=np.float64)
        for oy, ox, w in (
            (0, 0, (1 - fx) * (1 - fy)),
            (0, 1, fx * (1 - fy)),
            (1, 0, (1 - fx) * fy),
            (1, 1, fx * fy),
        ):
            cx, cy = x0 + ox, y0 + oy
            valid = (cx >= 0) & (cx < width) & (cy >= 0) & (cy < height) & (w > 0)
            index = (cy * width + cx)[valid]
            values += np.bincount(index, (w * aligned)[valid], minlength=values.size)
            weights += np.bincount(index, w[valid], minlength=weights.size)

        with np.errstate(invalid="ignore", divide="ignore"):
            original = np.where(weights > 1e-6, values / weights, 0.0)
        original = original.reshape(dims).astype(np.float32)

    # apply_remap transposes before warping.
    return original.T if remap.transpose else original


def _high_pass(image, sigma=6.0):
    image = np.nan_to_num(np.asarray(image, dtype=np.float64))
    image = image - ndimage.gaussian_filter(image, sigma)
    image = ndimage.gaussian_filter(image, 1.0)
    image -= image.mean()
    norm = np.linalg.norm(image)
    return image / norm if norm > 0 else image


def _vertex(minus, centre, plus):
    denominator = minus - 2.0 * centre + plus
    if denominator >= 0 or not np.isfinite(denominator):
        return 0.0
    return float(np.clip(0.5 * (minus - plus) / denominator, -0.5, 0.5))


def background_coherence(background, footprint_projection, *, max_shift=COHERENCE_MAX_SHIFT):
    """Does the background match the footprints of its own session?

    Cross-correlation of both high-passed images for shifts up to
    ``max_shift`` px; the peak's robust z score against the other shifts
    says whether it is a match at all, its position whether background and
    footprints are offset. Returns a dict: ``coherent``, ``correlation``
    (peak), ``zscore``, ``offset`` = (dx, dy) that moves the footprints onto
    the background (subpixel), ``correlation_zero`` (no shift), ``reason``.
    """
    background = np.asarray(background, dtype=np.float64)
    footprint_projection = np.asarray(footprint_projection, dtype=np.float64)
    if background.shape != footprint_projection.shape or not np.any(footprint_projection):
        return {
            "coherent": None,
            "reason": "no footprints to compare with",
        }

    a = _high_pass(background)
    b = _high_pass(footprint_projection)
    # c[t] = sum_x a(x) b(x - t): the footprints moved by t match a.
    surface = np.fft.irfft2(np.fft.rfft2(a) * np.conj(np.fft.rfft2(b)), s=a.shape)
    surface = np.fft.fftshift(surface)
    cy, cx = np.array(surface.shape) // 2
    m = int(min(max_shift, cy - 1, cx - 1))
    window = surface[cy - m : cy + m + 1, cx - m : cx + m + 1]

    iy, ix = np.unravel_index(np.argmax(window), window.shape)
    peak = float(window[iy, ix])
    rest = np.ones(window.shape, dtype=bool)
    rest[max(iy - 3, 0) : iy + 4, max(ix - 3, 0) : ix + 4] = False
    values = window[rest]
    median = float(np.median(values))
    spread = 1.4826 * float(np.median(np.abs(values - median)))
    zscore = (peak - median) / spread if spread > 0 else np.inf

    dy = iy - m
    dx = ix - m
    if 0 < iy < window.shape[0] - 1:
        dy += _vertex(window[iy - 1, ix], peak, window[iy + 1, ix])
    if 0 < ix < window.shape[1] - 1:
        dx += _vertex(window[iy, ix - 1], peak, window[iy, ix + 1])

    matched = zscore >= COHERENCE_MIN_ZSCORE
    offset = float(np.hypot(dx, dy))
    coherent = bool(matched and offset <= COHERENCE_MAX_OFFSET)
    if coherent:
        reason = None
    elif not matched:
        reason = (
            f"the background does not resemble the footprints "
            f"(best correlation {peak:.2f}, z {zscore:.1f} < {COHERENCE_MIN_ZSCORE:g})"
        )
    else:
        reason = (
            f"the footprints are offset from the background by "
            f"({dx:+.1f}, {dy:+.1f}) px"
        )
    return {
        "coherent": coherent,
        "correlation": peak,
        "zscore": float(zscore),
        "offset": (float(dx), float(dy)),
        "correlation_zero": float(surface[cy, cx]),
        "reason": reason,
    }
