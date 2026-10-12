from colorsys import hls_to_rgb
from hashlib import sha256

import numpy as np


def assignment_label(name):
    return "Working" if name == "local" else str(name)


def assignment_color(name):
    """Stable across processes and changes to the displayed dataset list."""
    digest = sha256(str(name).encode("utf-8")).digest()
    hue = int.from_bytes(digest[:4], "big") / 2**32

    return np.asarray(
        (*hls_to_rgb(hue, 0.46, 0.62), 1.0),
        dtype=np.float32,
    )


def assignment_shade(colors, style="default"):
    """Accept one RGBA color or an array of RGBA colors."""
    result = np.asarray(colors, dtype=np.float32).copy()
    rgb = result[..., :3]

    if style in {"default", "background"}:
        result[..., :3] = 0.55 * rgb + 0.45
    elif style == "hovered":
        result[..., :3] = 0.62 * rgb
    else:
        # Selected/focused: the original, stronger dataset color.
        result[..., :3] = rgb

    result[..., 3] = 1.0
    return result
