import json
from colorsys import hls_to_rgb, rgb_to_hls
from hashlib import sha256

import numpy as np
from PySide6.QtCore import QObject, Signal


def assignment_label(name):
    return "Project" if name == "local" else str(name)


# Automatic colours shown together keep at least this hue distance (of 1).
MIN_HUE_DISTANCE = 0.12
_LIGHTNESS, _SATURATION = 0.46, 0.62

# User-chosen colours (name -> RGBA), persisted in the settings.
SETTINGS_KEY = "statistics/assignment_colors"
_overrides: dict[str, tuple] = {}
_settings = None


class _ColorSignals(QObject):
    changed = Signal()


color_signals = _ColorSignals()


def _hash_hue(name):
    """Stable across processes and changes to the displayed dataset list."""
    digest = sha256(str(name).encode("utf-8")).digest()
    return int.from_bytes(digest[:4], "big") / 2**32


def _hue_color(hue):
    return np.asarray((*hls_to_rgb(hue, _LIGHTNESS, _SATURATION), 1.0), dtype=np.float32)


def _hue_distance(a, b):
    d = abs(a - b) % 1.0
    return min(d, 1.0 - d)


def assignment_color(name):
    return assignment_colors([name])[0]


def assignment_colors(names):
    """One RGBA colour per name; distinct names get distinct colours.

    User-chosen colours are kept. An automatic colour (the name's hashed
    hue) closer than MIN_HUE_DISTANCE to a colour already given moves to
    the free hue farthest from the others.
    """
    names = [str(name) for name in names]
    unique = list(dict.fromkeys(names))
    chosen = {}
    hues = []

    for name in unique:
        if name in _overrides:
            chosen[name] = np.asarray(_overrides[name], dtype=np.float32)
            hues.append(rgb_to_hls(*chosen[name][:3])[0])

    for name in unique:
        if name in chosen:
            continue

        hue = _hash_hue(name)
        if hues and min(_hue_distance(hue, other) for other in hues) < MIN_HUE_DISTANCE:
            candidates = (hue + k / 36 for k in range(36))
            hue = max(
                candidates,
                key=lambda h: min(_hue_distance(h, other) for other in hues),
            ) % 1.0

        hues.append(hue)
        chosen[name] = _hue_color(hue)

    return np.asarray([chosen[name] for name in names], dtype=np.float32).reshape(
        len(names), 4
    )


def has_custom_color(name):
    return str(name) in _overrides


def set_assignment_color(name, rgba=None):
    """User colour for ``name``; None returns to the automatic colour."""
    if rgba is None:
        _overrides.pop(str(name), None)
    else:
        _overrides[str(name)] = tuple(float(v) for v in rgba)

    if _settings is not None:
        _settings.setValue(SETTINGS_KEY, json.dumps(_overrides))

    color_signals.changed.emit()


def load_assignment_colors(settings):
    """Read the user colours from (and later save them to) ``settings``."""
    global _settings
    _settings = settings
    _overrides.clear()

    try:
        stored = json.loads(settings.value(SETTINGS_KEY, "{}") or "{}")
        _overrides.update(
            {str(name): tuple(float(v) for v in rgba) for name, rgba in stored.items()}
        )
    except (TypeError, ValueError, AttributeError):
        pass


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
