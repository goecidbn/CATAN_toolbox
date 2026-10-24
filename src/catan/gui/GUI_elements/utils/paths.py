"""Displaying file paths relative to the data root folder (main menu)."""

import os
from pathlib import Path

_display_root = None


def _absolute(path):
    # No resolve(): avoid file system access (network mounts).
    return Path(os.path.normpath(os.path.abspath(os.path.expanduser(os.fspath(path)))))


def set_display_root(root):
    """Called whenever Data.root changes."""
    global _display_root

    try:
        _display_root = _absolute(root) if root else None
    except (OSError, TypeError, ValueError):
        _display_root = None


def display_path(path, *, root=None):
    """``path`` relative to the root folder if it lies inside it, else the
    full path. Only for display; never use the result to open files."""
    if path is None or path == "":
        return ""

    text = os.fspath(path)
    base = _display_root if root is None else _absolute(root)

    if base is None:
        return text

    try:
        relative = _absolute(text).relative_to(base)
    except (OSError, TypeError, ValueError):
        return text

    return relative.as_posix() if relative.parts else "."


def shorten_paths(text):
    """``text`` with absolute paths inside the root folder made relative."""
    if not text or _display_root is None:
        return text

    root = str(_display_root).rstrip(os.sep) + os.sep
    return text.replace(root, "")
