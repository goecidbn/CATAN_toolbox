"""Access to immutable resources bundled with the CATAN GUI."""

from __future__ import annotations

from contextlib import AbstractContextManager
from importlib.resources import as_file, files
from importlib.resources.abc import Traversable
from pathlib import Path, PurePosixPath


_RESOURCE_PACKAGE = "catan.gui.resources"


def resource(relative_path: str) -> Traversable:
    """
    Return a resource relative to ``catan.gui.resources``.

    Parameters
    ----------
    relative_path:
        Forward-slash-separated resource path, for example
        ``"styles/main.qss"`` or ``"icons/open.svg"``.

    Raises
    ------
    ValueError
        If the path is absolute or attempts to leave the resource
        directory using ``..``.
    """
    path = PurePosixPath(relative_path)

    if path.is_absolute():
        raise ValueError(
            f"Resource path must be relative: {relative_path!r}"
        )

    if ".." in path.parts:
        raise ValueError(
            f"Resource path may not contain '..': {relative_path!r}"
        )

    item = files(_RESOURCE_PACKAGE)

    for part in path.parts:
        if part not in {"", "."}:
            item = item.joinpath(part)

    return item


def resource_file(relative_path: str) -> Traversable:
    """
    Return an existing file resource.

    Raises
    ------
    FileNotFoundError
        If the resource does not exist or is not a file.
    """
    item = resource(relative_path)

    if not item.is_file():
        raise FileNotFoundError(
            f"CATAN resource does not exist or is not a file: "
            f"{relative_path!r}"
        )

    return item


def resource_directory(relative_path: str) -> Traversable:
    """
    Return an existing resource directory.

    Raises
    ------
    NotADirectoryError
        If the resource does not exist or is not a directory.
    """
    item = resource(relative_path)

    if not item.is_dir():
        raise NotADirectoryError(
            f"CATAN resource does not exist or is not a directory: "
            f"{relative_path!r}"
        )

    return item


def read_text(
    relative_path: str,
    *,
    encoding: str = "utf-8",
) -> str:
    """Read a bundled text resource."""
    return resource_file(relative_path).read_text(encoding=encoding)


def read_bytes(relative_path: str) -> bytes:
    """Read a bundled binary resource."""
    return resource_file(relative_path).read_bytes()


def resource_path(
    relative_path: str,
) -> AbstractContextManager[Path]:
    """
    Temporarily provide a filesystem path for a resource.

    Use this only for APIs that require an actual filesystem path.

    Examples
    --------
    with resource_path("icons/open.svg") as path:
        external_library.load(str(path))

    Notes
    -----
    The path should not be retained after leaving the ``with`` block.
    """
    return as_file(resource_file(relative_path))


def load_stylesheet(name: str = "main.qss") -> str:
    """Load a stylesheet from the bundled styles directory."""
    return read_text(f"styles/{name}")


def load_default_text(name: str) -> str:
    """Load a bundled default configuration as text."""
    return read_text(f"defaults/{name}")


def combine_stylesheets(*styles: str | None) -> str:
    """Combine non-empty stylesheet fragments."""
    return "\n\n".join(
        style.strip()
        for style in styles
        if style and style.strip()
    )