from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from typing import Any

from catan.core.structures.load_config import FieldSpec

from .api import get_backend
from .types import (
    CompatibilityReport,
    FieldInfo,
    FileStructure,
    check_structure_compatibility,
)


class FileInspector:
    """Single-file metadata cache used by the field browser and validation.

    Only the most recently inspected file/root is retained, matching the GUI
    workflow where a user usually browses one source file at a time. This avoids
    repeated remote metadata traversal without introducing cache invalidation
    complexity across many files.
    """

    def __init__(self) -> None:
        self._cache_key: tuple[str, str] | None = None
        self._structure: FileStructure | None = None

    def clear(self) -> None:
        self._cache_key = None
        self._structure = None

    def inspect(
        self,
        path: str | Path,
        *,
        root: str = "/",
        refresh: bool = False,
    ) -> FileStructure:
        key = (str(Path(path)), root)
        if not refresh and self._cache_key == key and self._structure is not None:
            return self._structure

        backend = get_backend(path)
        structure = backend.inspect_file(path)
        # with backend.open_read(path) as ref:
        #     structure = backend.inspect(ref, root=root)

        self._cache_key = key
        self._structure = structure
        return structure

    def browse(
        self,
        path: str | Path,
        *,
        root: str = "/",
        subpath: str = "/",
        selected: str | None = None,
        refresh: bool = False,
        include_attributes: bool = True,
    ) -> list[FieldInfo]:
        structure = self.inspect(path, root=root, refresh=refresh)
        return structure.children(
            subpath,
            include_attributes=include_attributes,
            selected=selected,
        )

    def check_compatibility(
        self,
        path: str | Path,
        fields_to_load: dict[str, dict[str, FieldSpec]],
        *,
        root: str = "/",
        refresh: bool = False,
    ) -> CompatibilityReport:
        structure = self.inspect(path, root=root, refresh=refresh)
        return check_structure_compatibility(structure, fields_to_load)


_DEFAULT_INSPECTOR = FileInspector()


def inspect_file(
    path: str | Path,
    *,
    root: str = "/",
    refresh: bool = False,
) -> FileStructure:
    return _DEFAULT_INSPECTOR.inspect(path, root=root, refresh=refresh)


def browse_file_fields(
    path: str | Path,
    *,
    root: str = "/",
    subpath: str = "/",
    selected: str | None = None,
    refresh: bool = False,
    include_attributes: bool = True,
) -> list[FieldInfo]:
    return _DEFAULT_INSPECTOR.browse(
        path,
        root=root,
        subpath=subpath,
        selected=selected,
        refresh=refresh,
        include_attributes=include_attributes,
    )


def check_file_compatibility(
    path: str | Path,
    fields_to_load: dict[str, dict[str, FieldSpec]],
    *,
    root: str = "/",
    refresh: bool = False,
) -> CompatibilityReport:
    return _DEFAULT_INSPECTOR.check_compatibility(
        path,
        fields_to_load,
        root=root,
        refresh=refresh,
    )


def clear_inspection_cache() -> None:
    _DEFAULT_INSPECTOR.clear()
