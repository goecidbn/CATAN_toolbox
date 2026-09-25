from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from typing import Any

from catan.core.structures.load_config import FieldSpec

from .api import get_backend, resolve_source_path
from .types import (
    CompatibilityReport,
    FieldCompatibility,
    MultiSourceCompatibilityReport,
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


def check_fields_compatibility(
    primary_path: str | Path,
    fields_to_load: dict[str, dict[str, FieldSpec]],
    *,
    root: str = "/",
    refresh: bool = False,
) -> MultiSourceCompatibilityReport:

    primary_path = Path(primary_path)

    structures = {}
    checks = []

    for group_name, fields in fields_to_load.items():

        for label, spec in fields.items():

            source_path = resolve_source_path(primary_path, spec.source_path)
            source_root = root if spec.source_path is None else "/"

            cache_key = (str(source_path), source_root)

            try:
                if cache_key not in structures:
                    backend = get_backend(source_path)

                    structures[cache_key] = backend.inspect_file(
                        source_path, root=source_root
                    )

                structure = structures[cache_key]
                available = structure.matches(spec)

                reason = None if available else "Configured path is not available"

            except Exception as exc:

                available = False
                reason = f"{type(exc).__name__}: {exc}"

            checks.append(
                FieldCompatibility(
                    group=group_name,
                    label=label,
                    spec=spec,
                    available=available,
                    reason=reason,
                )
            )

    return MultiSourceCompatibilityReport(checks)


def evaluate_fields_compatibility(
    primary_path: str | Path,
    fields_to_load: dict[str, dict[str, FieldSpec]],
    *,
    root: str = "/",
    required_only: bool = False,
) -> bool:

    report = check_fields_compatibility(primary_path, fields_to_load, root=root)

    for field in report.fields:
        is_ok = field.available or (not field.spec.required and required_only)
        if not is_ok:
            return False

    return True


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


def evaluate_file_compatibility(
    path: str | Path,
    fields_to_load: dict[str, dict[str, FieldSpec]],
    *,
    root: str = "/",
    refresh: bool = False,
    required_only: bool = False,
) -> bool:
    report = check_file_compatibility(path, fields_to_load, root=root, refresh=refresh)
    possible = True
    for field in report.fields:
        is_ok = field.available or (not field.spec.required and required_only)
        if not is_ok:
            possible = False
            break
    return possible


def clear_inspection_cache() -> None:
    _DEFAULT_INSPECTOR.clear()
