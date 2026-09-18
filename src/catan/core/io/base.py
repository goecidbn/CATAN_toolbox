from __future__ import annotations

from abc import ABC, abstractmethod
from contextlib import AbstractContextManager
from pathlib import Path
from typing import Any

from catan.core.structures.load_config import FieldSpec

from .common import iter_save_entries
from .types import FileFormat, FileStructure, SaveEntry


class IOBackend(ABC):
    """Small backend contract used by CATAN's public IO dispatcher."""

    file_format: FileFormat

    @abstractmethod
    def open_read(self, path: str | Path) -> AbstractContextManager[Any]: ...

    @abstractmethod
    def open_write(self, path: str | Path) -> AbstractContextManager[Any]: ...

    def load(
        self,
        ref: Any,
        fields_to_load: dict[str, dict[str, FieldSpec]] | None,
        *,
        root: str = "/",
    ) -> dict[str, Any]:
        if fields_to_load is None:
            return self.load_all(ref, root=root)

        result: dict[str, Any] = {}
        for group_name, fields in fields_to_load.items():
            group_result: dict[str, Any] = {}
            for label, spec in fields.items():
                group_result.update(self.read_field(ref, spec, key=label, root=root))
            result[group_name] = group_result
        return result

    def write(
        self,
        ref: Any,
        data: Any,
        fields_to_save: dict[str, dict[str, FieldSpec]],
        *,
        root: str = "/",
    ) -> None:
        for entry in iter_save_entries(data, fields_to_save):
            self.write_entry(ref, entry, root=root)

    @abstractmethod
    def read_field(
        self,
        ref: Any,
        spec: FieldSpec,
        *,
        key: str | None,
        root: str,
    ) -> dict[str, Any]: ...

    @abstractmethod
    def write_entry(self, ref: Any, entry: SaveEntry, *, root: str) -> None: ...

    @abstractmethod
    def load_all(self, ref: Any, *, root: str = "/") -> dict[str, Any]: ...

    @abstractmethod
    def inspect(self, ref: Any, *, root: str = "/") -> FileStructure: ...

    def inspect_file(
        self,
        path: str | Path,
        *,
        root: str = "/",
    ) -> FileStructure:

        with self.open_read(path) as ref:
            return self.inspect(
                ref,
                root=root,
            )

    @abstractmethod
    def get_attribute(
        self,
        ref: Any,
        path: str,
        name: str,
        *,
        root: str = "/",
        default: Any = None,
    ) -> Any: ...

    @abstractmethod
    def set_attribute(
        self,
        ref: Any,
        path: str,
        name: str,
        value: Any,
        *,
        root: str = "/",
    ) -> None: ...

    @abstractmethod
    def list_groups(
        self,
        ref: Any,
        *,
        root: str = "/",
    ) -> list[str]:
        """Return immediate child-group names below ``root``."""
        ...

    @abstractmethod
    def exists(
        self,
        ref: Any,
        path: str,
    ) -> bool:
        """
        Check whether a logical path exists inside the opened source.
        """
        ...
