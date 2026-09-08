from __future__ import annotations

from dataclasses import dataclass, field, replace
from enum import Enum
from fnmatch import fnmatch
from pathlib import PurePosixPath
from typing import Iterable, Literal

# from catan.core.structures.load_config import FieldSpec


class FileFormat(str, Enum):
    HDF5 = "hdf5"
    MAT_PRE73 = "mat_pre73"
    MAT73 = "mat73"
    NPZ = "npz"
    ZARR = "zarr"

SourceTypes = Literal["session", "assignments", "model", "remapping"]

FieldKind = Literal["group", "field", "attribute"]
FieldSource = Literal["dataset", "field", "attribute"]
GroupType = Literal["static", "dynamic"]

@dataclass
class FieldSpec:
    path: str
    # ``dataset`` is kept as the serialized default for backwards
    # compatibility. The IO layer treats ``dataset`` and ``field`` identically.
    source: FieldSource = "dataset"
    attribute: str | None = None
    required: bool = False

    def to_dict(self) -> dict:
        return {
            "path": self.path,
            "source": self.source,
            "attribute": self.attribute,
            "required": self.required,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "FieldSpec":
        return cls(
            path=data["path"],
            source=data.get("source", "dataset"),
            attribute=data.get("attribute"),
            required=data.get("required", False),
        )


@dataclass
class FieldGroupSpec:
    title: str
    type: GroupType
    fields: dict[str, FieldSpec] = field(default_factory=dict)
    enabled: bool = True

    def add_field(self, name: str, spec: FieldSpec) -> None:
        if self.type != "dynamic":
            raise ValueError(f"Cannot add fields to static group {self.title!r}")
        if name in self.fields:
            raise KeyError(f"Field {name!r} already exists")
        self.fields[name] = spec

    def remove_field(self, name: str) -> None:
        if self.type != "dynamic":
            raise ValueError(f"Cannot remove fields from static group {self.title!r}")
        if name not in self.fields:
            raise KeyError(f"Unknown field {name!r}")
        del self.fields[name]

    def rename_field(self, old_name: str, new_name: str) -> None:
        if old_name == new_name:
            return
        
        if self.type != "dynamic":
            raise ValueError(f"Cannot rename fields in static group {self.title!r}")
        if old_name not in self.fields:
            raise KeyError(f"Unknown field {old_name!r}")
        if new_name in self.fields:
            raise KeyError(f"Field {new_name!r} already exists")
        self.fields[new_name] = self.fields.pop(old_name)

    def set_field_path(self, name: str, path: str) -> None:
        if name not in self.fields:
            raise KeyError(f"Unknown field {name!r}")
        self.fields[name].path = path

    def to_dict(self) -> dict:
        return {
            "title": self.title,
            "type": self.type,
            "enabled": self.enabled,
            "fields": {name: spec.to_dict() for name, spec in self.fields.items()},
        }

    @classmethod
    def from_dict(cls, data: dict) -> "FieldGroupSpec":
        return cls(
            title=data["title"],
            type=data["type"],
            enabled=data.get("enabled", True),
            fields={
                name: FieldSpec.from_dict(spec)
                for name, spec in data.get("fields", {}).items()
            },
        )

@dataclass(slots=True)
class FieldInfo:
    """Metadata for one logical field visible to CATAN.

    ``path`` is always relative to the logical root used for inspection and
    uses POSIX-style separators, independent of the host OS.
    """

    name: str
    path: str
    kind: FieldKind
    shape: tuple[int, ...] | None = None
    dtype: str | None = None
    attribute: str | None = None
    selected: bool = False

    @property
    def parent(self) -> str:
        if self.kind == "attribute":
            return self.path
        parent = str(PurePosixPath(self.path).parent)
        return "/" if parent in (".", "") else parent

    @property
    def display_shape(self) -> str:
        return "" if self.shape is None else str(tuple(self.shape))

    @property
    def display_dtype(self) -> str:
        return self.dtype or ""


@dataclass(slots=True)
class FileStructure:
    """Flat metadata view of a file/tree, suitable for GUI browsing."""

    file_format: FileFormat
    root: str = "/"
    entries: dict[str, FieldInfo] = field(default_factory=dict)
    attributes: dict[tuple[str, str], FieldInfo] = field(default_factory=dict)

    def add(self, info: FieldInfo) -> None:
        if info.kind == "attribute":
            if info.attribute is None:
                raise ValueError("Attribute FieldInfo requires attribute name")
            self.attributes[(normalize_path(info.path), info.attribute)] = info
        else:
            self.entries[normalize_path(info.path)] = info

    def children(
        self,
        subpath: str = "/",
        *,
        include_attributes: bool = True,
        selected: str | None = None,
    ) -> list[FieldInfo]:
        subpath = normalize_path(subpath)
        out: list[FieldInfo] = []

        for info in self.entries.values():
            if info.path == "/":
                continue
            parent = str(PurePosixPath(info.path).parent)
            parent = "/" if parent in (".", "") else normalize_path(parent)
            if parent == subpath:
                out.append(replace(info, selected=(selected == info.path)))

        if include_attributes:
            for (path, _), info in self.attributes.items():
                if path == subpath:
                    selection_key = f"{path}@{info.attribute}"
                    out.append(
                        replace(
                            info,
                            selected=(selected in {selection_key, info.attribute}),
                        )
                    )

        return out

    def matches(self, spec: FieldSpec) -> bool:
        source = normalized_source(spec.source)
        path = normalize_path(spec.path)

        if source == "attribute":
            attribute = spec.attribute
            if attribute is None:
                return False
            if attribute == "*":
                return any(attr_path == path for attr_path, _ in self.attributes)
            return (path, attribute) in self.attributes

        if has_wildcard(path):
            return any(
                info.kind == "field" and fnmatch(candidate, path)
                for candidate, info in self.entries.items()
            )

        info = self.entries.get(path)
        return info is not None and info.kind == "field"


@dataclass(slots=True)
class FieldCompatibility:
    group: str
    label: str
    spec: FieldSpec
    available: bool
    reason: str | None = None


@dataclass(slots=True)
class CompatibilityReport:
    file_format: FileFormat
    fields: list[FieldCompatibility]

    @property
    def compatible(self) -> bool:
        return not any(
            (not item.available) and item.spec.required
            for item in self.fields
        )

    @property
    def missing_required(self) -> list[FieldCompatibility]:
        return [
            item for item in self.fields
            if (not item.available) and item.spec.required
        ]

    @property
    def missing_optional(self) -> list[FieldCompatibility]:
        return [
            item for item in self.fields
            if (not item.available) and not item.spec.required
        ]

    @property
    def available(self) -> list[FieldCompatibility]:
        return [item for item in self.fields if item.available]


@dataclass(slots=True)
class SaveEntry:
    group: str
    label: str
    spec: FieldSpec
    value: object


def normalize_path(path: str | PurePosixPath) -> str:
    text = str(path).strip()
    if text in ("", ".", "/"):
        return "/"
    return "/" + text.strip("/")


def join_path(base: str, child: str) -> str:
    base = normalize_path(base)
    child = str(child).strip("/")
    if not child:
        return base
    if base == "/":
        return f"/{child}"
    return f"{base}/{child}"


def relative_path(path: str, root: str) -> str:
    path = normalize_path(path)
    root = normalize_path(root)
    if root == "/":
        return path
    if path == root:
        return "/"
    prefix = root.rstrip("/") + "/"
    if not path.startswith(prefix):
        raise ValueError(f"{path!r} is not below logical root {root!r}")
    return "/" + path[len(prefix):]


def has_wildcard(path: str) -> bool:
    return any(ch in path for ch in "*?[")


def normalized_source(source: str) -> Literal["field", "attribute"]:
    # Keep old JSON configs using ``dataset`` valid while making the IO layer
    # format-neutral.
    if source in ("dataset", "field"):
        return "field"
    if source == "attribute":
        return "attribute"
    raise ValueError(f"Unsupported field source: {source!r}")


def check_structure_compatibility(
    structure: FileStructure,
    fields_to_load: dict[str, dict[str, FieldSpec]],
) -> CompatibilityReport:
    checks: list[FieldCompatibility] = []
    for group_name, fields in fields_to_load.items():
        for label, spec in fields.items():
            available = structure.matches(spec)
            checks.append(
                FieldCompatibility(
                    group=group_name,
                    label=label,
                    spec=spec,
                    available=available,
                    reason=None if available else "Configured path is not available",
                )
            )
    return CompatibilityReport(structure.file_format, checks)
