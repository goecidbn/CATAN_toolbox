from __future__ import annotations

from contextlib import contextmanager
from fnmatch import fnmatch
from pathlib import Path, PurePosixPath
from typing import Any, Iterator

import numpy as np
from scipy import sparse

from catan.core.structures.load_config import FieldSpec

from .base import IOBackend
from .common import nested_dict_group, nested_dict_set, normalize_scalar
from .types import (
    FieldInfo,
    FileFormat,
    FileStructure,
    SaveEntry,
    has_wildcard,
    join_path,
    normalize_path,
    normalized_source,
)

NPZ_ATTRS_SEGMENT = "__catan_attrs__"
NPZ_SPARSE_SEGMENT = "__catan_sparse__"
ATTRIBUTES_KEY = "__attrs__"


class NPZBackend(IOBackend):
    file_format = FileFormat.NPZ

    def __init__(self, *, compressed: bool = True) -> None:
        self.compressed = compressed

    @contextmanager
    def open_read(self, path: str | Path) -> Iterator[np.lib.npyio.NpzFile]:
        with np.load(path, allow_pickle=False) as ref:
            yield ref

    @contextmanager
    def open_write(self, path: str | Path) -> Iterator[dict[str, Any]]:
        data: dict[str, Any] = {}
        yield data
        writer = np.savez_compressed if self.compressed else np.savez
        writer(path, **data)

    def read_field(
        self,
        ref: np.lib.npyio.NpzFile,
        spec: FieldSpec,
        *,
        key: str | None,
        root: str,
    ) -> dict[str, Any]:
        source = normalized_source(spec.source)
        logical = normalize_path(spec.path)

        if source == "attribute":
            if spec.attribute == "*":
                prefix = _attribute_prefix(root, logical)
                return {
                    candidate[len(prefix):]: normalize_npz_value(ref[candidate])
                    for candidate in ref.files
                    if candidate.startswith(prefix) and "/" not in candidate[len(prefix):]
                }
            if spec.attribute is None:
                return {}
            attr_key = _attribute_key(root, logical, spec.attribute)
            if attr_key not in ref.files:
                if spec.required:
                    raise KeyError(f"NPZ attribute {spec.attribute!r} not found at {logical!r}")
                return {}
            return {key or spec.attribute: normalize_npz_value(ref[attr_key])}

        full_key = _field_key(root, logical)

        if has_wildcard(logical):
            pure = PurePosixPath(full_key)
            parent = "" if str(pure.parent) == "." else str(pure.parent).strip("/")
            pattern = pure.name
            prefix = parent + "/" if parent else ""
            out: dict[str, Any] = {}
            for candidate in _logical_field_keys(ref):
                if "/" in candidate[len(prefix):] if candidate.startswith(prefix) else True:
                    continue
                name = candidate[len(prefix):]
                if fnmatch(name, pattern):
                    out[name] = _read_npz_logical_field(ref, candidate)
            if not out and spec.required:
                raise KeyError(f"No NPZ fields match {logical!r}")
            return out

        if not _npz_logical_field_exists(ref, full_key):
            if spec.required:
                raise KeyError(f"NPZ field {logical!r} not found")
            return {}
        return {key or PurePosixPath(logical).name: _read_npz_logical_field(ref, full_key)}

    def write_entry(self, ref: dict[str, Any], entry: SaveEntry, *, root: str) -> None:
        spec = entry.spec
        logical = normalize_path(spec.path)
        if normalized_source(spec.source) == "attribute":
            if spec.attribute is None:
                raise ValueError(f"Attribute field {entry.label!r} has no attribute name")
            ref[_attribute_key(root, logical, spec.attribute)] = _npz_storable(entry.value)
            return

        key = _field_key(root, logical)
        _write_npz_logical_field(ref, key, entry.value)

    def load_all(self, ref: np.lib.npyio.NpzFile, *, root: str = "/") -> dict[str, Any]:
        root_prefix = _root_prefix(root)
        result: dict[str, Any] = {}

        # Logical regular/sparse fields.
        for key in _logical_field_keys(ref):
            if root_prefix and not key.startswith(root_prefix):
                continue
            relative = key[len(root_prefix):] if root_prefix else key
            if not relative or relative.startswith(NPZ_ATTRS_SEGMENT + "/"):
                continue
            nested_dict_set(result, "/" + relative, _read_npz_logical_field(ref, key))

        # Reconstruct attributes as __attrs__ dictionaries.
        for key in ref.files:
            marker = f"/{NPZ_ATTRS_SEGMENT}/"
            if key.startswith(NPZ_ATTRS_SEGMENT + "/"):
                obj_path = "/"
                attr_name = key[len(NPZ_ATTRS_SEGMENT) + 1:]
            elif marker in key:
                before, attr_name = key.split(marker, 1)
                obj_path = "/" + before
            else:
                continue
            full_obj = _field_key(root, obj_path)
            if root_prefix and not full_obj.startswith(root_prefix.rstrip("/")):
                continue
            relative_obj = _relative_key(full_obj, root)
            target = nested_dict_group(result, relative_obj)
            attrs = target.setdefault(ATTRIBUTES_KEY, {})
            attrs[attr_name] = normalize_npz_value(ref[key])

        return result

    def inspect(self, ref: np.lib.npyio.NpzFile, *, root: str = "/") -> FileStructure:
        structure = FileStructure(self.file_format, root=normalize_path(root))
        root_prefix = _root_prefix(root)
        logical_fields = _logical_field_keys(ref)

        groups: set[str] = set()
        for key in logical_fields:
            if root_prefix and not key.startswith(root_prefix):
                continue
            relative = key[len(root_prefix):] if root_prefix else key
            if not relative:
                continue
            parts = PurePosixPath(relative).parts
            for i in range(1, len(parts)):
                groups.add("/" + "/".join(parts[:i]))

        for group in sorted(groups):
            structure.add(FieldInfo(PurePosixPath(group).name, group, "group"))

        for key in sorted(logical_fields):
            if root_prefix and not key.startswith(root_prefix):
                continue
            relative = key[len(root_prefix):] if root_prefix else key
            if not relative:
                continue
            logical_path = "/" + relative
            shape, dtype = _npz_field_metadata(ref, key)
            structure.add(
                FieldInfo(
                    name=PurePosixPath(logical_path).name,
                    path=logical_path,
                    kind="field",
                    shape=shape,
                    dtype=dtype,
                )
            )

        for key in ref.files:
            parsed = _parse_attribute_key(key)
            if parsed is None:
                continue
            object_key, attr_name = parsed
            if root_prefix and object_key and not object_key.startswith(root_prefix.rstrip("/")):
                continue
            relative_obj = _relative_key(object_key, root)
            shape, dtype = _npy_member_metadata(ref, key)
            structure.add(
                FieldInfo(
                    name=attr_name,
                    path=relative_obj,
                    kind="attribute",
                    shape=shape,
                    dtype=dtype,
                    attribute=attr_name,
                )
            )

        return structure

    def get_attribute(
        self,
        ref: np.lib.npyio.NpzFile,
        path: str,
        name: str,
        *,
        root: str = "/",
        default: Any = None,
    ) -> Any:
        key = _attribute_key(root, path, name)
        return normalize_npz_value(ref[key]) if key in ref.files else default

    def set_attribute(
        self,
        ref: dict[str, Any],
        path: str,
        name: str,
        value: Any,
        *,
        root: str = "/",
    ) -> None:
        ref[_attribute_key(root, path, name)] = _npz_storable(value)

    def list_groups(self, ref: np.lib.npyio.NpzFile, *, root: str = "/") -> list[str]:
        prefix = _root_prefix(root)
        groups: set[str] = set()
        for key in ref.files:
            if prefix and not key.startswith(prefix):
                continue
            relative = key[len(prefix):] if prefix else key
            first = relative.split("/", 1)[0]
            if not first or first in {NPZ_ATTRS_SEGMENT, NPZ_SPARSE_SEGMENT}:
                continue
            if "/" in relative:
                groups.add(first)
        return sorted(groups)

    def exists(
        self,
        ref,
        path: str,
    ) -> bool:

        path = path.strip("/")

        if not path:
            return True

        # Exact array
        if path in ref:
            return True

        # Logical group:
        prefix = path + "/"

        return any(
            key.startswith(prefix)
            for key in ref.keys()
        )


def normalize_npz_value(value: Any) -> Any:
    arr = np.asarray(value)
    if arr.shape == ():
        return normalize_scalar(arr.item())
    if arr.dtype.kind == "S":
        return arr.astype(str)
    return arr


def _npz_storable(value: Any) -> Any:
    if isinstance(value, Path):
        return np.asarray(str(value))
    if isinstance(value, str):
        return np.asarray(value)
    if isinstance(value, tuple):
        return np.asarray(value)
    return value


def _root_prefix(root: str) -> str:
    root = normalize_path(root).strip("/")
    return root + "/" if root else ""


def _field_key(root: str, path: str) -> str:
    root_part = normalize_path(root).strip("/")
    path_part = normalize_path(path).strip("/")
    return "/".join(part for part in (root_part, path_part) if part)


def _attribute_prefix(root: str, path: str) -> str:
    base = _field_key(root, path)
    return f"{base}/{NPZ_ATTRS_SEGMENT}/" if base else f"{NPZ_ATTRS_SEGMENT}/"


def _attribute_key(root: str, path: str, name: str) -> str:
    return _attribute_prefix(root, path) + name


def _parse_attribute_key(key: str) -> tuple[str, str] | None:
    root_prefix = NPZ_ATTRS_SEGMENT + "/"
    if key.startswith(root_prefix):
        return "", key[len(root_prefix):]
    marker = f"/{NPZ_ATTRS_SEGMENT}/"
    if marker not in key:
        return None
    obj, attr = key.split(marker, 1)
    return obj, attr


def _sparse_prefix(key: str) -> str:
    return f"{key}/{NPZ_SPARSE_SEGMENT}"


def _write_npz_logical_field(ref: dict[str, Any], key: str, value: Any) -> None:
    if sparse.issparse(value):
        if sparse.isspmatrix_csr(value):
            matrix = value.tocsr()
            fmt = "csr"
        else:
            matrix = value.tocsc()
            fmt = "csc"
        prefix = _sparse_prefix(key)
        ref[f"{prefix}/data"] = matrix.data
        ref[f"{prefix}/indices"] = matrix.indices
        ref[f"{prefix}/indptr"] = matrix.indptr
        ref[f"{prefix}/shape"] = np.asarray(matrix.shape, dtype=np.int64)
        ref[f"{prefix}/format"] = np.asarray(fmt)
        return
    ref[key] = _npz_storable(value)


def _npz_logical_field_exists(ref: np.lib.npyio.NpzFile, key: str) -> bool:
    return key in ref.files or f"{_sparse_prefix(key)}/data" in ref.files


def _read_npz_logical_field(ref: np.lib.npyio.NpzFile, key: str) -> Any:
    if key in ref.files:
        return normalize_npz_value(ref[key])
    prefix = _sparse_prefix(key)
    data_key = f"{prefix}/data"
    if data_key not in ref.files:
        raise KeyError(key)
    fmt = str(normalize_npz_value(ref[f"{prefix}/format"])).lower()
    shape = tuple(int(v) for v in np.asarray(ref[f"{prefix}/shape"]).ravel())
    args = (
        ref[data_key],
        ref[f"{prefix}/indices"],
        ref[f"{prefix}/indptr"],
    )
    if fmt == "csr":
        return sparse.csr_matrix(args, shape=shape)
    if fmt == "csc":
        return sparse.csc_matrix(args, shape=shape)
    raise ValueError(f"Unsupported NPZ sparse format {fmt!r}")


def _logical_field_keys(ref: np.lib.npyio.NpzFile) -> set[str]:
    keys: set[str] = set()
    sparse_marker = f"/{NPZ_SPARSE_SEGMENT}/"
    for key in ref.files:
        if _parse_attribute_key(key) is not None:
            continue
        if sparse_marker in key:
            keys.add(key.split(sparse_marker, 1)[0])
        elif key.startswith(NPZ_SPARSE_SEGMENT + "/"):
            continue
        else:
            keys.add(key)
    return keys


def _npz_field_metadata(
    ref: np.lib.npyio.NpzFile,
    key: str,
) -> tuple[tuple[int, ...] | None, str | None]:
    if key in ref.files:
        return _npy_member_metadata(ref, key)
    prefix = _sparse_prefix(key)
    shape = tuple(int(v) for v in np.asarray(ref[f"{prefix}/shape"]).ravel())
    _, dtype = _npy_member_metadata(ref, f"{prefix}/data")
    return shape, dtype


def _npy_member_metadata(
    ref: np.lib.npyio.NpzFile,
    key: str,
) -> tuple[tuple[int, ...] | None, str | None]:
    # NpzFile exposes the underlying ZipFile as ``zip``. Reading only the NPY
    # header avoids inflating remote/large arrays during field inspection.
    member = key + ".npy"
    with ref.zip.open(member) as fh:
        version = np.lib.format.read_magic(fh)
        if version == (1, 0):
            shape, _, dtype = np.lib.format.read_array_header_1_0(fh)
        else:
            shape, _, dtype = np.lib.format.read_array_header_2_0(fh)
    return tuple(shape), str(dtype)


def _relative_key(object_key: str, root: str) -> str:
    root_key = normalize_path(root).strip("/")
    if not object_key:
        return "/"
    if not root_key:
        return "/" + object_key.strip("/")
    if object_key == root_key:
        return "/"
    prefix = root_key + "/"
    if object_key.startswith(prefix):
        return "/" + object_key[len(prefix):]
    return "/" + object_key.strip("/")
