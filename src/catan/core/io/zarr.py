from __future__ import annotations

from contextlib import contextmanager
from fnmatch import fnmatch
from pathlib import Path, PurePosixPath
from typing import Any, Iterator

import numpy as np
from scipy import sparse

from catan.core.structures.load_config import FieldSpec

from .base import IOBackend
from .common import normalize_scalar
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

ATTRIBUTES_KEY = "__attrs__"


def _zarr():
    try:
        import zarr
    except ImportError as exc:
        raise ImportError(
            "Zarr support requires the optional dependency 'zarr'."
        ) from exc
    return zarr


class ZarrBackend(IOBackend):
    file_format = FileFormat.ZARR

    @contextmanager
    def open_read(self, path: str | Path) -> Iterator[Any]:
        zarr = _zarr()
        yield zarr.open_group(str(path), mode="r")

    @contextmanager
    def open_write(self, path: str | Path) -> Iterator[Any]:
        zarr = _zarr()
        yield zarr.open_group(str(path), mode="w")

    def read_field(self, ref, spec: FieldSpec, *, key: str | None, root: str):
        base = resolve_zarr_path(ref, root)
        source = normalized_source(spec.source)
        path = PurePosixPath(normalize_path(spec.path))

        if source == "attribute":
            try:
                obj = resolve_zarr_path(base, spec.path)
            except KeyError:
                if spec.required:
                    raise
                return {}
            attrs = dict(obj.attrs)
            if spec.attribute == "*":
                return {str(k): normalize_scalar(v) for k, v in attrs.items()}
            if spec.attribute is None or spec.attribute not in attrs:
                if spec.required:
                    raise KeyError(
                        f"Zarr attribute {spec.attribute!r} not found at {spec.path!r}"
                    )
                return {}
            return {key or spec.attribute: normalize_scalar(attrs[spec.attribute])}

        if has_wildcard(spec.path):
            try:
                group = resolve_zarr_path(base, str(path.parent))
            except KeyError:
                if spec.required:
                    raise
                return {}
            out = {}
            for name in _zarr_child_names(group):
                if fnmatch(name, path.name):
                    out[name] = read_zarr_item(group[name])
            if not out and spec.required:
                raise KeyError(f"No Zarr fields match {spec.path!r}")
            return out

        try:
            item = resolve_zarr_path(base, spec.path)
        except KeyError:
            if spec.required:
                raise
            return {}
        return {key or path.name: read_zarr_item(item)}

    def write_entry(self, ref, entry: SaveEntry, *, root: str) -> None:
        base = ensure_zarr_group(ref, root)
        spec = entry.spec
        if normalized_source(spec.source) == "attribute":
            if spec.attribute is None:
                raise ValueError(f"Attribute field {entry.label!r} has no attribute name")
            target = ensure_zarr_group(base, spec.path)
            target.attrs[spec.attribute] = _zarr_storable(entry.value)
            return
        _write_zarr_value(base, spec.path, entry.value)

    def load_all(self, ref, *, root: str = "/") -> dict[str, Any]:
        base = resolve_zarr_path(ref, root)
        return zarr_to_dict(base)

    def inspect(self, ref, *, root: str = "/") -> FileStructure:
        base = resolve_zarr_path(ref, root)
        structure = FileStructure(self.file_format, root=normalize_path(root))
        _inspect_zarr_group(base, structure, logical_path="/")
        return structure

    def get_attribute(self, ref, path: str, name: str, *, root="/", default=None):
        try:
            base = resolve_zarr_path(ref, root)
            obj = resolve_zarr_path(base, path)
        except KeyError:
            return default
        attrs = dict(obj.attrs)
        return normalize_scalar(attrs[name]) if name in attrs else default

    def set_attribute(self, ref, path: str, name: str, value: Any, *, root="/"):
        base = ensure_zarr_group(ref, root)
        obj = ensure_zarr_group(base, path)
        obj.attrs[name] = _zarr_storable(value)

    def list_groups(self, ref, *, root: str = "/") -> list[str]:
        base = resolve_zarr_path(ref, root)
        return [name for name in _zarr_child_names(base) if _is_zarr_group(base[name])]

    def exists(
        self,
        ref,
        path: str,
    ) -> bool:

        path = path.strip()

        if path in ("", "/", "."):
            return True

        relative_path = path.strip("/")

        try:
            ref[relative_path]
            return True
        except KeyError:
            return False

def _is_zarr_group(obj: Any) -> bool:
    return hasattr(obj, "group_keys") or hasattr(obj, "groups")


def _is_zarr_array(obj: Any) -> bool:
    return hasattr(obj, "shape") and hasattr(obj, "dtype") and not _is_zarr_group(obj)


def _zarr_child_names(group: Any) -> list[str]:
    try:
        return list(group.keys())
    except AttributeError:
        names = [name for name, _ in group.groups()]
        names += [name for name, _ in group.arrays()]
        return names


def resolve_zarr_path(ref: Any, path: str) -> Any:
    path = normalize_path(path)
    if path == "/":
        return ref
    return ref[path.strip("/")]


def ensure_zarr_group(ref: Any, path: str) -> Any:
    path = normalize_path(path)
    if path == "/":
        return ref
    return ref.require_group(path.strip("/"))


def is_sparse_matrix_group(group: Any) -> bool:
    if not _is_zarr_group(group):
        return False
    keys = set(_zarr_child_names(group))
    return {"data", "indices", "indptr"} <= keys and (
        "shape" in dict(group.attrs) or "shape" in keys
    )


def _sparse_shape(group: Any) -> tuple[int, int]:
    attrs = dict(group.attrs)
    if "shape" in attrs:
        shape = attrs["shape"]
    elif "shape" in _zarr_child_names(group):
        shape = np.asarray(group["shape"][:])
    else:
        raise KeyError("Sparse Zarr group has no shape")
    result = tuple(int(v) for v in np.asarray(shape).ravel())
    if len(result) != 2:
        raise ValueError(f"Invalid sparse shape {result!r}")
    return result


def read_sparse_matrix(group: Any) -> sparse.spmatrix:
    attrs = dict(group.attrs)
    fmt = attrs.get("format", "csc")
    if isinstance(fmt, bytes):
        fmt = fmt.decode()
    fmt = str(fmt).lower()
    shape = _sparse_shape(group)
    args = (
        np.asarray(group["data"][:]),
        np.asarray(group["indices"][:]),
        np.asarray(group["indptr"][:]),
    )
    if fmt == "csr":
        return sparse.csr_matrix(args, shape=shape)
    if fmt == "csc":
        return sparse.csc_matrix(args, shape=shape)
    raise ValueError(f"Unsupported Zarr sparse format {fmt!r}")


def write_sparse_matrix(group: Any, matrix: sparse.spmatrix) -> None:
    if sparse.isspmatrix_csr(matrix):
        matrix = matrix.tocsr()
        fmt = "csr"
    else:
        matrix = matrix.tocsc()
        fmt = "csc"
    group.attrs["format"] = fmt
    group.attrs["shape"] = tuple(matrix.shape)
    _zarr_create_array(group, "data", matrix.data)
    _zarr_create_array(group, "indices", matrix.indices)
    _zarr_create_array(group, "indptr", matrix.indptr)


def read_zarr_item(item: Any) -> Any:
    if _is_zarr_array(item):
        value = np.asarray(item[:]) if item.shape else np.asarray(item[()])
        if value.shape == ():
            return normalize_scalar(value.item())
        return value
    if _is_zarr_group(item):
        if is_sparse_matrix_group(item):
            return read_sparse_matrix(item)
        return zarr_to_dict(item)
    raise TypeError(f"Unsupported Zarr object type: {type(item)}")


def zarr_to_dict(group: Any) -> dict[str, Any]:
    result: dict[str, Any] = {}
    attrs = dict(group.attrs)
    if attrs:
        result[ATTRIBUTES_KEY] = {str(k): normalize_scalar(v) for k, v in attrs.items()}
    for name in _zarr_child_names(group):
        result[name] = read_zarr_item(group[name])
    return result


def _zarr_create_array(group: Any, name: str, data: Any) -> None:
    if name in _zarr_child_names(group):
        del group[name]
    if hasattr(group, "create_array"):
        group.create_array(name, data=data)
    else:
        group.create_dataset(name, data=data, shape=np.shape(data), dtype=np.asarray(data).dtype)


def _write_zarr_value(base: Any, path: str, value: Any) -> None:
    path = normalize_path(path)
    pure = PurePosixPath(path)
    parent = ensure_zarr_group(base, str(pure.parent))
    name = pure.name
    if name in _zarr_child_names(parent):
        del parent[name]

    if sparse.issparse(value):
        group = parent.require_group(name)
        write_sparse_matrix(group, value)
        return
    if isinstance(value, dict):
        group = parent.require_group(name)
        for child_name, child_value in value.items():
            _write_zarr_value(group, f"/{child_name}", child_value)
        return
    _zarr_create_array(parent, name, _zarr_storable(value))


def _zarr_storable(value: Any) -> Any:
    value = normalize_scalar(value)
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, tuple):
        return np.asarray(value)
    return value


def _shape_dtype(value: Any) -> tuple[tuple[int, ...] | None, str | None]:
    arr = np.asarray(value)
    return tuple(arr.shape), str(arr.dtype)


def _inspect_zarr_group(group: Any, structure: FileStructure, *, logical_path: str) -> None:
    for attr_name, value in dict(group.attrs).items():
        shape, dtype = _shape_dtype(value)
        structure.add(
            FieldInfo(
                name=str(attr_name),
                path=normalize_path(logical_path),
                kind="attribute",
                shape=shape,
                dtype=dtype,
                attribute=str(attr_name),
            )
        )

    for name in _zarr_child_names(group):
        item = group[name]
        path = join_path(logical_path, name)
        if _is_zarr_array(item):
            structure.add(
                FieldInfo(name, path, "field", tuple(item.shape), str(item.dtype))
            )
        elif is_sparse_matrix_group(item):
            structure.add(
                FieldInfo(name, path, "field", _sparse_shape(item), str(item["data"].dtype))
            )
        elif _is_zarr_group(item):
            structure.add(FieldInfo(name, path, "group"))
            _inspect_zarr_group(item, structure, logical_path=path)
