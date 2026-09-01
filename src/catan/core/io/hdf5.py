from __future__ import annotations

from contextlib import contextmanager
from fnmatch import fnmatch
from pathlib import Path, PurePosixPath
from typing import Any, Iterator

import h5py
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


class HDF5Backend(IOBackend):
    file_format = FileFormat.HDF5

    @contextmanager
    def open_read(self, path: str | Path) -> Iterator[h5py.File]:
        with h5py.File(path, "r") as ref:
            yield ref

    @contextmanager
    def open_write(self, path: str | Path) -> Iterator[h5py.File]:
        with h5py.File(path, "w") as ref:
            yield ref

    def read_field(
        self,
        ref: h5py.File | h5py.Group,
        spec: FieldSpec,
        *,
        key: str | None,
        root: str,
    ) -> dict[str, Any]:
        base = resolve_hdf5_path(ref, root)
        source = normalized_source(spec.source)
        path = PurePosixPath(normalize_path(spec.path))

        if source == "attribute":
            try:
                obj = resolve_hdf5_path(base, spec.path)
            except KeyError:
                if spec.required:
                    raise
                return {}

            if spec.attribute == "*":
                return {
                    name: normalize_scalar(value)
                    for name, value in obj.attrs.items()
                }

            if spec.attribute is None or spec.attribute not in obj.attrs:
                if spec.required:
                    raise KeyError(
                        f"Attribute {spec.attribute!r} not found at {spec.path!r}"
                    )
                return {}

            return {
                key or spec.attribute: normalize_scalar(obj.attrs[spec.attribute])
            }

        if has_wildcard(spec.path):
            parent_path = str(path.parent)
            try:
                group = resolve_hdf5_path(base, parent_path)
            except KeyError:
                if spec.required:
                    raise
                return {}
            if not isinstance(group, h5py.Group):
                if spec.required:
                    raise TypeError(f"{parent_path!r} is not an HDF5 group")
                return {}
            return {
                name: read_hdf5_item(item)
                for name, item in group.items()
                if fnmatch(name, path.name)
            }

        try:
            item = resolve_hdf5_path(base, spec.path)
        except KeyError:
            if spec.required:
                raise
            return {}

        return {key or path.name: read_hdf5_item(item)}

    def write_entry(
        self,
        ref: h5py.File | h5py.Group,
        entry: SaveEntry,
        *,
        root: str,
    ) -> None:
        base = ensure_hdf5_group(ref, root)
        spec = entry.spec

        if normalized_source(spec.source) == "attribute":
            if spec.attribute is None:
                raise ValueError(f"Attribute field {entry.label!r} has no attribute name")
            target = ensure_hdf5_group(base, spec.path)
            target.attrs[spec.attribute] = _hdf5_storable(entry.value)
            return

        _write_hdf5_value(base, spec.path, entry.value)

    def load_all(
        self,
        ref: h5py.File | h5py.Group,
        *,
        root: str = "/",
    ) -> dict[str, Any]:
        base = resolve_hdf5_path(ref, root)
        if not isinstance(base, (h5py.File, h5py.Group)):
            raise TypeError(f"Logical root {root!r} is not an HDF5 group")
        return hdf5_to_dict(base)

    def inspect(
        self,
        ref: h5py.File | h5py.Group,
        *,
        root: str = "/",
    ) -> FileStructure:
        base = resolve_hdf5_path(ref, root)
        if not isinstance(base, (h5py.File, h5py.Group)):
            raise TypeError(f"Logical root {root!r} is not an HDF5 group")
        structure = FileStructure(self.file_format, root=normalize_path(root))
        _inspect_hdf5_group(base, structure, logical_path="/")
        return structure

    def get_attribute(
        self,
        ref: h5py.File | h5py.Group,
        path: str,
        name: str,
        *,
        root: str = "/",
        default: Any = None,
    ) -> Any:
        try:
            base = resolve_hdf5_path(ref, root)
            obj = resolve_hdf5_path(base, path)
        except KeyError:
            return default
        return normalize_scalar(obj.attrs[name]) if name in obj.attrs else default

    def set_attribute(
        self,
        ref: h5py.File | h5py.Group,
        path: str,
        name: str,
        value: Any,
        *,
        root: str = "/",
    ) -> None:
        base = ensure_hdf5_group(ref, root)
        obj = ensure_hdf5_group(base, path)
        obj.attrs[name] = _hdf5_storable(value)

    def list_groups(
        self,
        ref: h5py.File | h5py.Group,
        *,
        root: str = "/",
    ) -> list[str]:
        base = resolve_hdf5_path(ref, root)
        if not isinstance(base, (h5py.File, h5py.Group)):
            return []
        return [name for name, obj in base.items() if isinstance(obj, h5py.Group)]

    def exists(
        self,
        ref: h5py.File | h5py.Group,
        path: str,
    ) -> bool:

        path = path.strip()

        if path in ("", "/", "."):
            return True

        relative_path = path.strip("/")

        return relative_path in ref

def resolve_hdf5_path(
    ref: h5py.File | h5py.Group,
    path: str,
) -> h5py.File | h5py.Group | h5py.Dataset:
    path = normalize_path(path)
    if path == "/":
        return ref
    return ref[path.strip("/")]


def ensure_hdf5_group(
    ref: h5py.File | h5py.Group,
    path: str,
) -> h5py.File | h5py.Group:
    path = normalize_path(path)
    if path == "/":
        return ref
    return ref.require_group(path.strip("/"))


def is_sparse_matrix_group(group: h5py.Group) -> bool:
    return all(key in group for key in ("data", "indices", "indptr")) and (
        "shape" in group or "shape" in group.attrs
    )


def _sparse_shape(group: h5py.Group) -> tuple[int, int]:
    if "shape" in group:
        shape = group["shape"][()]
    elif "shape" in group.attrs:
        shape = group.attrs["shape"]
    else:
        raise KeyError(f"Sparse matrix group {group.name!r} has no shape")
    values = tuple(int(v) for v in np.asarray(shape).ravel())
    if len(values) != 2:
        raise ValueError(f"Invalid sparse shape {values!r}")
    return values


def read_sparse_matrix(group: h5py.Group) -> sparse.spmatrix:
    shape = _sparse_shape(group)
    fmt = group.attrs.get("format")
    if isinstance(fmt, bytes):
        fmt = fmt.decode("utf-8")
    if fmt is not None:
        fmt = str(fmt).lower()

    if fmt not in (None, "csc", "csr"):
        raise ValueError(f"Unsupported sparse matrix format: {fmt!r}")

    if fmt is None:
        n_indptr = len(group["indptr"])
        csr_possible = n_indptr == shape[0] + 1
        csc_possible = n_indptr == shape[1] + 1
        if csr_possible and not csc_possible:
            fmt = "csr"
        elif csc_possible and not csr_possible:
            fmt = "csc"
        else:
            # Old CaImAn/CATAN files are CSC and square matrices are
            # structurally ambiguous without explicit metadata.
            fmt = "csc"

    args = (
        group["data"][()],
        group["indices"][()],
        group["indptr"][()],
    )
    if fmt == "csr":
        return sparse.csr_matrix(args, shape=shape)
    return sparse.csc_matrix(args, shape=shape)


def write_sparse_matrix(group: h5py.Group, matrix: sparse.spmatrix) -> None:
    if sparse.isspmatrix_csr(matrix):
        matrix = matrix.tocsr()
        fmt = "csr"
    else:
        matrix = matrix.tocsc()
        fmt = "csc"

    group.attrs["format"] = fmt
    group.attrs["shape"] = matrix.shape
    group.create_dataset("data", data=matrix.data)
    group.create_dataset("indices", data=matrix.indices)
    group.create_dataset("indptr", data=matrix.indptr)


def read_hdf5_item(item: h5py.Dataset | h5py.Group) -> Any:
    if isinstance(item, h5py.Dataset):
        return _decode_hdf5_value(item[()])
    if isinstance(item, h5py.Group):
        if is_sparse_matrix_group(item):
            return read_sparse_matrix(item)
        return hdf5_to_dict(item)
    raise TypeError(f"Unsupported HDF5 object type: {type(item)}")


def hdf5_to_dict(group: h5py.File | h5py.Group) -> dict[str, Any]:
    data: dict[str, Any] = {}
    if group.attrs:
        data[ATTRIBUTES_KEY] = {
            key: normalize_scalar(value) for key, value in group.attrs.items()
        }
    for name, item in group.items():
        data[name] = read_hdf5_item(item)
    return data


def _write_hdf5_value(
    base: h5py.File | h5py.Group,
    path: str,
    value: Any,
) -> None:
    path = normalize_path(path)
    if path == "/":
        raise ValueError("Cannot write a dataset directly to logical root '/'")

    pure = PurePosixPath(path)
    parent_path = str(pure.parent)
    parent = ensure_hdf5_group(base, parent_path)
    name = pure.name

    if name in parent:
        del parent[name]

    if sparse.issparse(value):
        group = parent.create_group(name)
        write_sparse_matrix(group, value)
        return

    value = _hdf5_storable(value)
    if isinstance(value, dict):
        group = parent.create_group(name)
        for child_name, child_value in value.items():
            _write_hdf5_value(group, f"/{child_name}", child_value)
        return

    parent.create_dataset(name, data=value)


def _hdf5_storable(value: Any) -> Any:
    value = normalize_scalar(value)
    if isinstance(value, str):
        return value
    if isinstance(value, (list, tuple)):
        # Convert homogeneous path/string collections explicitly; numpy/h5py
        # handles numeric collections naturally.
        if all(isinstance(v, (str, Path)) for v in value):
            return np.asarray([str(v) for v in value], dtype=h5py.string_dtype("utf-8"))
        return np.asarray(value)
    return value


def _decode_hdf5_value(value: Any) -> Any:
    if isinstance(value, bytes):
        if value == b"NoneType":
            return None
        return value.decode("utf-8")
    if isinstance(value, np.ndarray) and value.dtype.kind == "S":
        return value.astype(str)
    return normalize_scalar(value)


def _inspect_hdf5_group(
    group: h5py.File | h5py.Group,
    structure: FileStructure,
    *,
    logical_path: str,
) -> None:
    for attr_name, value in group.attrs.items():
        arr = np.asarray(value)
        structure.add(
            FieldInfo(
                name=attr_name,
                path=normalize_path(logical_path),
                kind="attribute",
                shape=tuple(arr.shape) if arr.shape else (),
                dtype=str(arr.dtype),
                attribute=attr_name,
            )
        )

    for name, item in group.items():
        path = join_path(logical_path, name)
        if isinstance(item, h5py.Dataset):
            structure.add(
                FieldInfo(
                    name=name,
                    path=path,
                    kind="field",
                    shape=tuple(item.shape),
                    dtype=str(item.dtype),
                )
            )
            for attr_name, value in item.attrs.items():
                arr = np.asarray(value)
                structure.add(
                    FieldInfo(
                        name=attr_name,
                        path=path,
                        kind="attribute",
                        shape=tuple(arr.shape) if arr.shape else (),
                        dtype=str(arr.dtype),
                        attribute=attr_name,
                    )
                )
            continue

        if is_sparse_matrix_group(item):
            matrix_shape = _sparse_shape(item)
            structure.add(
                FieldInfo(
                    name=name,
                    path=path,
                    kind="field",
                    shape=matrix_shape,
                    dtype=str(item["data"].dtype),
                )
            )
            continue

        structure.add(FieldInfo(name=name, path=path, kind="group"))
        _inspect_hdf5_group(item, structure, logical_path=path)
