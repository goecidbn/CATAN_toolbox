from __future__ import annotations

from collections.abc import Mapping
from contextlib import contextmanager
from fnmatch import fnmatch
from pathlib import Path, PurePosixPath
from typing import Any, Iterator

import h5py
import numpy as np
from scipy import sparse
from scipy.io import loadmat as scipy_loadmat
from scipy.io import savemat as scipy_savemat
from scipy.io import whosmat as scipy_whosmat

from catan.core.structures.load_config import FieldSpec

from .base import IOBackend
from .common import (
    nested_dict_get,
    nested_dict_group,
    nested_dict_set,
    normalize_scalar,
)
from .hdf5 import HDF5Backend
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

MAT_METADATA_KEYS = {"__header__", "__version__", "__globals__"}
MAT_ATTRS_KEY = "CATAN_attrs"


class MatPre73Backend(IOBackend):
    file_format = FileFormat.MAT_PRE73

    @contextmanager
    def open_read(self, path: str | Path) -> Iterator[dict[str, Any]]:
        data = scipy_loadmat(path, simplify_cells=True)
        clean = {k: v for k, v in data.items() if k not in MAT_METADATA_KEYS}
        yield clean

    @contextmanager
    def open_write(self, path: str | Path) -> Iterator[dict[str, Any]]:
        data: dict[str, Any] = {}
        yield data
        scipy_savemat(
            path, _prepare_mat_tree(data), do_compression=True, long_field_names=True
        )

    def read_field(
        self,
        ref: dict[str, Any],
        spec: FieldSpec,
        *,
        key: str | None,
        root: str,
    ) -> dict[str, Any]:
        base = _resolve_mapping_root(ref, root)
        source = normalized_source(spec.source)
        path = PurePosixPath(normalize_path(spec.path))

        if source == "attribute":
            try:
                target = nested_dict_get(base, spec.path)
            except (KeyError, TypeError):
                if spec.required:
                    raise
                return {}
            if not isinstance(target, Mapping):
                if spec.required:
                    raise TypeError(f"MAT path {spec.path!r} has no CATAN attributes")
                return {}
            attrs = target.get(MAT_ATTRS_KEY, {})
            if not isinstance(attrs, Mapping):
                attrs = {}
            if spec.attribute == "*":
                return {str(k): normalize_mat_value(v) for k, v in attrs.items()}
            if spec.attribute is None or spec.attribute not in attrs:
                if spec.required:
                    raise KeyError(
                        f"Attribute {spec.attribute!r} not found at {spec.path!r}"
                    )
                return {}
            return {key or spec.attribute: normalize_mat_value(attrs[spec.attribute])}

        if has_wildcard(spec.path):
            try:
                parent = nested_dict_get(base, str(path.parent))
            except (KeyError, TypeError):
                if spec.required:
                    raise
                return {}
            if not isinstance(parent, Mapping):
                if spec.required:
                    raise TypeError(f"MAT path {path.parent!s} is not a struct")
                return {}
            return {
                str(name): normalize_mat_value(value)
                for name, value in parent.items()
                if name != MAT_ATTRS_KEY and fnmatch(str(name), path.name)
            }

        try:
            value = nested_dict_get(base, spec.path)
        except (KeyError, TypeError):
            if spec.required:
                raise
            return {}
        return {key or path.name: normalize_mat_value(value)}

    def write_entry(self, ref: dict[str, Any], entry: SaveEntry, *, root: str) -> None:
        base = _ensure_mapping_root(ref, root)
        spec = entry.spec
        if normalized_source(spec.source) == "attribute":
            if spec.attribute is None:
                raise ValueError(
                    f"Attribute field {entry.label!r} has no attribute name"
                )
            target = nested_dict_group(base, spec.path)
            attrs = target.setdefault(MAT_ATTRS_KEY, {})
            if not isinstance(attrs, dict):
                raise TypeError(f"Reserved {MAT_ATTRS_KEY!r} is not a mapping")
            attrs[spec.attribute] = _mat_storable(entry.value)
            return
        nested_dict_set(base, spec.path, _mat_storable(entry.value))

    def load_all(self, ref: dict[str, Any], *, root: str = "/") -> dict[str, Any]:
        base = _resolve_mapping_root(ref, root)
        if not isinstance(base, Mapping):
            raise TypeError(f"Logical MAT root {root!r} is not a struct")
        return _mat_to_dict(base)

    def inspect(self, ref: dict[str, Any], *, root: str = "/") -> FileStructure:
        base = _resolve_mapping_root(ref, root)
        if not isinstance(base, Mapping):
            raise TypeError(f"Logical MAT root {root!r} is not a struct")
        structure = FileStructure(self.file_format, root=normalize_path(root))
        _inspect_mapping(base, structure, logical_path="/")
        return structure

    def inspect_file(
        self,
        path: str | Path,
        *,
        root: str = "/",
    ) -> FileStructure:

        if normalize_path(root) != "/":
            # fallback for now
            return super().inspect_file(
                path,
                root=root,
            )

        structure = FileStructure(
            self.file_format,
            root="/",
        )

        for name, shape, mat_class in scipy_whosmat(path):
            structure.add(
                FieldInfo(
                    name=name,
                    path=join_path("/", name),
                    kind="group" if mat_class == "struct" else "field",
                    shape=tuple(shape),
                    dtype=mat_class,
                )
            )

        return structure

    def get_attribute(
        self,
        ref: dict[str, Any],
        path: str,
        name: str,
        *,
        root: str = "/",
        default: Any = None,
    ) -> Any:
        try:
            base = _resolve_mapping_root(ref, root)
            target = nested_dict_get(base, path)
        except (KeyError, TypeError):
            return default
        if not isinstance(target, Mapping):
            return default
        attrs = target.get(MAT_ATTRS_KEY, {})
        if not isinstance(attrs, Mapping):
            return default
        return normalize_mat_value(attrs.get(name, default))

    def set_attribute(
        self,
        ref: dict[str, Any],
        path: str,
        name: str,
        value: Any,
        *,
        root: str = "/",
    ) -> None:
        base = _ensure_mapping_root(ref, root)
        target = nested_dict_group(base, path)
        attrs = target.setdefault(MAT_ATTRS_KEY, {})
        attrs[name] = _mat_storable(value)

    def list_groups(self, ref: dict[str, Any], *, root: str = "/") -> list[str]:
        base = _resolve_mapping_root(ref, root)
        if not isinstance(base, Mapping):
            return []
        return [
            str(name)
            for name, value in base.items()
            if name != MAT_ATTRS_KEY and isinstance(value, Mapping)
        ]

    def exists(
        self,
        ref: dict,
        path: str,
    ) -> bool:

        try:
            resolve_mat_path(ref, path)
            return True

        except (KeyError, TypeError):
            return False


class Mat73Backend(HDF5Backend):
    """
    MATLAB >= 7.3 backend.

    Reading and inspection use the normal HDF5 backend because MATLAB
    v7.3 files are HDF5 containers.

    Writing uses ``hdf5storage`` in order to create genuine
    MATLAB-compatible v7.3 MAT files.

    If MATLAB-compatible writing is not required, this class could
    simply inherit HDF5Backend without any further overrides.
    """

    file_format = FileFormat.MAT73

    def __init__(self) -> None:
        super().__init__()

        # Only needed for writing to the intermediate dict used by
        # hdf5storage.
        self._mapping = MatPre73Backend()

    @staticmethod
    def _hdf5storage():
        try:
            import hdf5storage
        except ImportError:
            return None

        return hdf5storage

    # ============================================================
    # Reading
    # ============================================================

    # No overrides needed!
    #
    # These are inherited directly from HDF5Backend:
    #
    # open_read()
    # read_field()
    # load_all()
    # inspect()
    # get_attribute()
    # list_groups()
    # exists()
    #
    # All of these operate directly on the underlying HDF5 tree.

    # ============================================================
    # Writing
    # ============================================================

    @contextmanager
    def open_write(
        self,
        path: str | Path,
    ):
        """
        Build an intermediate Python mapping and write it as a genuine
        MATLAB v7.3 file using hdf5storage when the context exits.
        """

        hdf5storage = self._hdf5storage()

        if hdf5storage is None:
            raise ImportError(
                "Writing MATLAB v7.3 files requires the optional "
                "dependency 'hdf5storage'."
            )

        data: dict[str, Any] = {}

        yield data

        hdf5storage.savemat(
            str(path),
            _prepare_mat_tree(data),
            format="7.3",
            matlab_compatible=True,
            store_python_metadata=False,
        )

    def write_entry(
        self,
        ref,
        entry,
        *,
        root: str,
    ):
        """
        Write into the intermediate mapping used by hdf5storage.
        """

        if not isinstance(ref, dict):
            raise TypeError(
                "MATLAB v7.3 writing expects a dictionary-backed "
                "intermediate representation."
            )

        return self._mapping.write_entry(
            ref,
            entry,
            root=root,
        )

    def set_attribute(
        self,
        ref,
        path: str,
        name: str,
        value,
        *,
        root: str = "/",
    ):
        """
        Store CATAN logical attributes in the MATLAB mapping.

        The mapping backend is responsible for translating these into
        the reserved MATLAB-side representation.
        """

        if not isinstance(ref, dict):
            raise TypeError(
                "MATLAB v7.3 writing expects a dictionary-backed "
                "intermediate representation."
            )

        return self._mapping.set_attribute(
            ref,
            path,
            name,
            value,
            root=root,
        )


def is_mat73(path: str | Path) -> bool:
    return h5py.is_hdf5(path)


def normalize_mat_value(value: Any) -> Any:
    if sparse.issparse(value):
        return value
    if isinstance(value, Mapping):
        return _mat_to_dict(value)
    if isinstance(value, list):
        return [normalize_mat_value(v) for v in value]
    if isinstance(value, tuple):
        return tuple(normalize_mat_value(v) for v in value)
    if isinstance(value, np.ndarray) and value.dtype == object:
        return np.array(
            [normalize_mat_value(v) for v in value.flat], dtype=object
        ).reshape(value.shape)
    return normalize_scalar(value)


def _mat_to_dict(obj: Any) -> Any:
    if isinstance(obj, Mapping):
        return {
            str(k): _mat_to_dict(v)
            for k, v in obj.items()
            if k not in MAT_METADATA_KEYS
        }
    if isinstance(obj, list):
        return [_mat_to_dict(v) for v in obj]
    if isinstance(obj, tuple):
        return tuple(_mat_to_dict(v) for v in obj)
    if isinstance(obj, np.ndarray) and obj.dtype == object:
        return np.array([_mat_to_dict(v) for v in obj.flat], dtype=object).reshape(
            obj.shape
        )
    return normalize_mat_value(obj)


def _mat_storable(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, tuple):
        return np.asarray(value)
    return value


def _prepare_mat_tree(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {str(k): _prepare_mat_tree(v) for k, v in obj.items()}
    if isinstance(obj, Path):
        return str(obj)
    if obj is None:
        return np.array([])
    return obj


def _resolve_mapping_root(ref: Mapping[str, Any], root: str) -> Any:
    return nested_dict_get(ref, root)


def _ensure_mapping_root(ref: dict[str, Any], root: str) -> dict[str, Any]:
    return nested_dict_group(ref, root)


def _shape_dtype(value: Any) -> tuple[tuple[int, ...] | None, str | None]:
    if sparse.issparse(value):
        return tuple(value.shape), str(value.dtype)
    if isinstance(value, np.ndarray):
        return tuple(value.shape), str(value.dtype)
    if isinstance(value, np.generic):
        return (), str(value.dtype)
    if np.isscalar(value):
        arr = np.asarray(value)
        return (), str(arr.dtype)
    return None, type(value).__name__


def _inspect_mapping(
    mapping: Mapping[str, Any],
    structure: FileStructure,
    *,
    logical_path: str,
) -> None:
    attrs = mapping.get(MAT_ATTRS_KEY, {})
    if isinstance(attrs, Mapping):
        for attr_name, value in attrs.items():
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

    for name, value in mapping.items():
        if name == MAT_ATTRS_KEY:
            continue
        path = join_path(logical_path, str(name))
        if isinstance(value, Mapping):
            structure.add(FieldInfo(str(name), path, "group"))
            _inspect_mapping(value, structure, logical_path=path)
        else:
            shape, dtype = _shape_dtype(value)
            structure.add(
                FieldInfo(
                    name=str(name),
                    path=path,
                    kind="field",
                    shape=shape,
                    dtype=dtype,
                )
            )


def resolve_mat_path(
    ref: dict[str, Any],
    path: str,
) -> Any:
    """
    Resolve a CATAN-style logical path inside a nested MATLAB dict.

    Examples
    --------
    "/" -> ref
    "/estimates/A" -> ref["estimates"]["A"]
    """

    path = path.strip()

    if path in ("", "/", "."):
        return ref

    current: Any = ref

    for part in PurePosixPath(path).parts:
        if part == "/":
            continue

        if not isinstance(current, dict):
            raise TypeError(
                f"Cannot resolve {path!r}: "
                f"{part!r} is below a non-dict object "
                f"of type {type(current).__name__}."
            )

        current = current[part]

    return current
