from __future__ import annotations

from fnmatch import fnmatch
from pathlib import PurePosixPath
from typing import Any

import h5py
import numpy as np
from scipy import sparse
from scipy.io import loadmat

from catan.core.structures.load_config import FieldSpec
from .hdf5 import load_hdf5


MAT_METADATA_KEYS = {
    "__header__",
    "__version__",
    "__globals__",
}


def load_mat(
    path: str,
    fields_to_load: dict[str, dict[str, FieldSpec]] | None = None,
) -> dict[str, Any]:
    """
    Load a MATLAB file.

    MATLAB v7.3 files are HDF5 files and are delegated to the HDF5
    loader. Earlier MAT versions are read using scipy.io.loadmat().

    Parameters
    ----------
    path
        Path to the MAT file.

    fields_to_load
        Same FieldSpec-based configuration used by the other loaders.

        If None, the complete MAT file is converted to a nested dict.
    """

    # MATLAB v7.3 uses HDF5 internally.
    if h5py.is_hdf5(path):
        with h5py.File(path, "r") as h5ref:
            return load_hdf5(
                h5ref,
                fields_to_load=fields_to_load,
            )

    # simplify_cells gives much friendlier Python objects for
    # MATLAB structs/cells than the default loadmat representation.
    mat = loadmat(
        path,
        simplify_cells=True,
    )

    mat = {
        key: value
        for key, value in mat.items()
        if key not in MAT_METADATA_KEYS
    }

    if fields_to_load is None:
        return mat_to_dict(mat)

    result = {}

    for group_name, fields in fields_to_load.items():
        result[group_name] = fields_from_mat(
            mat,
            fields,
        )

    return result

def fields_from_mat(
    mat: dict[str, Any],
    fields: dict[str, FieldSpec],
) -> dict[str, Any]:

    data = {}

    for key, spec in fields.items():
        output = read_mat_field(
            mat,
            spec,
            key=key,
        )

        data.update(output)

    return data

def read_mat_field(
    mat: dict[str, Any],
    spec: FieldSpec,
    key: str | None = None,
) -> dict[str, Any]:
    """
    Read one configured field from a classic MATLAB file.

    Paths use the same slash-separated convention as the HDF5 loader:

        /A
        /estimates/C
        /quality/*

    Here nested path elements correspond to MATLAB structs/dicts.
    """

    # Classic MAT files don't have HDF5 attributes.
    if spec.source == "attribute":
        if spec.required:
            raise ValueError(
                f"Field {key!r} requests an HDF5 attribute, "
                "but classic MAT files do not provide HDF5 attributes."
            )

        return {}

    path = PurePosixPath(spec.path)

    # ------------------------------------------------------------
    # Wildcard, e.g. /quality/*
    # ------------------------------------------------------------

    if "*" in path.name:

        try:
            parent = resolve_mat_path(
                mat,
                str(path.parent),
            )

        except (KeyError, TypeError):
            if spec.required:
                raise
            return {}

        if not isinstance(parent, dict):
            if spec.required:
                raise TypeError(
                    f"MAT path {path.parent!s} does not resolve "
                    "to a struct/dictionary."
                )
            return {}

        return {
            name: normalize_mat_value(value)
            for name, value in parent.items()
            if fnmatch(name, path.name)
        }

    # ------------------------------------------------------------
    # Explicit field
    # ------------------------------------------------------------

    try:
        value = resolve_mat_path(
            mat,
            spec.path,
        )

    except (KeyError, TypeError):
        if spec.required:
            raise
        return {}

    output_key = (
        key
        if key is not None
        else path.name
    )

    return {
        output_key: normalize_mat_value(value)
    }


def resolve_mat_path(
    mat: dict[str, Any],
    path: str,
) -> Any:
    """
    Resolve a CATAN-style path in a MATLAB dictionary.

    "/" or "." means the supplied root dictionary.
    """

    path = path.strip()

    if path in ("", "/", "."):
        return mat

    current: Any = mat

    for part in PurePosixPath(path).parts:
        if part == "/":
            continue

        if not isinstance(current, dict):
            raise TypeError(
                f"Cannot enter {part!r}; parent object "
                f"is {type(current).__name__}, not a MATLAB struct."
            )

        current = current[part]

    return current

def mat_to_dict(
    obj: Any,
) -> Any:
    """
    Recursively normalize MATLAB objects into ordinary Python
    dictionaries / lists / NumPy arrays / sparse matrices.
    """

    if isinstance(obj, dict):
        return {
            key: mat_to_dict(value)
            for key, value in obj.items()
            if key not in MAT_METADATA_KEYS
        }

    if isinstance(obj, list):
        return [
            mat_to_dict(value)
            for value in obj
        ]

    if isinstance(obj, tuple):
        return tuple(
            mat_to_dict(value)
            for value in obj
        )

    if isinstance(obj, np.ndarray) and obj.dtype == object:
        return np.array(
            [
                mat_to_dict(value)
                for value in obj.flat
            ],
            dtype=object,
        ).reshape(obj.shape)

    return normalize_mat_value(obj)


def normalize_mat_value(
    value: Any,
) -> Any:

    # Leave SciPy sparse matrices intact
    if sparse.issparse(value):
        return value

    # Convert NumPy scalar -> Python scalar
    if isinstance(value, np.generic):
        return value.item()

    return value














# def loadmat(filename):
#     """
#     this function should be called instead of direct spio.loadmat
#     as it cures the problem of not properly recovering python dictionaries
#     from mat files. It calls the function check keys to cure all entries
#     which are still mat-objects
#     """
#     data = spio.loadmat(filename, struct_as_record=False, squeeze_me=True)

#     ### get rid of some unnecessary entries
#     for key in ["__header__", "__version__", "__globals__"]:
#         del data[key]

#     return _check_keys(data)


# def _check_keys(dict):
#     """
#     checks if entries in dictionary are mat-objects. If yes
#     todict is called to change them to nested dictionaries
#     """
#     for key in dict:
#         if isinstance(dict[key], spio.matlab.mio5_params.mat_struct):
#             dict[key] = _todict(dict[key])
#     return dict


# def _todict(matobj):
#     """
#     A recursive function which constructs from matobjects nested dictionaries
#     """
#     dict = {}
#     for strg in matobj._fieldnames:
#         elem = matobj.__dict__[strg]
#         if isinstance(elem, spio.matlab.mio5_params.mat_struct):
#             dict[strg] = _todict(elem)
#         else:
#             dict[strg] = elem
#     return dict