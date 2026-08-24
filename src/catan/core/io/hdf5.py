
from pathlib import PurePosixPath
from typing import Any

import h5py
import numpy as np
from scipy import sparse

from catan.core.structures.load_config import FieldSpec

ATTRIBUTES_KEY = "__attrs__"

def load_hdf5(
    h5ref: h5py.File | h5py.Group,
    fields_to_load: dict[str, dict[str, FieldSpec]] | None = None,
) -> dict[str, Any]:
    """
    Load data from an HDF5 File or Group.

    Parameters
    ----------
    h5ref
        HDF5 File or Group acting as the logical root.

    fields_to_load
        Config-driven field specification.

        Structure:
            {
                "spatial": {
                    "footprints": FieldSpec(...),
                    ...
                },
                "quality": {
                    "*": FieldSpec(...),
                },
            }

        If None, the complete HDF5 subtree is loaded recursively.

    Returns
    -------
    dict
        Loaded data.
    """

    if fields_to_load is None:
        ## Load the entire HDF5 subtree recursively
        return hdf5_to_dict(h5ref)

    ## Load only the specified fields
    result = {}
    for group_name, fields in fields_to_load.items():
        result[group_name] = fields_from_hdf5(
            h5ref,
            fields,
        )

    return result

def fields_from_hdf5(
    h5ref: h5py.File | h5py.Group,
    fields: dict[str, FieldSpec],
) -> dict[str, Any]:

    data = {}

    for key, spec in fields.items():

        output = read_hdf5_field(
            h5ref,
            spec,
            key=key,
        )

        data.update(output)

    return data

def read_hdf5_field(
    h5ref: h5py.File | h5py.Group,
    spec: FieldSpec,
    key: str | None = None,
) -> dict[str, Any]:

    path = PurePosixPath(spec.path)

    # ============================================================
    # Attributes
    # ============================================================

    if spec.source == "attribute":

        try:
            obj = resolve_hdf5_path(
                h5ref,
                spec.path,
            )

        except KeyError:
            if spec.required:
                raise

            return {}

        # Load all attributes
        if spec.attribute == "*":
            return {
                name: normalize_hdf5_value(value)
                for name, value in obj.attrs.items()
            }

        if spec.attribute not in obj.attrs:
            if spec.required:
                raise KeyError(
                    f"Attribute {spec.attribute!r} "
                    f"not found at {spec.path!r}"
                )

            return {}

        output_key = (
            key
            if key is not None
            else spec.attribute
        )

        return {
            output_key:
                normalize_hdf5_value(
                    obj.attrs[spec.attribute]
                )
        }

    # ============================================================
    # Dataset wildcard:
    #
    #     /quality/*
    # ============================================================

    if path.name == "*":

        parent_path = str(path.parent)

        try:
            group = resolve_hdf5_path(
                h5ref,
                parent_path,
            )

        except KeyError:
            if spec.required:
                raise

            return {}

        if not isinstance(group, h5py.Group):
            if spec.required:
                raise TypeError(
                    f"{parent_path!r} is not an HDF5 group."
                )

            return {}

        return {
            name: read_hdf5_item(item)
            for name, item in group.items()
        }

    # ============================================================
    # Explicit field
    # ============================================================

    try:
        item = resolve_hdf5_path(
            h5ref,
            spec.path,
        )

    except KeyError:
        if spec.required:
            raise

        return {}

    output_key = (
        key
        if key is not None
        else path.name
    )

    return {
        output_key: read_hdf5_item(item)
    }

def read_hdf5_item(
    item: h5py.Dataset | h5py.Group,
) -> Any:

    if isinstance(item, h5py.Dataset):
        return normalize_hdf5_value(
            item[()]
        )

    if isinstance(item, h5py.Group):

        if is_sparse_matrix_group(item):
            return read_sparse_matrix(item)

        return hdf5_to_dict(item)

    raise TypeError(
        f"Unsupported HDF5 object type: {type(item)}"
    )

def is_sparse_matrix_group(
    group: h5py.Group,
) -> bool:

    return all(
        key in group
        for key in (
            "data",
            "indices",
            "indptr",
        )
    )

def hdf5_to_dict(
    h5ref: h5py.File | h5py.Group,
    *,
    include_attributes: bool = True,
) -> dict[str, Any]:
    """
    Recursively convert an HDF5 File/Group into nested dictionaries.

    Attributes are stored under the reserved "__attrs__" key.
    """

    data = {}

    # ------------------------------------------------------------
    # Attributes belonging to this File/Group
    # ------------------------------------------------------------

    if include_attributes and h5ref.attrs:
        data[ATTRIBUTES_KEY] = {
            key: normalize_hdf5_value(value)
            for key, value in h5ref.attrs.items()
        }

    # ------------------------------------------------------------
    # Children
    # ------------------------------------------------------------

    for name, item in h5ref.items():

        if isinstance(item, h5py.Dataset):

            data[name] = normalize_hdf5_value(
                item[()]
            )

        elif isinstance(item, h5py.Group):

            if is_sparse_matrix_group(item):
                data[name] = read_sparse_matrix(
                    item
                )

            else:
                data[name] = hdf5_to_dict(
                    item,
                    include_attributes=include_attributes,
                )

    return data



### Helper functions ###
def read_sparse_matrix(h5ref: h5py.Group) -> sparse.csc_matrix:
    """Read a CSC matrix from an HDF5 group."""
    matrix_format = h5ref.attrs.get("format", "csc")

    if isinstance(matrix_format, bytes):
        matrix_format = matrix_format.decode("utf-8")

    if matrix_format != "csc":
        raise ValueError(f"Unsupported sparse matrix format: {matrix_format!r}")

    if "shape" in h5ref.keys():
        # ensure consistency with CaImAn style saving
        shape = tuple(h5ref["shape"][()])
    else:
        shape = tuple(int(value) for value in h5ref.attrs["shape"])

    return sparse.csc_matrix(
        (
            h5ref["data"][()],
            h5ref["indices"][()],
            h5ref["indptr"][()],
        ),
        shape=shape,
    )

def resolve_hdf5_path(
    h5ref: h5py.File | h5py.Group,
    path: str,
):
    """
    Resolve a CATAN HDF5 path relative to `h5ref`.

    "/" or "." means `h5ref` itself.
    Leading "/" does NOT mean physical HDF5 file root.
    """

    path = path.strip()

    if path in ("", "/", "."):
        return h5ref

    relative_path = path.strip("/")

    return h5ref[relative_path]


def normalize_hdf5_value(
    value,
):
    if isinstance(value, bytes):
        return value.decode("utf-8")

    if isinstance(value, np.generic):
        return value.item()

    return value


### -------------------------------------------------------- ###
### --------------------- load helpers --------------------- ###
### -------------------------------------------------------- ###


def decode_hdf5_value(value):
    if value == b"NoneType":
        return None

    if isinstance(value, bytes):
        return value.decode()

    return value




def write_sparse_matrix(
    h5ref: h5py.File | h5py.Group,
    matrix: sparse.spmatrix,
) -> None:
    """Write a SciPy sparse matrix into an HDF5 group."""
    matrix = sparse.csc_matrix(matrix)

    h5ref.attrs["format"] = "csc"
    h5ref.attrs["shape"] = matrix.shape

    h5ref.create_dataset("data", data=matrix.data)
    h5ref.create_dataset("indices", data=matrix.indices)
    h5ref.create_dataset("indptr", data=matrix.indptr)

def write_optional_array(
    h5ref: h5py.File | h5py.Group,
    name: str,
    value: np.ndarray | None,
    **dataset_kwargs: Any,
):
    if value is not None:
        h5ref.create_dataset(name, data=value, **dataset_kwargs)

def write_optional_attr(
    h5ref: h5py.File | h5py.Group,
    name: str,
    value: Any | None,
):
    if value is not None:
        h5ref.attrs[name] = value



def read_optional_array(
    h5ref: h5py.File | h5py.Group,
    name: str,
) -> Any:
    if name not in h5ref:
        return None

    item = h5ref[name]
    if isinstance(item, h5py.Dataset):
        return item[()]
    else:
        raise ValueError(f"Expected {name} to be a dataset, but found {type(item)}")

def read_optional_attr(
    h5ref: h5py.File | h5py.Group,
    name: str,
    default: Any = None,
) -> Any:
    return h5ref.attrs[name] if name in h5ref.attrs else default
