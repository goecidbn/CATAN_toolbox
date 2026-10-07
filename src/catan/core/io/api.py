from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Iterable, Literal, Sequence

from catan.core.structures.load_config import FieldSpec, LoadConfig

from .base import IOBackend
from .detection import detect_file_format
from .hdf5 import HDF5Backend
from .matlab import Mat73Backend, MatPre73Backend
from .npz import NPZBackend
from .types import FileFormat
from .zarr import ZarrBackend
from .image import ImageBackend
from .common import get_data_value

_HDF5 = HDF5Backend()
_MAT_PRE73 = MatPre73Backend()
_MAT73 = Mat73Backend()
_NPZ = NPZBackend()
_ZARR = ZarrBackend()
_IMAGE = ImageBackend()

NATIVE_SESSION_OBJECT_TYPES = {"SessionData", "SessionList"}

NATIVE_SESSION_CONFIG = "catan_session.json"
NATIVE_REMAP_CONFIG = "catan_remap.json"
NATIVE_MODEL_CONFIG = "catan_model.json"
NATIVE_ASSIGNMENTS_CONFIG = "catan_assignments.json"


def get_backend(
    path: str | Path,
    *,
    for_write: bool = False,
    mat_version: Literal["pre73", "7.3"] = "7.3",
) -> IOBackend:
    fmt = detect_file_format(path, for_write=for_write, mat_version=mat_version)
    return {
        FileFormat.HDF5: _HDF5,
        FileFormat.MAT_PRE73: _MAT_PRE73,
        FileFormat.MAT73: _MAT73,
        FileFormat.NPZ: _NPZ,
        FileFormat.ZARR: _ZARR,
        FileFormat.IMAGE: _IMAGE,
    }[fmt]


def load_file(
    path: str | Path,
    fields_to_load: dict[str, dict[str, FieldSpec]] | None = None,
    config_name: str | None = None,
    *,
    root: str = "/",
) -> dict[str, Any]:
    """Load one logical object from any supported file type."""
    backend = get_backend(path)
    with backend.open_read(path) as ref:
        if fields_to_load is not None:
            return backend.load(ref, fields_to_load, root=root)
        elif config_name is not None:
            config = LoadConfig.load_resource(config_name)
            return load_from_config(backend, ref, config, root=root)
        else:
            raise ValueError("Either fields_to_load or config_name must be provided.")


def load_from_config(
    backend: IOBackend,
    ref,
    config: LoadConfig,
    *,
    root: str = "/",
) -> dict[str, Any]:

    # Main object's ordinary fields
    data = backend.load(
        ref,
        config.get_fields_to_load(
            enabled_only=False,
        ),
        root=root,
    )

    # Nested objects
    for name, subconfig_spec in config.subconfigs.items():

        subroot = join_io_path(
            root,
            subconfig_spec.path,
        )

        if not backend.exists(
            ref,
            subroot,
        ):
            if subconfig_spec.required:
                raise KeyError(
                    f"Required subobject {name!r} " f"not found at {subroot!r}"
                )

            data[name] = None
            continue

        subconfig = LoadConfig.load_resource(subconfig_spec.config)

        data[name] = load_from_config(
            backend,
            ref,
            subconfig,
            root=subroot,
        )

    return data


def load_fields_from_sources(
    primary_path: str | Path,
    fields_to_load: dict[str, dict[str, FieldSpec]],
    *,
    root: str = "/",
) -> dict[str, Any]:
    """
    Load one logical CATAN object whose fields may come
    from different physical sources.

    Each FieldSpec.source_path overrides primary_path for
    that field.
    """

    primary_path = Path(primary_path)

    # physical source ->
    #     group ->
    #         label -> spec
    grouped_by_source: dict[Path, dict[str, dict[str, FieldSpec]]] = {}

    for group_name, fields in fields_to_load.items():

        for label, spec in fields.items():

            source = resolve_source_path(primary_path, spec.source_path)
            source_fields = grouped_by_source.setdefault(source, {})
            group_fields = source_fields.setdefault(group_name, {})
            group_fields[label] = spec

    result: dict[str, Any] = {}

    for source_path, source_fields in grouped_by_source.items():

        loaded = load_file(source_path, source_fields, root=root)
        for group_name, group_data in loaded.items():

            if group_name not in result:
                result[group_name] = {}

            result[group_name].update(group_data)

    return result


def read_assignments_source(
    path,
    fields_to_load,
    *,
    load_data=False,
):
    from copy import deepcopy
    from dataclasses import replace

    import numpy as np

    from .inspection import check_fields_compatibility

    backend = get_backend(path)
    with backend.open_read(path) as ref:
        object_type = backend.get_attribute(ref, "/", "object_type", default=None)

    if object_type is not None:
        value = np.asarray(object_type)
        if value.size == 1:
            object_type = value.item()

        if isinstance(object_type, bytes):
            object_type = object_type.decode("utf-8", errors="replace")

        object_type = str(object_type).strip()

    if object_type in {"SessionData", "SessionList", "ModelData"}:
        raise ValueError(
            f"The selected file is a CATAN {object_type} file, "
            "not an assignments file. Select an assignments file "
            "or an external file containing a neuron-by-session IDs array."
        )

    fields = deepcopy(fields_to_load or {})

    # Never use editable/preset mappings for CATAN's internal metadata.
    fields.pop("curation", None)

    ids_spec = fields.get("assignments", {}).get("ids")
    if ids_spec is None:
        return {
            "problems": [
                "No enabled field is configured for assignments.ids. "
                "Select a preset or enable and configure that field."
            ],
            "data": None,
        }

    required = {}
    for group, specs in fields.items():
        selected = {
            label: spec
            for label, spec in specs.items()
            if spec.required or (group == "assignments" and label == "ids")
        }
        if selected:
            required[group] = selected

    report = check_fields_compatibility(
        path,
        required,
        refresh=True,
        raise_source_errors=True,
    )

    problems = [
        (
            f"{field.group}.{field.label}: {field.spec.path}\n"
            f"Source: {resolve_source_path(path, field.spec.source_path)}\n"
            f"{field.reason or 'Configured field is unavailable.'}"
        )
        for field in report.fields
        if not field.available
    ]

    if problems or not load_data:
        return {"problems": problems, "data": None}

    # Metadata belongs to the native IDs stored in this file.
    # Do not attach it to IDs selected from a different physical source.
    ids_source = resolve_source_path(path, ids_spec.source_path)
    if object_type == "AssignmentsData" and ids_source == Path(path):
        native = LoadConfig.fields_from_resource(
            NATIVE_ASSIGNMENTS_CONFIG,
            enabled_only=False,
        )
        fields["curation"] = {
            label: replace(spec, required=False, source_path=None)
            for label, spec in native.get("curation", {}).items()
        }

    return {
        "problems": [],
        "data": load_fields_from_sources(path, fields),
    }


def save_file(
    path: str | Path,
    data: Any,
    fields_to_save: dict[str, dict[str, FieldSpec]],
    *,
    root: str = "/",
    mat_version: Literal["pre73", "7.3"] = "7.3",
    root_attributes: dict[str, Any] | None = None,
) -> None:
    """Save one logical object according to a config-derived field mapping."""
    backend = get_backend(path, for_write=True, mat_version=mat_version)
    with backend.open_write(path) as ref:
        if root_attributes:
            for name, value in root_attributes.items():
                backend.set_attribute(ref, "/", name, value, root=root)
        backend.write(ref, data, fields_to_save, root=root)


def save_from_config(
    backend: IOBackend,
    ref,
    data: Any,
    config: LoadConfig,
    *,
    root: str = "/",
) -> None:

    backend.write(
        ref,
        data,
        config.get_fields_to_save(
            enabled_only=False,
        ),
        root=root,
    )

    for name, subconfig_spec in config.subconfigs.items():

        subdata = get_data_value(
            data,
            name,
            default=None,
        )

        if subdata is None:
            if subconfig_spec.required:
                raise ValueError(f"Required subobject {name!r} is missing.")

            continue

        subconfig = LoadConfig.load_resource(subconfig_spec.config)

        save_from_config(
            backend,
            ref,
            subdata,
            subconfig,
            root=join_io_path(
                root,
                subconfig_spec.path,
            ),
        )


def resolve_source_path(
    primary_path: str | Path,
    source_path: str | Path | None,
) -> Path:
    """
    Resolve the physical source for one configured field.

    source_path is None
        -> use primary_path

    source_path is absolute
        -> use it directly

    source_path is relative
        -> resolve relative to the directory containing
           primary_path
    """

    primary_path = Path(primary_path)

    if source_path is None:
        return primary_path

    source_path = Path(source_path)

    if source_path.is_absolute():
        return source_path

    return primary_path.parent / source_path


def join_io_path(
    root: str,
    path: str,
) -> str:

    if path in ("", "/", "."):
        return root

    root = root.strip("/")
    path = path.strip("/")

    if not root:
        return f"/{path}"

    return f"/{root}/{path}"
