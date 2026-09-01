from __future__ import annotations

from collections.abc import Mapping
from dataclasses import replace
from pathlib import Path, PurePosixPath
from typing import Any

from scipy import sparse

from catan.core.structures.load_config import FieldSpec
from .types import SaveEntry, has_wildcard, normalize_path


_MISSING = object()
_NO_DEFAULT = object()

def normalize_scalar(value: Any) -> Any:
    """Normalize common scalar/path values without touching ndarrays."""
    try:
        import numpy as np
        if isinstance(value, np.generic):
            return value.item()
    except Exception:
        pass

    if isinstance(value, bytes):
        return value.decode("utf-8")
    if isinstance(value, Path):
        return str(value)
    return value


def get_member(obj: Any, name: str, default: Any = _NO_DEFAULT) -> Any:
    if isinstance(obj, Mapping):
        if name in obj:
            return obj[name]
    elif hasattr(obj, name):
        return getattr(obj, name)

    if default is not _NO_DEFAULT:
        return default
    raise KeyError(name)



def get_data_value(
    data: Any,
    key: str,
    *,
    default: Any = _MISSING,
) -> Any:
    """
    Retrieve a value from either a mapping-like object or an attribute.

    Lookup order:
        1. dictionary key
        2. object attribute
        3. default, if supplied

    Raises
    ------
    KeyError
        If the value cannot be found and no default was supplied.
    """

    if isinstance(data, dict):
        if key in data:
            return data[key]

    elif hasattr(data, key):
        return getattr(data, key)

    if default is not _MISSING:
        return default

    raise KeyError(
        f"Could not find {key!r} in object of type "
        f"{type(data).__name__}."
    )

def resolve_data_value(data: Any, group_name: str, label: str) -> Any:
    """Resolve a CATAN field label against a dict or lightweight data object.

    Resolution order intentionally supports both of the layouts CATAN already
    uses:

    * ``data[group_name][label]`` / ``data.<group_name>[label]``
    * ``data[label]`` / ``data.<label>``

    This lets e.g. ``quality`` and ``traces`` remain dictionaries while static
    fields such as ``footprints`` can stay direct SessionData attributes.
    """
    group_obj = get_member(data, group_name, _MISSING)
    if group_obj is not _MISSING:
        value = get_member(group_obj, label, _MISSING)
        if value is not _MISSING:
            return value

    value = get_member(data, label, _MISSING)
    if value is not _MISSING:
        return value

    raise KeyError(
        f"Could not resolve CATAN field {label!r} in group {group_name!r}"
    )


def resolve_wildcard_mapping(data: Any, group_name: str, label: str) -> Mapping[str, Any]:
    group_obj = get_member(data, group_name, _MISSING)

    # Common case: SessionData.quality / SessionData.traces / object.stats.
    if group_obj is not _MISSING:
        if isinstance(group_obj, Mapping):
            nested = get_member(group_obj, label, _MISSING)
            if isinstance(nested, Mapping):
                return nested
            return group_obj

        nested = get_member(group_obj, label, _MISSING)
        if isinstance(nested, Mapping):
            return nested

    root_value = get_member(data, label, _MISSING)
    if isinstance(root_value, Mapping):
        return root_value

    raise TypeError(
        f"Wildcard field {label!r} in group {group_name!r} requires a mapping value"
    )


def expand_wildcard_path(path: str, name: str) -> str:
    path = normalize_path(path)
    # Config wildcards currently refer to the final component. Supporting a
    # single wildcard here keeps the semantics predictable for writing.
    pure = PurePosixPath(path)
    if not has_wildcard(pure.name):
        raise ValueError(f"Path {path!r} does not contain a final-component wildcard")
    return str(pure.parent / name) if str(pure.parent) != "." else f"/{name}"


def iter_save_entries(
    data: Any,
    fields_to_save: dict[str, dict[str, FieldSpec]],
):
    """Expand config fields into concrete values to be written.

    Dynamic wildcard fields are expanded from the corresponding CATAN mapping.
    Optional values that are absent/None are skipped; required ones raise.
    """
    for group_name, fields in fields_to_save.items():
        for label, spec in fields.items():
            if spec.source != "attribute" and has_wildcard(spec.path):
                try:
                    mapping = resolve_wildcard_mapping(data, group_name, label)
                except (KeyError, TypeError):
                    if spec.required:
                        raise
                    continue

                for dynamic_label, value in mapping.items():
                    if value is None:
                        continue
                    concrete_spec = replace(
                        spec,
                        path=expand_wildcard_path(spec.path, str(dynamic_label)),
                    )
                    yield SaveEntry(
                        group=group_name,
                        label=str(dynamic_label),
                        spec=concrete_spec,
                        value=value,
                    )
                continue

            if spec.source == "attribute" and spec.attribute == "*":
                try:
                    mapping = resolve_wildcard_mapping(data, group_name, label)
                except (KeyError, TypeError):
                    if spec.required:
                        raise
                    continue

                for attr_name, value in mapping.items():
                    if value is None:
                        continue
                    concrete_spec = replace(spec, attribute=str(attr_name))
                    yield SaveEntry(group_name, str(attr_name), concrete_spec, value)
                continue

            try:
                value = resolve_data_value(data, group_name, label)
            except KeyError:
                if spec.required:
                    raise
                continue

            if value is None:
                if spec.required:
                    raise ValueError(
                        f"Required field {group_name}.{label} has value None"
                    )
                continue

            yield SaveEntry(group_name, label, spec, value)


def nested_dict_set(root: dict[str, Any], path: str, value: Any) -> None:
    path = normalize_path(path)
    if path == "/":
        raise ValueError("Cannot assign a field value to logical root '/' directly")
    parts = PurePosixPath(path).parts
    current = root
    for part in parts[:-1]:
        if part == "/":
            continue
        child = current.setdefault(part, {})
        if not isinstance(child, dict):
            raise TypeError(f"Cannot create {path!r}; {part!r} is not a mapping")
        current = child
    current[parts[-1]] = value


def nested_dict_get(root: Mapping[str, Any], path: str) -> Any:
    path = normalize_path(path)
    if path == "/":
        return root
    current: Any = root
    for part in PurePosixPath(path).parts:
        if part == "/":
            continue
        if not isinstance(current, Mapping):
            raise TypeError(f"Cannot enter {part!r}; parent is {type(current).__name__}")
        current = current[part]
    return current


def nested_dict_group(root: dict[str, Any], path: str) -> dict[str, Any]:
    path = normalize_path(path)
    if path == "/":
        return root
    current = root
    for part in PurePosixPath(path).parts:
        if part == "/":
            continue
        child = current.setdefault(part, {})
        if not isinstance(child, dict):
            raise TypeError(f"{path!r} overlaps a non-group value at {part!r}")
        current = child
    return current


def normalize_for_storage(value: Any) -> Any:
    """Small, non-destructive normalization shared by writers."""
    if isinstance(value, Path):
        return str(value)
    if sparse.issparse(value):
        return value
    return value
