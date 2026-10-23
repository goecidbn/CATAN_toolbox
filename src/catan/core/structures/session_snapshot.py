import json, os
from pathlib import Path

import numpy as np
from scipy import sparse

from .session import SessionData
from .remap import Remapping
from .load_config import FieldSpec, LoadConfig

SESSION_FIELDS = (
    "name",
    "path",
    "active",
    "time_offset",
    "session_color",
    "params",
    "dims",
    "footprints",
    "background",
    "background_template",
    "background_origin",
    "included",
    "synthetic",
    "centroids",
    "n_neurons",
    "_traces",
    "_default_trace",
    "quality",
    "status",
    "alignment_issue",
    "alignment_metrics",
    "derived_statistics",
    "known_trace_keys",
    "trace_revision",
)
REMAP_FIELDS = (
    "shift",
    "rotation",
    "matrix",
    "flow",
    "transpose",
    "remap_data",
    "success",
    "dims",
    "max_shift",
    "max_rotation",
    "c_min",
    "min_zcorr",
    "rotation_step",
    "rotation_refine_step",
    "c_max",
    "c_zscored",
    "method",
    "flow_info",
    "confirmation",
)


def prepare_session_snapshot(
    session,
    *,
    processing=None,
    copy_arrays=False,
):
    if session.path is None:
        raise ValueError("Cannot save a session without its original source path.")

    arrays = {}

    def encode(value):
        if isinstance(value, np.ndarray) or sparse.issparse(value):
            key = f"a{len(arrays):04d}"
            arrays[key] = value.copy() if copy_arrays else value
            return {"__array__": key}
        if isinstance(value, np.generic):
            return value.item()
        if isinstance(value, Path):
            return str(value)
        raise TypeError(f"Cannot serialize {type(value).__name__}.")

    values = {key: getattr(session, key) for key in SESSION_FIELDS}
    values["path"] = os.path.abspath(os.path.expanduser(os.fspath(session.path)))
    payload = {
        "version": 2,
        "session": values,
        "source_config": (
            None if session.source_config is None else session.source_config.to_dict()
        ),
        "remap": (
            None
            if session.remap is None
            else {key: getattr(session.remap, key) for key in REMAP_FIELDS}
        ),
        "processing": processing
        or {
            "geometry_revision": 0,
            "alignment_stale": False,
        },
    }
    document = json.dumps(payload, default=encode)
    return {
        "document": document,
        "arrays": arrays,
    }


def write_prepared_session_snapshot(backend, ref, snapshot, *, root="/"):
    arrays = snapshot["arrays"]

    fields = {
        "arrays": {
            key: FieldSpec(
                path=f"/snapshot_arrays/{key}",
                required=True,
            )
            for key in arrays
        },
        "snapshot": {
            "document": FieldSpec(
                path="/snapshot_json",
                required=True,
            ),
        },
    }

    backend.write(
        ref,
        {
            "arrays": arrays,
            "snapshot": {"document": snapshot["document"]},
        },
        fields,
        root=root,
    )


def write_session_snapshot(backend, ref, session, *, root="/", processing=None):
    # Preserve the existing API for other callers.
    snapshot = prepare_session_snapshot(
        session,
        processing=processing,
    )
    write_prepared_session_snapshot(
        backend,
        ref,
        snapshot,
        root=root,
    )


def read_session_snapshot(backend, ref, *, root="/"):
    fields = {
        "snapshot": {
            "document": FieldSpec(path="/snapshot_json", required=True),
        }
    }
    document = backend.load(ref, fields, root=root)["snapshot"]["document"]
    if isinstance(document, bytes):
        document = document.decode("utf-8")

    def decode(value):
        if set(value) != {"__array__"}:
            return value
        key = value["__array__"]
        fields = {
            "arrays": {
                key: FieldSpec(path=f"/snapshot_arrays/{key}", required=True),
            }
        }
        return backend.load(ref, fields, root=root)["arrays"][key]

    payload = json.loads(document, object_hook=decode)
    if payload["version"] != 2:
        raise ValueError("Unsupported session snapshot version.")

    # Empty construction: no footprint processing or alignment.
    session = SessionData()
    optional_fields = {
        "derived_statistics",
        "known_trace_keys",
        "trace_revision",
    }

    for key in SESSION_FIELDS:
        if key in payload["session"]:
            setattr(session, key, payload["session"][key])
        elif key not in optional_fields:
            raise ValueError(f"Session snapshot is missing {key!r}.")

    session.dims = tuple(session.dims)
    session.source_config = (
        None
        if payload["source_config"] is None
        else LoadConfig.from_dict(payload["source_config"])
    )

    session.remap = None
    if payload["remap"] is not None:
        session.remap = Remapping(evaluate=False)
        for key in REMAP_FIELDS:
            if (
                key in {"method", "flow_info", "confirmation"}
                and key not in payload["remap"]
            ):
                continue  # Older snapshots retain the constructor default.
            setattr(session.remap, key, payload["remap"][key])
        if session.remap.dims is not None:
            session.remap.dims = tuple(session.remap.dims)

    if session.status["spatial_loaded"]:
        expected = (int(np.prod(session.dims)), session.n_neurons)
        if session.footprints.shape != expected:
            raise ValueError("Saved footprints have inconsistent dimensions.")
        for key in ("included", "synthetic"):
            if getattr(session, key).shape != (session.n_neurons,):
                raise ValueError(f"Saved {key} flags have inconsistent dimensions.")
        if session.background_template is None:
            raise ValueError("Saved spatial data lack the original template.")

        from catan.core.spatial_geometry import check_shapes

        for image in (
            session.background_template,
            session.background,
        ):
            if image is not None:
                check_shapes(
                    session.dims,
                    session.footprints.shape,
                    image.shape,
                    "as_stored",
                )

    session.evaluate_alignment_status()
    # Counts and assignments are separate files, not restored here.
    session.status["registered_to_model"] = False
    session.status["matched"] = False
    session._restored_processing = payload["processing"]
    return session


# ----------------------------------------------------------------------
# Display-only background (template + alignment, no footprints), e.g. the
# Overview background stored with assignments.
# ----------------------------------------------------------------------

BACKGROUND_FIELDS = ("name", "path", "params", "dims", "status")


def prepare_background_snapshot(session):
    """{"document": JSON text, "arrays": {key: array}} of a background-only
    SessionData (background_template, background, remap)."""
    arrays = {}

    def encode(value):
        if isinstance(value, np.ndarray):
            key = f"a{len(arrays):04d}"
            arrays[key] = value
            return {"__array__": key}
        if isinstance(value, np.generic):
            return value.item()
        if isinstance(value, Path):
            return str(value)
        raise TypeError(f"Cannot serialize {type(value).__name__}.")

    values = {key: getattr(session, key) for key in BACKGROUND_FIELDS}
    values["path"] = None if session.path is None else str(session.path)
    payload = {
        "version": 1,
        "session": values,
        "background_template": session.background_template,
        "background": session.background,
        "remap": (
            None
            if session.remap is None
            else {key: getattr(session.remap, key) for key in REMAP_FIELDS}
        ),
    }
    return {"document": json.dumps(payload, default=encode), "arrays": arrays}


def restore_background_snapshot(document, arrays):
    """SessionData from ``prepare_background_snapshot`` output."""
    if isinstance(document, bytes):
        document = document.decode("utf-8")

    def decode(value):
        if set(value) != {"__array__"}:
            return value
        return np.asarray(arrays[value["__array__"]])

    payload = json.loads(document, object_hook=decode)
    if payload.get("version") != 1:
        raise ValueError("Unsupported background snapshot version.")

    session = SessionData()
    for key in BACKGROUND_FIELDS:
        setattr(session, key, payload["session"][key])
    session.dims = tuple(session.dims)

    session.background_template = np.asarray(
        payload["background_template"], dtype=np.float32
    )
    session.background = np.asarray(payload["background"], dtype=np.float32)
    for image in (session.background_template, session.background):
        if image.shape != session.dims:
            raise ValueError("Stored background has inconsistent dimensions.")

    session.remap = None
    if payload["remap"] is not None:
        session.remap = Remapping(evaluate=False)
        for key in REMAP_FIELDS:
            if key in payload["remap"]:
                setattr(session.remap, key, payload["remap"][key])
        if session.remap.dims is not None:
            session.remap.dims = tuple(session.remap.dims)

    return session
