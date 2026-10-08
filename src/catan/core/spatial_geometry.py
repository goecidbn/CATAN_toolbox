"""Image geometry uses (height, width); sparse pixels use C order."""

import numpy as np


def validate_dims(values):
    values = np.asarray(values)

    if values.size != 2 or values.dtype.kind not in "iuf":
        raise ValueError("Dimensions must contain exactly two numbers: height, width.")

    values = values.reshape(-1)

    if (
        not np.all(np.isfinite(values))
        or np.any(values < 1)
        or np.any(values != np.floor(values))
    ):
        raise ValueError("Height and width must be positive integers.")

    return tuple(int(value) for value in values)


def image_dims(shape):
    shape = tuple(shape)

    if len(shape) == 2 or (len(shape) == 3 and shape[2] == 3):
        return validate_dims(shape[:2])

    raise ValueError(f"Expected a grayscale or RGB image; received shape {shape}.")


def background_transpose(shape, dims, orientation):
    shape = image_dims(shape)
    dims = validate_dims(dims)

    if orientation not in {"auto", "as_stored", "transpose"}:
        raise ValueError(f"Unknown background orientation: {orientation!r}")

    transpose = orientation == "transpose"

    if orientation == "auto" and shape != dims and shape[::-1] == dims:
        transpose = True

    corrected = shape[::-1] if transpose else shape

    if corrected != dims:
        raise ValueError(
            f"Background shape {shape}, after orientation "
            f"{orientation!r}, is {corrected}; expected {dims}."
        )

    return transpose


def check_shapes(
    dims,
    footprint_shape,
    background_shape=None,
    orientation="auto",
    expected_dims=None,
):
    dims = validate_dims(dims)

    if len(footprint_shape) != 2 or footprint_shape[0] != dims[0] * dims[1]:
        raise ValueError(
            f"Footprints have shape {footprint_shape}; expected "
            f"({dims[0] * dims[1]}, neurons) for dimensions {dims}."
        )

    if background_shape is not None:
        background_transpose(background_shape, dims, orientation)

    if expected_dims is not None and dims != tuple(expected_dims):
        raise ValueError(
            f"This session has dimensions {dims}; loaded sessions use "
            f"{tuple(expected_dims)}. Mixing image grids requires "
            "resampling."
        )

    return dims


def inspect_dimensions(
    path,
    fields,
    dimensions,
    orientation="auto",
    expected_dims=None,
):
    """Run in the isolated reader; read only a dimension vector if needed."""
    from dataclasses import replace

    from catan.core.io.api import (
        load_fields_from_sources,
        resolve_source_path,
    )
    from catan.core.io.inspection import inspect_file
    from catan.core.io.types import FieldSpec, normalize_path

    structures = {}

    def structure_for(spec):
        source = resolve_source_path(path, spec.source_path)
        key = str(source)

        if key not in structures:
            structures[key] = inspect_file(
                source,
                refresh=True,
            )

        return structures[key]

    def info(spec):
        structure = structure_for(spec)
        key = normalize_path(spec.path)

        if spec.source == "attribute":
            item = structure.attributes.get((key, spec.attribute))
        else:
            item = structure.entries.get(key)

        if item is None:
            raise ValueError(f"Configured field {spec.path!r} was not found.")

        return item

    try:
        spatial = fields.get("spatial", {})
        footprint = spatial.get("footprints")

        if footprint is None:
            raise ValueError(
                "Select the spatial footprints field before " "checking dimensions."
            )

        fp_info = info(footprint)

        if fp_info.shape is None:
            raise ValueError("The footprint field has no array shape.")

        mode = dimensions.get("mode", "manual")

        if mode == "manual":
            dims = validate_dims(
                [
                    dimensions.get("height", 512),
                    dimensions.get("width", 512),
                ]
            )

        elif mode in {"values", "image"}:
            raw = dimensions.get("field")
            spec = FieldSpec.from_dict(raw) if raw else spatial.get("background")

            if spec is None:
                raise ValueError(
                    "Choose a dimension source with Browse, "
                    "or specify dimensions manually."
                )

            meta = info(spec)

            if meta.shape is None:
                raise ValueError("The dimension source has no array shape.")

            if mode == "values":
                if np.prod(meta.shape) != 2:
                    raise ValueError(
                        "The dimensions field must contain two values, "
                        f"not shape {meta.shape}."
                    )

                loaded = load_fields_from_sources(
                    path,
                    {
                        "geometry": {
                            "dims": replace(spec, required=True),
                        }
                    },
                )
                dims = validate_dims(loaded["geometry"]["dims"])

            else:
                dims = image_dims(meta.shape)

                if orientation == "transpose":
                    dims = dims[::-1]

        else:
            raise ValueError(f"Unknown dimensions mode: {mode!r}")

        background_shape = None
        background = spatial.get("background")

        if background is not None:
            structure = structure_for(background)
            if structure.matches(background):
                background_shape = info(background).shape

        dims = check_shapes(
            dims,
            fp_info.shape,
            background_shape,
            orientation,
            expected_dims,
        )

        return {
            "ok": True,
            "dims": dims,
            "message": (f"Height {dims[0]}, width {dims[1]}: shapes agree."),
        }

    except ValueError as exc:
        return {
            "ok": False,
            "dims": None,
            "message": str(exc),
        }


def bounded_view(bounds, rect, viewport):
    """Limit zoom-out at full-image fit while retaining square pixels."""
    x0, y0, x1, y1 = bounds
    viewport_width, viewport_height = viewport
    ratio = viewport_width / viewport_height

    left, bottom, width, height = rect
    center_x = left + width / 2
    center_y = bottom + height / 2

    width = max(abs(width), abs(height) * ratio, 1e-6)
    width = min(width, max(x1 - x0, (y1 - y0) * ratio))
    height = width / ratio

    left = (
        (x0 + x1 - width) / 2
        if width >= x1 - x0
        else np.clip(center_x - width / 2, x0, x1 - width)
    )
    bottom = (
        (y0 + y1 - height) / 2
        if height >= y1 - y0
        else np.clip(center_y - height / 2, y0, y1 - height)
    )

    return float(left), float(bottom), float(width), float(height)
