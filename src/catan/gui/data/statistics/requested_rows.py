from copy import deepcopy

import numpy as np

from catan.gui.background_tasks.runtime import current_task_context

from .sparse_values import SparseStatisticArray


def evaluate_requested_rows(
    dimensions,
    read_rows,
    requested_refs,
    *,
    reference_aliases=None,
    reduction_aliases=None,
    applied_filters=(),
    select_values=None,
    block_rows=1024,
):
    """Evaluate exact rows while preserving complete coordinate domains."""
    reference_aliases = reference_aliases or {}

    refs = {
        name: np.asarray(values, dtype=np.int64)
        for name, values in requested_refs.items()
    }

    if not refs or any(values.ndim != 1 for values in refs.values()):
        raise ValueError("Requested coordinates must be vectors.")

    lengths = {len(values) for values in refs.values()}
    if len(lengths) != 1:
        raise ValueError("Requested coordinate vectors must have equal lengths.")

    if any(info.mode == "reduced" for info in dimensions.values()):
        raise ValueError("Requested-row evaluation requires unreduced coordinates.")

    output_dims = tuple(
        name for name, info in dimensions.items() if info.mode == "remaining"
    )

    values_parts = []
    position_parts = []
    ctx = current_task_context()

    for start in range(0, next(iter(lengths)), block_rows):
        if ctx is not None:
            ctx.check_cancelled()

        block = {
            name: values[start : start + block_rows] for name, values in refs.items()
        }

        valid = np.ones(len(next(iter(block.values()))), dtype=bool)
        logical = {}

        # Resolve linked coordinates, including "with previous".
        for name, values in block.items():
            base, offset = reference_aliases.get(name, (name, 0))

            if base not in dimensions:
                raise ValueError(f"Unknown requested coordinate: {name}")

            coords = values - offset

            if base in logical:
                valid &= logical[base] == coords
            else:
                logical[base] = coords

            info = dimensions[base]
            if info.mode == "fixed":
                valid &= coords == info.parameter

        columns = []

        for name in output_dims:
            if name not in logical:
                raise ValueError(f"Missing requested coordinate: {name}")

            lookup = {
                int(value): index for index, value in enumerate(dimensions[name].coords)
            }

            positions = np.fromiter(
                (lookup.get(int(value), -1) for value in logical[name]),
                dtype=np.int64,
                count=len(valid),
            )

            valid &= positions >= 0
            columns.append(positions)

        positions = (
            np.column_stack(columns)
            if columns
            else np.empty((len(valid), 0), dtype=np.int64)
        )

        if not np.any(valid):
            continue

        block = {name: values[valid] for name, values in block.items()}
        positions = positions[valid]

        values = np.asarray(read_rows(block), dtype=np.float32)

        if values.shape != (len(positions),):
            raise ValueError("Requested-row reader returned an invalid shape.")

        mask = (
            np.ones(values.shape, dtype=bool)
            if select_values is None
            else np.asarray(select_values(values), dtype=bool)
        )

        if mask.shape != values.shape:
            raise ValueError("Value selector returned an invalid shape.")

        mask &= ~np.isnan(values)

        if np.any(mask):
            values_parts.append(values[mask])
            position_parts.append(positions[mask])

    result = SparseStatisticArray(
        name="",
        title="",
        dimensions=deepcopy(dimensions),
        values=(
            np.concatenate(values_parts)
            if values_parts
            else np.empty(0, dtype=np.float32)
        ),
        positions=(
            np.concatenate(position_parts)
            if position_parts
            else np.empty((0, len(output_dims)), dtype=np.int64)
        ),
        reference_aliases=dict(reference_aliases),
        reduction_aliases=dict(reduction_aliases or {}),
        applied_pair_filters=tuple(applied_filters),
    )

    result.validate()
    return result
