"""Evaluate labelled Cartesian statistics with bounded intermediates."""

from copy import deepcopy
from math import prod

import numpy as np

from .types import StatisticArray
from .queries import ReductionSpec
from catan.gui.background_tasks.runtime import current_task_context


def evaluate_blockwise(
    dimensions,
    read_block,
    query,
    *,
    aliases=None,
    reference_aliases=None,
    applied_filters=(),
    block_values=262_144,
    max_output_values=2_000_000,
    select_values=None,
):
    aliases = aliases or {}
    dimensions = deepcopy(dimensions)

    base_dims = tuple(d for d, info in dimensions.items() if info.mode == "remaining")

    reductions = query.reduction_dict()
    order = tuple(query.reduction_order or ())
    order += tuple(d for d in reductions if d not in order)

    operations = {}

    for original in order:
        spec = reductions.get(original, ReductionSpec("keep"))

        if spec.method in ("keep", "single"):
            continue

        target = aliases.get(original, original)

        if target not in base_dims:
            continue

        if target in operations and operations[target] != spec:
            raise ValueError(f"Conflicting reductions for linked axis {target}.")

        operations[target] = spec

    operations = tuple(operations.items())

    if sum(spec.error_method != "none" for _, spec in operations) > 1:
        raise ValueError("Only one error-producing reduction is supported.")

    if any(spec.error_method == "bootstrap" for _, spec in operations):
        raise NotImplementedError("Bootstrap errors are not implemented.")

    stage_dims = [base_dims]

    for dim, _ in operations:
        stage_dims.append(tuple(d for d in stage_dims[-1] if d != dim))

    output_dims = stage_dims[-1]
    output_shape = tuple(len(dimensions[d].coords) for d in output_dims)

    count = prod(output_shape)

    if select_values is None and count > max_output_values:
        raise ValueError(
            f"The full statistic would contain {count:,} values before filtering. "
            "Reduce dimensions or select sessions for plotting. "
            "Curator can filter supported statistics in blocks."
        )

    # Exact medians need at least one complete reduction fibre.
    block_values = max(
        block_values,
        *(len(dimensions[d].coords) for d in base_dims),
        1,
    )

    ctx = current_task_context()

    def calculate(level, selected):
        if ctx is not None:
            ctx.check_cancelled()

        dims = stage_dims[level]
        shape = tuple(len(selected[d]) for d in dims)
        expanded = prod(shape)

        if level:
            axis_name, spec = operations[level - 1]
            expanded *= len(dimensions[axis_name].coords)

        # Split output coordinates. Never average independently
        # reduced chunks, which would change reduction semantics.
        if expanded > block_values and any(size > 1 for size in shape):
            split = max(dims, key=lambda d: len(selected[d]))
            axis = dims.index(split)
            coords = selected[split]

            length = max(
                1,
                int(len(coords) * block_values // expanded),
            )

            result = None

            for start in range(0, len(coords), length):
                stop = min(start + length, len(coords))

                part = dict(selected)
                part[split] = coords[start:stop]
                chunk = calculate(level, part)

                if result is None:
                    result = chunk
                    result.dimensions = deepcopy(chunk.dimensions)
                    result.dimensions[split].coords = coords

                    arrays = {}
                    for name in ("values", "errors_low", "errors_high", "n"):
                        value = getattr(chunk, name)
                        arrays[name] = (
                            None if value is None else np.empty(shape, value.dtype)
                        )

                index = [slice(None)] * len(dims)
                index[axis] = slice(start, stop)

                for name, array in arrays.items():
                    if array is not None:
                        array[tuple(index)] = getattr(chunk, name)

            for name, array in arrays.items():
                setattr(result, name, array)

            return result

        if level:
            expanded_selection = dict(selected)
            expanded_selection[axis_name] = dimensions[axis_name].coords

            result = calculate(level - 1, expanded_selection)

            if result.values.shape[result.axis(axis_name)] == 0:
                result.values = np.full(
                    shape,
                    0.0 if spec.method == "sum" else np.nan,
                )

                if result.has_errors or spec.error_method != "none":
                    result.errors_low = np.full(shape, np.nan)
                    result.errors_high = np.full(shape, np.nan)
                    result.n = np.zeros(shape, dtype=int)

                info = result.dimensions[axis_name]
                info.mode = "reduced"
                info.parameter = {
                    "method": spec.method,
                    "error_method": spec.error_method,
                }
            else:
                result.reduce_dimension(axis_name, spec)

            return result

        metadata = deepcopy(dimensions)

        for dim in base_dims:
            metadata[dim].coords = selected[dim]

        values = np.asarray(read_block(selected), dtype=np.float32)

        if values.shape != shape:
            raise ValueError(f"Block shape {values.shape} does not match {shape}.")

        return StatisticArray(
            name="",
            title="",
            values=values,
            dimensions=metadata,
            reduction_aliases=dict(aliases),
            reference_aliases=dict(reference_aliases or {}),
            applied_pair_filters=tuple(applied_filters),
        )

    if select_values is not None:
        from .sparse_values import SparseStatisticArray

        def output_blocks(selected, offsets):
            size = prod(len(selected[d]) for d in output_dims)

            if size > block_values:
                dim = max(output_dims, key=lambda d: len(selected[d]))
                coords = selected[dim]
                length = max(1, len(coords) * block_values // size)

                for start in range(0, len(coords), length):
                    part = dict(selected)
                    part[dim] = coords[start : start + length]

                    origin = dict(offsets)
                    origin[dim] += start

                    yield from output_blocks(part, origin)
            else:
                yield selected, offsets

        names = ("values", "errors_low", "errors_high", "n")
        parts = {name: [] for name in names}
        position_parts = []
        metadata = None

        for selected, offsets in output_blocks(
            {d: dimensions[d].coords for d in output_dims},
            {d: 0 for d in output_dims},
        ):
            # Complete all requested reductions before selecting rows.
            chunk = calculate(len(operations), selected)
            chunk.validate()

            if metadata is None:
                metadata = deepcopy(chunk.dimensions)

                # Preserve the complete coordinate domains.
                for dim in output_dims:
                    metadata[dim].coords = dimensions[dim].coords

            values = np.asarray(chunk.values).reshape(-1)
            mask = select_values(values)

            if mask is None:
                raise ValueError("The value selector returned no mask.")

            mask = np.asarray(mask, dtype=bool)

            if mask.shape != values.shape:
                raise ValueError("The value selector returned an invalid mask shape.")

            rows = np.flatnonzero(mask & ~np.isnan(values))

            if not rows.size:
                continue

            if output_dims:
                positions = np.column_stack(np.unravel_index(rows, chunk.values.shape))
                positions += np.array([offsets[d] for d in output_dims])
            else:
                positions = np.empty(
                    (rows.size, 0),
                    dtype=np.int64,
                )

            position_parts.append(positions)

            for name in names:
                array = getattr(chunk, name)

                if array is not None:
                    parts[name].append(np.asarray(array).reshape(-1)[rows])

        def joined(name):
            if parts[name]:
                return np.concatenate(parts[name])

            array = getattr(chunk, name)

            return None if array is None else np.empty(0, dtype=array.dtype)

        result = SparseStatisticArray(
            name="",
            title="",
            dimensions=metadata,
            values=joined("values"),
            positions=(
                np.concatenate(position_parts, axis=0)
                if position_parts
                else np.empty(
                    (0, len(output_dims)),
                    dtype=np.int64,
                )
            ),
            errors_low=joined("errors_low"),
            errors_high=joined("errors_high"),
            n=joined("n"),
            reduction_aliases=dict(aliases),
            reference_aliases=dict(reference_aliases or {}),
            applied_pair_filters=tuple(applied_filters),
        )

        result.validate()
        return result

    result = calculate(
        len(operations),
        {d: dimensions[d].coords for d in output_dims},
    )

    result.validate()
    return result
