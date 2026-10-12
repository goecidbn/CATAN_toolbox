import numpy as np

from . import plotdata_histogram

from catan.gui.data.assignment_colors import assignment_color
from . import plotdata_scatter
from .errors import StatisticsPlotError
from catan.gui.data.assignment_colors import assignment_label

ASSIGNMENTS_REF = "__assignments__"


class ComparisonHistogram(plotdata_histogram.PlotData):
    """Several histograms with common bins and adjacent dataset bars."""

    def __init__(self, tables, bins, data):
        self.names = tuple(tables)

        if not self.names:
            raise ValueError("Select at least one assignments dataset.")

        self.data = data
        self.local_selection = None

        # Determine the shared range without concatenating all datasets.
        lower = np.inf
        upper = -np.inf

        for table in tables.values():
            values = np.asarray(table.values).reshape(-1)

            for start in range(0, values.size, 65_536):
                block = values[start : start + 65_536]
                finite = block[np.isfinite(block)]

                if finite.size:
                    lower = min(lower, float(finite.min()))
                    upper = max(upper, float(finite.max()))

        if np.isfinite(lower):
            edges = np.histogram_bin_edges(
                np.asarray([lower, upper]),
                bins=bins,
            )
        else:
            edges = np.linspace(0.0, 1.0, bins + 1)

        self.shared_edges = edges

        self.parts = tuple(
            plotdata_histogram.build_plot_data(table, bins=edges)
            for table in tables.values()
        )

        dataset_count = len(self.parts)
        bin_count = len(edges) - 1

        self.offsets = np.cumsum([0] + [part.table.n_rows for part in self.parts])

        self.title = dict(self.parts[0].title)

        # There is no single underlying table. Methods requiring one
        # are overridden below.
        self.table = None

        # Rendering slots subdivide each actual statistic bin.
        # Row membership still uses the original shared bin edges.
        self.bin_edges = np.r_[
            np.concatenate(
                [
                    np.linspace(left, right, dataset_count + 1)[:-1]
                    for left, right in zip(edges[:-1], edges[1:])
                ]
            ),
            edges[-1],
        ]

        self.bin_counts = np.column_stack(
            [part.bin_counts for part in self.parts]
        ).ravel()

        self.bin_rows = [
            self.parts[dataset].bin_rows[bin_index] + self.offsets[dataset]
            for bin_index in range(bin_count)
            for dataset in range(dataset_count)
        ]

        self.row_to_bin = np.concatenate(
            [
                np.where(
                    part.row_to_bin >= 0,
                    part.row_to_bin * dataset_count + dataset,
                    -1,
                )
                for dataset, part in enumerate(self.parts)
            ]
        )

        self.dataset_colors = np.asarray(
            [assignment_color(name) for name in self.names],
            dtype=np.float32,
        )

        self.bar_colors = np.tile(
            self.dataset_colors,
            (bin_count, 1),
        )

    def ref_sets_for_rows(self, rows):
        rows = np.atleast_1d(rows).astype(int, copy=False)
        result = []

        for dataset, part in enumerate(self.parts):
            start = self.offsets[dataset]
            stop = self.offsets[dataset + 1]

            local_rows = rows[(rows >= start) & (rows < stop)] - start

            if local_rows.size:
                refs = part.table.refs_for_rows(
                    local_rows,
                    include_fixed=True,
                )
                result.append(
                    {
                        **refs,
                        ASSIGNMENTS_REF: self.names[dataset],
                    }
                )

        return tuple(result)

    def refs_for_bin(self, bin_index):
        refs = self.ref_sets_for_rows(self.rows_for_bin(bin_index))
        return refs[0] if refs else {}

    def rows_for_threshold(self, spec):
        if not spec.active:
            return np.empty(0, dtype=int)

        return np.concatenate(
            [
                np.flatnonzero(
                    np.isfinite(part.table.values) & spec.mask(part.table.values)
                )
                + self.offsets[dataset]
                for dataset, part in enumerate(self.parts)
            ]
        )

    def tooltip_for_bin(self, bin_index):
        actual_bin, dataset = divmod(bin_index, len(self.names))
        closing = "]" if actual_bin == len(self.shared_edges) - 2 else ")"

        return (
            f"{self.names[dataset]}\n"
            f"{self.title['x']}\n"
            f"Values: [{self.shared_edges[actual_bin]:.4g}, "
            f"{self.shared_edges[actual_bin + 1]:.4g}{closing}\n"
            f"Count: {int(self.bin_counts[bin_index])}"
        )

    def map_components(self, components, source, target):
        result = []

        for component in components:
            result.extend(
                self.data.map_assignment_components(
                    [component],
                    source,
                    target,
                    whole_neurons=component.session_id is None,
                )
            )

        return list(dict.fromkeys(result))

    def selections_by_dataset(self, components):
        components = components or []

        if not isinstance(components, (list, tuple, set)):
            components = [components]

        # Preserve the original clicked identities. Mapping to the
        # active dataset and back can otherwise broaden or lose them.
        if self.local_selection is not None:
            origin, expected_active_selection = self.local_selection

            if frozenset(components) == expected_active_selection:
                result = {}

                for name in self.names:
                    if name in origin:
                        result[name] = list(dict.fromkeys(origin[name]))
                    else:
                        result[name] = list(
                            dict.fromkeys(
                                component
                                for source, selected in origin.items()
                                for component in self.map_components(
                                    selected, source, name
                                )
                            )
                        )

                return result

        return {
            name: self.map_components(
                components,
                self.data.current_assignments,
                name,
            )
            for name in self.names
        }

    def rows_matching_components(self, components):
        selections = self.selections_by_dataset(components)

        return np.concatenate(
            [
                part.rows_matching_components(selections[name]) + self.offsets[dataset]
                for dataset, (name, part) in enumerate(zip(self.names, self.parts))
            ]
        )

    def selected_histogram_for_components(self, components):
        rows = self.rows_matching_components(components)
        bar_ids = self.row_to_bin[rows]
        valid = bar_ids >= 0

        return (
            np.bincount(
                bar_ids[valid],
                minlength=len(self.bin_counts),
            ),
            rows[valid],
            bar_ids[valid],
        )

    def values_matching_components(self, components):
        selections = self.selections_by_dataset(components)

        return np.concatenate(
            [
                part.values_matching_components(selections[name])
                for name, part in zip(self.names, self.parts)
            ]
        )


class ComparisonScatter(plotdata_scatter.PlotData):
    """One point per X neuron, paired with its best Y partner."""

    def __init__(self, x_table, y_table, x_name, y_name, data):
        self.names = (x_name, y_name)
        self.data = data
        self.local_selection = None
        self.unmatched_count = 0

        super().__init__(
            x_table=x_table,
            y_table=y_table,
            row_matcher=self._best_rows,
        )

        self.title = {
            "x": (f"{assignment_label(x_name)}: " f"{x_table.stat.display_title}"),
            "y": (f"{assignment_label(y_name)}: " f"{y_table.stat.display_title}"),
        }

        if not self.n_markers:
            raise StatisticsPlotError(
                "No matched neurons have finite values on both axes."
            )

    @staticmethod
    def _neuron_ids(table):
        if table.canonical_dims != ("neuron",):
            raise StatisticsPlotError(
                "Assignments comparison requires one value per neuron "
                "on each axis. Keep the neuron dimension and reduce "
                "all session dimensions."
            )

        ids = np.asarray(
            table.canonical_refs()["neuron"],
            dtype=np.int64,
        )

        if np.unique(ids).size != ids.size:
            raise StatisticsPlotError(
                "Assignments comparison requires a unique row per neuron."
            )

        return ids

    def _best_rows(self, x_table, y_table):
        self.x_ids = self._neuron_ids(x_table)
        self.y_ids = self._neuron_ids(y_table)

        y_lookup = {int(neuron): row for row, neuron in enumerate(self.y_ids)}

        comparison = self.data.assignment_comparison(*self.names)
        x_rows = []
        y_rows = []

        for row, neuron in enumerate(self.x_ids):
            partners = comparison.partners(int(neuron))

            if not partners:
                continue

            # Always use the best partner. Do not substitute a weaker
            # partner merely because its statistic happens to be available.
            target_row = y_lookup.get(partners[0].neuron_id)

            if target_row is None:
                continue

            x_rows.append(row)
            y_rows.append(target_row)

        self.unmatched_count = len(self.x_ids) - len(x_rows)

        return (
            np.asarray(x_rows, dtype=np.intp),
            np.asarray(y_rows, dtype=np.intp),
        )

    def ref_sets_for_rows(self, rows):
        rows = np.atleast_1d(rows).astype(int, copy=False)

        return (
            {
                "neuron": self.x_ids[self.x_rows[rows]],
                ASSIGNMENTS_REF: self.names[0],
            },
            {
                "neuron": self.y_ids[self.y_rows[rows]],
                ASSIGNMENTS_REF: self.names[1],
            },
        )

    # Share the existing dataset-aware selection translation.
    map_components = ComparisonHistogram.map_components
    selections_by_dataset = ComparisonHistogram.selections_by_dataset

    def markers_matching_components(self, components):
        selections = self.selections_by_dataset(components)
        selected = np.zeros(self.n_markers, dtype=bool)

        for name, ids, rows in (
            (self.names[0], self.x_ids, self.x_rows),
            (self.names[1], self.y_ids, self.y_rows),
        ):
            neurons = [component.neuron_id for component in selections[name]]

            selected |= np.isin(
                ids[rows[self.marker_rows]],
                neurons,
            )

        return np.flatnonzero(selected)

    def tooltip_for_marker(self, marker_index):
        row = self.marker_rows[marker_index]

        return (
            f"X — {assignment_label(self.names[0])}: "
            f"neuron {self.x_ids[self.x_rows[row]]}\n"
            f"Y — {assignment_label(self.names[1])}: "
            f"neuron {self.y_ids[self.y_rows[row]]}\n"
            f"{self.title['x']}: {self.x[marker_index]:.4g}\n"
            f"{self.title['y']}: {self.y[marker_index]:.4g}"
        )
