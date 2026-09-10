from typing import Dict, Optional, Tuple
import textwrap
from dataclasses import dataclass
from catan.gui.data.statistics.queries import ReductionSpec
import numpy as np
from scipy import sparse
from vispy import scene, color
from vispy.scene import visuals

from PySide6.QtWidgets import (
    QComboBox,
    QCheckBox,
    QLabel,
)


from catan.gui.structures.state import NeuronComponent
from catan.gui.plots import BasePlot
from catan.gui.interaction import click_events

from catan.gui.data.statistics.dimensions import SESSION_DIMS, neuron_bound_dim
from catan.gui.plots.StatisticsData import StatisticQuerySelector

STATISTIC_CMAPS = {
    "viridis": "viridis",
    "coolwarm": "coolwarm",
    "cubehelix": "cubehelix",
    "red-green": color.Colormap(
        [
            "#b2182b",
            "#f2f2f2",
            "#1a9850",
        ]
    ),
    "blue-orange": color.Colormap(
        [
            "#2166ac",
            "#f2f2f2",
            "#b35806",
        ]
    ),
}


@dataclass
class OverviewRecord:
    # visual: scene.Markers
    pos: np.ndarray
    ids: np.ndarray
    vals: np.ndarray
    n_rois: int
    color: list | tuple | str


class Display(BasePlot.BaseCanvas):
    def __init__(self, parent, controls, config=None):
        super().__init__(parent, controls, config)

        print("fix mouseover to only detect currently displayed neurons!")
        self.unfreeze()
        self.grid = self.central_widget.add_grid(spacing=0)
        self.view = self.grid.add_view(row=0, col=0)
        self.view.stretch = (1, 1)  # expand a lot
        ## set camera
        self.view.camera = scene.PanZoomCamera(aspect=None)
        self.view.camera.interactive = False

        self.plot_root = scene.Node(parent=self.view.scene)

        self.changes_on_click = "selected"

        self.neuron_statistic_values = None
        self.statistic_title = None

        self.statistic_cmap_name = "viridis"
        self.statistic_cmap_reversed = False

        self.statistic_clim = None
        self.build_colorbar()

        self.statistic_nan_color = np.asarray(
            color.Color("#686868").rgba,
            dtype=np.float32,
        )
        self.statistic_nan_color[3] = 0.55

        # self.plot_components: Dict[int, OverviewVisualRecord] = {}

        self.freeze()

    def build_colorbar(self):

        cbar_size = (120, 10)
        cbar_center = (90, 20)
        cbar_hmargin = 15
        cbar_vmargin = 10

        self.statistic_colorbar_bg = visuals.Rectangle(
            center=(cbar_center[0], cbar_center[1] + 15),
            width=cbar_size[0] + 2 * cbar_hmargin,
            height=cbar_size[1] + 2 * cbar_vmargin + 30,
            color=(1.0, 1.0, 1.0, 0.78),
            border_color=None,
            parent=self.scene,
        )

        self.statistic_colorbar_bg.set_gl_state(
            depth_test=False,
            blend=True,
            blend_func=("src_alpha", "one_minus_src_alpha"),
        )
        self.statistic_colorbar_bg.order = 10000
        self.statistic_colorbar_bg.visible = False

        self.statistic_colorbar = visuals.ColorBar(
            cmap=(
                self.statistic_cmap_name
                if not self.statistic_cmap_reversed
                else self.statistic_cmap_name + "_r"
            ),
            orientation="bottom",
            size=cbar_size,
            pos=cbar_center,
            # We'll draw these ourselves
            label="",
            clim=("", ""),
            border_width=1,
            border_color="black",
            parent=self.scene,
        )

        self.statistic_colorbar.order = 10001
        self.statistic_colorbar.visible = False

        self.statistic_colorbar_low = visuals.Text(
            "",
            pos=(cbar_center[0] - cbar_size[0] // 2 + 5, cbar_center[1] + 20),
            anchor_x="center",
            anchor_y="top",
            font_size=9,
            color="black",
            parent=self.scene,
        )

        self.statistic_colorbar_high = visuals.Text(
            "",
            pos=(cbar_center[0] + cbar_size[0] // 2 - 5, cbar_center[1] + 20),
            anchor_x="center",
            anchor_y="top",
            font_size=9,
            color="black",
            parent=self.scene,
        )

        self.statistic_colorbar_title = visuals.Text(
            "",
            pos=(cbar_center[0], cbar_center[1] + 35),
            anchor_x="center",
            anchor_y="top",
            font_size=10,
            color="black",
            parent=self.scene,
        )

        for text in (
            self.statistic_colorbar_low,
            self.statistic_colorbar_high,
            self.statistic_colorbar_title,
        ):
            text.order = 10002
            text.visible = False

    def initialize_overlays(self):

        for style in self.plotting["overlays"]:
            self.plotting["overlays"][style] = visuals.Markers(
                parent=self.plot_root,
            )
            self.plotting["overlays"][style].set_gl_state(
                depth_test=False,
                blend=True,
                blend_func=("src_alpha", "one_minus_src_alpha"),
            )
            self.plotting["overlays"][style].visible = False

        self.plotting["overlays"]["hovered"].order = 100
        self.plotting["overlays"]["highlighted"].order = 90
        self.plotting["overlays"]["focused"].order = 80
        self.plotting["overlays"]["selected"].order = 70

    def plot_background(self):

        if self.data.current_session is None:
            return

        ## remove previous background if exists
        if self.plotting.get("background") is not None:
            self.plotting["background"].parent = None
            del self.plotting["background"]

        background = self.data.current_session.background
        if background is None:
            return

        background /= np.percentile(background, 90)
        self.plotting["background"] = visuals.Image(
            background.astype(np.float32),  # .T,
            cmap="viridis",
            method="subdivide",
            parent=self.plot_root,
        )
        self.plotting["background"].order = -1000  # ensure it's in the back
        self.plotting["background"].set_gl_state(depth_test=False, blend=False)

        H, W = background.shape[:2]
        self.view.camera.set_range(
            x=(0, W),
            y=(0, H),
            margin=0,
        )
        self.update()

    def clean_highlight(self):
        for _, visual in self.plotting["overlays"].items():
            if visual is None:
                continue
            visual.visible = False

    def build_neuron_visuals_union(self):

        if self.data.assignments is None or self.data.assignments.union is None:
            return

        roi_pos, neuron_ids, roi_vals = sparse_A_to_points(
            self.data.assignments.union.footprints, self.data.sessions[0].dims, thr=0.5
        )

        self.plotting["data"]["union"] = OverviewRecord(
            pos=roi_pos,
            ids=neuron_ids,
            vals=roi_vals,
            n_rois=self.data.assignments.union.n_neurons,
            color=None,
        )

    def build_neuron_visuals_session(self, session_id=None, thr=0.2):

        if session_id is None:
            session_id = self.state.current_session_id
        if (
            session_id is None
            or session_id >= len(self.data.sessions)
            or self.data.sessions[session_id] is None
            or not self.data.sessions[session_id].status["spatial_loaded"]
            or not self.data.session_assigned(session_id)
        ):
            return

        if self.display_mode == "tracked_overview" and "union" in self.plotting["data"]:
            present_neuron_ids = np.where(
                self.state.assignments[:, self.state.current_session_id] >= 0
            )[0]

            union = self.plotting["data"]["union"]

            mask = np.isin(union.ids, present_neuron_ids)

            roi_pos = union.pos[mask]
            roi_vals = union.vals[mask]
            neuron_ids = union.ids[mask]

            n_rois = len(present_neuron_ids)
        else:
            roi_pos, roi_ids, roi_vals = sparse_A_to_points(
                self.data.sessions[session_id].footprints,
                self.data.sessions[session_id].dims,
                thr=thr,
            )
            n_rois = self.data.sessions[session_id].footprints.shape[1]

            footprint_to_component = np.array(
                [self.state.get_component_from_footprint(i) for i in range(n_rois)]
            )
            footprint_to_neuron_id = np.array(
                [c.neuron_id if c is not None else -1 for c in footprint_to_component]
            )
            neuron_ids = footprint_to_neuron_id[roi_ids]

        self.plotting["data"][session_id] = OverviewRecord(
            pos=roi_pos,
            ids=neuron_ids,
            vals=roi_vals,
            n_rois=n_rois,
            color=self.state.session_colors[session_id],
        )

    def plot_neurons_visuals(self):

        # print("Plotting neurons")
        for key in self.plotting["data"]:
            if key in self.plotting["visuals"]:
                continue

            footprints = visuals.Markers(parent=self.plot_root)
            footprints.set_gl_state(
                depth_test=False,
                blend=True,
                blend_func=("src_alpha", "one"),  # Make overlaps visible
            )

            record = self.plotting["data"][key]

            colors = self._statistic_colors_for_record(record)

            plot_options = self.styles.get_plot_options(
                style="background" if key == "union" else "default",
                plot_type="marker",
                values=record.vals,
                colors=colors,
                edge_width=0,
            )

            footprints.set_data(self.plotting["data"][key].pos, **plot_options)

            if key == "union":
                footprints.order = 0
            else:
                footprints.order = 1
            self.plotting["visuals"][key] = footprints

    def clean(self, with_union=True):
        for key, visual in self.plotting["visuals"].items():
            if visual is None or (not with_union and key == "union"):
                continue
            visual.visible = False
            visual.parent = None
        self.plotting["visuals"] = (
            {}
            if with_union
            else {k: v for k, v in self.plotting["visuals"].items() if k == "union"}
        )

        for key, data in self.plotting["data"].items():
            if not with_union and key == "union":
                continue
        self.plotting["data"] = (
            {}
            if with_union
            else {k: v for k, v in self.plotting["data"].items() if k == "union"}
        )
        self.clean_highlight()

    ### ==================================================================== ###
    ### ======================= STATISTICS SECTION ========================= ###
    ### ==================================================================== ###

    def set_neuron_statistic_values(
        self,
        values,
        *,
        title=None,
    ):
        self.neuron_statistic_values = (
            None
            if values is None
            else np.asarray(
                values,
                dtype=float,
            )
        )

        self.statistic_title = title

        if self.neuron_statistic_values is None:
            self.statistic_clim = None

        else:
            finite = self.neuron_statistic_values[
                np.isfinite(self.neuron_statistic_values)
            ]

            if finite.size == 0:
                self.statistic_clim = None
            else:
                self.statistic_clim = (
                    float(np.min(finite)),
                    float(np.max(finite)),
                )

        self._update_statistic_colorbar()
        self._update_statistic_colors()

    def _statistic_colors_for_record(
        self,
        record: OverviewRecord,
    ):

        ids = np.asarray(
            record.ids,
            dtype=int,
        )

        point_values = np.full(
            ids.shape,
            np.nan,
            dtype=float,
        )

        if self.neuron_statistic_values is not None:

            valid_ids = (ids >= 0) & (ids < len(self.neuron_statistic_values))

            point_values[valid_ids] = self.neuron_statistic_values[ids[valid_ids]]

        # Start ALL points as the NaN style.
        nan_color = np.asarray(
            color.Color("#707070").rgba,
            dtype=np.float32,
        )
        nan_color[3] = 0.65

        rgba = np.tile(
            nan_color,
            (len(ids), 1),
        )

        if self.statistic_clim is None or self.neuron_statistic_values is None:
            return rgba

        finite = np.isfinite(point_values)

        if not np.any(finite):
            return rgba

        lo, hi = self.statistic_clim

        if hi > lo:
            normalized = (point_values[finite] - lo) / (hi - lo)
        else:
            normalized = np.full(
                np.count_nonzero(finite),
                0.5,
            )

        normalized = np.clip(
            normalized,
            0.0,
            1.0,
        )

        rgba[finite] = self._statistic_colormap().map(normalized).astype(np.float32)

        return rgba

    ### =========================== COLORSTYLES ============================ ###
    def set_statistic_colormap(
        self,
        name: str,
        *,
        reverse: bool = False,
    ):
        self.statistic_cmap_name = name
        self.statistic_cmap_reversed = reverse

        self._update_statistic_colorbar()
        self._update_statistic_colors()

    def _statistic_colormap(self):

        cmap = color.get_colormap(STATISTIC_CMAPS[self.statistic_cmap_name])

        if not self.statistic_cmap_reversed:
            return cmap

        return color.Colormap(cmap.map(np.linspace(1.0, 0.0, 256)))

    def _update_statistic_colorbar(self):

        visible = (
            self.neuron_statistic_values is not None and self.statistic_clim is not None
        )

        self.statistic_colorbar.visible = visible
        self.statistic_colorbar_bg.visible = visible

        for text in (
            self.statistic_colorbar_low,
            self.statistic_colorbar_high,
            self.statistic_colorbar_title,
        ):
            text.visible = visible

        if not visible:
            return

        lo, hi = self.statistic_clim

        self.statistic_colorbar.cmap = self._statistic_colormap()

        lo_text, hi_text = format_colorbar_limits(
            lo,
            hi,
        )

        self.statistic_colorbar_low.text = lo_text
        self.statistic_colorbar_high.text = hi_text

        self.statistic_colorbar_title.text = self._format_colorbar_title(
            self.statistic_title or ""
        )

    def _update_statistic_colors(self):
        """
        Recolor existing neuron visuals from the current statistic.

        Does not rebuild positions/footprints.
        """

        for key, record in self.plotting["data"].items():

            visual = self.plotting["visuals"].get(key)

            if visual is None:
                continue

            # -------------------------------------------
            # No statistic selected:
            # restore ordinary overview colors
            # -------------------------------------------
            if self.neuron_statistic_values is None:

                if key == "union":
                    colors = None
                    style = "background"
                else:
                    colors = self.state.session_colors[key]

                    style = (
                        "default"
                        if key == self.state.current_session_id
                        else "background"
                    )

            # -------------------------------------------
            # Statistic selected
            # -------------------------------------------
            else:
                colors = self._statistic_colors_for_record(record)

                # For now all statistic-colored data use the
                # ordinary visual style. The colors themselves
                # encode the statistic.
                style = "default"

            plot_options = self.styles.get_plot_options(
                style=style,
                plot_type="marker",
                values=record.vals,
                colors=colors,
                edge_width=0,
            )

            visual.set_data(
                record.pos,
                **plot_options,
            )

        self.update()

    def _format_colorbar_title(
        self,
        title: str,
    ) -> str:
        return "\n".join(
            textwrap.wrap(
                title,
                width=24,
                break_long_words=False,
            )
        )

    ### ==================================================================== ###
    ### ======================= GENERAL STYLE HELPER ======================= ###
    ### ==================================================================== ###

    def update_session_styles(self, session_id):

        if session_id not in self.plotting["visuals"]:
            return

        if session_id == "union":
            self.plotting["visuals"][session_id].visible = True
        else:
            self.plotting["visuals"][session_id].visible = self.data.sessions[
                session_id
            ].active

        self._update_statistic_colors()

    def update_style(self, component, style="default"):

        if component is None:
            self.plotting["overlays"][style].visible = False
            self.update()
            return

        if not isinstance(component, list):
            component = [component]

        if self.display_mode == "session_overview":
            key = self.state.current_session_id
        else:
            key = "union"

        if key is None:
            ## can happen on session unregistration
            self.plotting["overlays"][style].visible = False
            self.update()
            return

        neuron_ids = [c.neuron_id for c in component]
        mask = np.isin(self.plotting["data"][key].ids, neuron_ids)
        if not np.any(mask):
            self.plotting["overlays"][style].visible = False
            self.update()
            return
        vals = self.plotting["data"][key].vals[mask]
        vals += 0.5

        plot_options = self.styles.get_plot_options(style, "marker", vals, edge_width=0)

        self.plotting["overlays"][style].set_data(
            self.plotting["data"][key].pos[mask].astype(np.float32),
            **plot_options,
        )
        self.plotting["overlays"][style].visible = True
        self.update()

    def find_closest_component(self, mouse_pos) -> Optional[NeuronComponent]:
        # print("neurons:", self.data.neurons)
        if (
            self.state.current_session_id is None
            or "background" not in self.plotting
            or self.data.assignments is None
            or self.data.assignments.union is None
            or self.data.assignments.union.centroids is None
        ):
            return None

        mouse_pos = click_events.canvas_to_visual(
            self.plotting["background"], mouse_pos
        )

        # union_centroids = np.nanmean(self.data.neurons.centroids, axis=1)
        union_centroids = self.data.assignments.union.centroids

        # find closest footprint
        distances = (union_centroids[:, 0] - mouse_pos[0]) ** 2 + (
            union_centroids[:, 1] - mouse_pos[1]
        ) ** 2
        if self.display_mode == "session_overview":
            mask = self.state.assignments[:, self.state.current_session_id] >= 0
            distances[~mask] = np.inf

        neuron_id = np.argmin(distances).astype(int)
        if np.sqrt(distances[neuron_id]) > 10.0:
            return None

        return NeuronComponent(self.state.current_session_id, neuron_id)


class Controller(BasePlot.CanvasController):

    canvas: Display  # type hint for better code completion

    def build_controls(self):

        self.current_statistic_query = None
        self.statistic_table = None

        self.controls["statistics"] = StatisticQuerySelector(
            engine=self.data.statistic_engine,
        )
        self.section.x_options_layout.addWidget(self.controls["statistics"])

        self.controls["statistics"].set_default_reduction_provider(
            self._overview_statistic_defaults
        )
        self.controls["statistics"].queryChanged.connect(
            self._on_statistic_query_changed
        )
        self.controls["statistics"].set_query_mode("neuron_bound")
        self.data.statistic_engine.values_changed.connect(self._recalculate_statistic)

        self.controls["stat_cmap_label"] = QLabel("Colormap:")

        self.controls["stat_cmap"] = QComboBox()
        self.controls["stat_cmap"].addItems(STATISTIC_CMAPS.keys())

        self.controls["stat_reverse"] = QCheckBox("Reverse")

        self.section.x_options_layout.addWidget(self.controls["stat_cmap_label"])
        self.section.x_options_layout.addWidget(self.controls["stat_cmap"])
        self.section.x_options_layout.addWidget(self.controls["stat_reverse"])

        self.controls["stat_cmap"].currentTextChanged.connect(
            self._on_statistic_color_options_changed
        )

        self.controls["stat_reverse"].toggled.connect(
            self._on_statistic_color_options_changed
        )

    def _on_statistic_query_changed(self, query):
        self.current_statistic_query = query
        self._recalculate_statistic()

    def _recalculate_statistic(self):

        query = self.current_statistic_query

        if query is None:
            self.statistic_table = None
            self.canvas.set_neuron_statistic_values(
                None,
                title=None,
            )
            return

        def evaluate(ctx=None):
            return self.data.statistic_engine.evaluate_table(query)

        self.state.tasks.start(
            "calculating",
            f"Calculate {query.statistic_key}",
            evaluate,
            on_result=lambda table, query=query: self._on_statistic_ready(
                query,
                table,
            ),
        )

    def _on_statistic_ready(
        self,
        query,
        table,
    ):
        # User changed the selector while this calculation
        # was queued/running.
        if query != self.current_statistic_query:
            return

        self.statistic_table = table

        if table is None:
            self.canvas.set_neuron_statistic_values(
                None,
                title=None,
            )
            return
        values = self._dense_neuron_values(table)

        self.canvas.set_neuron_statistic_values(
            values,
            title=table.stat.title,
        )

    def _dense_neuron_values(self, table):

        neuron_dim = neuron_bound_dim(table.dims)

        if neuron_dim is None:
            raise ValueError(
                "Neuron-bound query returned no neuron dimension: " f"{table.dims}"
            )

        return table.indexed_values(
            neuron_dim,
            self.state.assignments.shape[0],
        )

    def _overview_statistic_defaults(
        self,
        stat_def,
    ):
        reductions = {}

        session_dims = [dim for dim in stat_def.dims if dim in SESSION_DIMS]

        if self.section.display_mode == "session_overview":
            session_id = self.state.current_session_id

            if session_id is not None:
                for dim in session_dims:
                    reductions[dim] = ReductionSpec(
                        "single",
                        index=session_id,
                    )

        elif self.section.display_mode == "tracked_overview":
            for dim in session_dims:
                reductions[dim] = ReductionSpec("mean")

        return reductions

    def _on_statistic_color_options_changed(
        self,
        *_,
    ):
        self.canvas.set_statistic_colormap(
            self.controls["stat_cmap"].currentText(),
            reverse=self.controls["stat_reverse"].isChecked(),
        )

    def initialize_display(self):

        self.canvas.clean()
        self.canvas.initialize_overlays()

        self._on_session_changed()

    def _on_data_changed(self, input: Tuple[str, int]):
        data_type, data_val = input

        if (
            data_type == "sessions"
            and self.section.display_mode == "session_overview"
            and data_val != self.state.current_session_id
        ):
            return

        self.initialize_display()

    def _on_session_changed(self):
        if (
            self.data.assignments is None
            or self.data.assignments.ids.shape[1] == 0
            or self.state.current_session_id is None
        ):
            return

        self.canvas.plot_background()
        if (
            self.section.display_mode == "tracked_overview"
            # and self.data.neurons is not None
            and "union" not in self.canvas.plotting["data"]
        ):
            self.canvas.build_neuron_visuals_union()

        self.canvas.clean(with_union=self.section.display_mode == "session_overview")

        self.canvas.build_neuron_visuals_session(
            session_id=self.state.current_session_id, thr=0.2
        )
        self.canvas.plot_neurons_visuals()
        # self.canvas.update_session_styles()

        super()._on_session_changed()

    def _on_session_style_changed(self, session_id):
        self.canvas.update_session_styles(session_id)

    def _on_selection_changed(self):
        # self.side_menu.update_display()
        super()._on_selection_changed()

    def _on_focus_changed(self):
        # self.side_menu.highlight_display()
        super()._on_focus_changed()

    def update_neuron_selection(self):
        self.update_styles()

    def deactivate(self):
        # self.side_menu.parent = None
        # self.side_menu.deleteLater()
        self.canvas.clean()
        super().deactivate()


def sparse_A_to_points(A_csc, dims, thr=0.2):
    """
    A_csc: scipy.sparse.csc_matrix, shape (d, n)
    dims: (H, W)
    Returns:
        pos: (Nnz, 2) float32, (x,y) pixel coords
        roi: (Nnz,) int32, neuron index for each point
        val: (Nnz,) float32, footprint weight
    """
    H, W = dims

    # max_vals = A_csc.max(axis=0).toarray()
    # print("###", max_vals.shape, A_csc.shape)

    # Normalize each column by its 99th percentile

    A = normalize_sparse_array(A_csc, relative_threshold=thr, format="coo")
    pix = A.row.astype(np.int64)
    roi = A.col.astype(np.int32)
    val = A.data.astype(np.float32)
    val /= np.percentile(val, 99) + 1e-8  # normalize to [0,1] for color mapping

    x = (pix // W).astype(np.float32)
    y = (pix % W).astype(np.float32)

    pos = np.column_stack([y, x]).astype(np.float32)  # (Nnz, 2)
    return pos, roi, val


def normalize_sparse_array(
    A, relative_threshold=0.001, minimum_nonzero_entries=10, format="csc"
):
    """
    normalizing sparse arrays with some thresholding
      - relative_threshold: float
          fraction of peak value, at below which entries are considered to be 0
      - minimum_nonzero_entries: int
          minimum number of nonzero entries of the footprint to be considered for further analyses
    """
    return sparse.vstack(
        [
            # a.multiply(a>relative_threshold*a.max())/a.max()  # threshold footprint
            (
                a.multiply(a > (relative_threshold * a.max()))
                / a[a > 0.001 * a.max()].sum()  # threshold footprint
                if (a > 0).sum()
                > minimum_nonzero_entries  # require minimum non-zero entries ...
                else sparse.csr_matrix(a.shape)
            )  # ... otherwise return empty slice
            for a in A.T  # loop through all footprints
        ],
        format=format,
    ).T


def format_colorbar_limits(
    lo: float,
    hi: float,
    significant_digits: int = 4,
) -> tuple[str, str]:

    max_abs = max(abs(lo), abs(hi))

    if max_abs == 0:
        return ("0", "0")

    order = int(np.floor(np.log10(max_abs)))

    if order >= 5 or order <= -4:
        return (
            f"{lo:.{significant_digits - 1}e}",
            f"{hi:.{significant_digits - 1}e}",
        )

    decimals = max(
        0,
        significant_digits - 1 - order,
    )

    return (
        f"{lo:.{decimals}f}",
        f"{hi:.{decimals}f}",
    )
