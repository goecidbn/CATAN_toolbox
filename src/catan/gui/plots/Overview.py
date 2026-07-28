from typing import Dict, Optional, Tuple
import numpy as np
from scipy import sparse
from vispy import scene, color
from vispy.scene import visuals


from dataclasses import dataclass

from catan.gui.structures.state import NeuronComponent
from catan.gui.plots import BasePlot
from catan.gui.interaction import click_events

print("reloading Overview.py")


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

        self.unfreeze()
        self.grid = self.central_widget.add_grid(spacing=0)
        self.view = self.grid.add_view(row=0, col=0)
        self.view.stretch = (1, 1)  # expand a lot
        ## set camera
        self.view.camera = scene.PanZoomCamera(aspect=None)
        self.view.camera.interactive = False

        self.plot_root = scene.Node(parent=self.view.scene)

        self.changes_on_click = "selected"

        # self.plot_components: Dict[int, OverviewVisualRecord] = {}

        self.freeze()

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

        background = self.data.current_session.Cn
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

        roi_pos, neuron_ids, roi_vals = sparse_A_to_points(
            self.data.union.A, self.data.sessions[0].dims, thr=0.5
        )

        self.plotting["data"]["union"] = OverviewRecord(
            pos=roi_pos,
            ids=neuron_ids,
            vals=roi_vals,
            n_rois=self.data.union.n_neurons,
            color=None,
        )

    def build_neuron_visuals_session(self, session_id=None, thr=0.2):

        if session_id is None:
            session_id = self.state.current_session_id
        if (
            session_id is None
            or session_id >= len(self.data.sessions)
            or self.data.sessions[session_id] is None
        ):
            return

        if self.display_mode == "tracked_overview" and "union" in self.plotting["data"]:
            neuron_ids = np.where(
                self.state.assignments[:, self.state.current_session_id] >= 0
            )[0]
            mask = np.isin(self.plotting["data"]["union"].ids, neuron_ids)
            roi_pos = self.plotting["data"]["union"].pos[mask]
            roi_vals = self.plotting["data"]["union"].vals[mask]
            n_rois = len(neuron_ids)
        else:
            roi_pos, roi_ids, roi_vals = sparse_A_to_points(
                self.data.sessions[session_id].A,
                self.data.sessions[session_id].dims,
                thr=thr,
            )
            n_rois = self.data.sessions[session_id].A.shape[1]

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

            plot_options = self.styles.get_plot_options(
                style="background" if key == "union" else "default",
                plot_type="marker",
                values=self.plotting["data"][key].vals,
                colors=self.plotting["data"][key].color,
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

    def update_session_styles(self, session_id):

        # print(
        #     f"Updating session {key}, plotting data: {self.plotting['data'][key]}"
        # )
        if session_id not in self.plotting["visuals"]:
            return
        if session_id == "union" or session_id == self.state.current_session_id:
            style = "default"
        else:
            style = "background"

        if session_id == "union":
            self.plotting["data"][session_id].color = None
        else:
            self.plotting["data"][session_id].color = self.state.session_colors[
                session_id
            ]

        plot_options = self.styles.get_plot_options(
            style,
            "marker",
            self.plotting["data"][session_id].vals,
            colors=self.plotting["data"][session_id].color,
            edge_width=0,
        )
        print("updating style for session", session_id, "with style", plot_options)

        self.plotting["visuals"][session_id].set_data(
            self.plotting["data"][session_id].pos, **plot_options
        )
        if session_id == "union":
            self.plotting["visuals"][session_id].visible = True
        else:
            self.plotting["visuals"][session_id].visible = self.state.session_active[
                session_id
            ]  # [1]

        self.update()

    def update_style(self, component, style="default"):

        if component is None:
            self.plotting["overlays"][style].visible = False
            self.update()
            return

        if not isinstance(component, list):
            component = [component]

        if self.display_mode == "single_overview":
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
            self.data.current_session is None
            or "background" not in self.plotting
            or self.data.union.centroids is None
        ):
            return None

        mouse_pos = click_events.canvas_to_visual(
            self.plotting["background"], mouse_pos
        )

        # union_centroids = np.nanmean(self.data.neurons.centroids, axis=1)
        union_centroids = self.data.union.centroids
        # find closest footprint
        distances = (union_centroids[:, 0] - mouse_pos[0]) ** 2 + (
            union_centroids[:, 1] - mouse_pos[1]
        ) ** 2
        neuron_id = np.argmin(distances).astype(int)
        if np.sqrt(distances[neuron_id]) > 10.0:
            return None

        return NeuronComponent(self.state.current_session_id, neuron_id)


class Controller(BasePlot.CanvasController):

    canvas: Display  # type hint for better code completion

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
        if self.state.current_session_id is None:
            return

        self.canvas.plot_background()
        if (
            self.section.display_mode == "tracked_overview"
            # and self.data.neurons is not None
            and "union" not in self.canvas.plotting["data"]
        ):
            # print("initializing tracked overview display")
            self.canvas.build_neuron_visuals_union()

        # if self.section.display_mode == "session_overview":
        self.canvas.clean(with_union=self.section.display_mode == "session_overview")

        self.canvas.build_neuron_visuals_session(
            session_id=self.state.current_session_id, thr=0.2
        )
        self.canvas.plot_neurons_visuals()
        # self.canvas.update_session_styles()

        super()._on_session_changed()

        # for key in self.canvas.plotting["data"]:
        #     if isinstance(key, int) and self.section.display_mode == "tracked_overview":

        #         self.state.session_active[key] = (
        #             key,
        #             key == self.state.current_session_id,
        #         )

        # print(f"Session {key} active: {self.state.session_active[key]}")

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


def colormap(
    values: np.ndarray,
    base_color: list | tuple | str = "viridis",
    alpha_scale: float = 1.0,
    offset=0.6,
):
    """
    values: array-like, assumed normalized to [0,1]
    returns: (N,4) RGBA array
    """
    # print(f"cmap which: {which}")
    val_min, val_max = np.percentile(values, [5, 95])
    v = (values - val_min) / (val_max - val_min + 1e-8)
    v = np.clip(v, 0.0, 1.0)

    # optional offset (keeps low values visible)
    if offset > 0:
        v = offset + (1.0 - offset) * v

    if isinstance(base_color, str):
        # map to RGBA using cmap
        cmap = color.get_colormap(base_color)
        rgba = cmap.map(v).astype(np.float32)

        # control alpha separately (very useful for overlap)
        rgba[:, 3] *= alpha_scale

        return rgba
    else:
        rgba = np.tile(base_color, (len(v), 1))
        rgba[:, 3] = v * alpha_scale
        return rgba.astype(np.float32)
