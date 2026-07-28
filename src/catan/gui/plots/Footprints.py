import numpy as np
from dataclasses import dataclass
from typing import Optional, Tuple, Dict

from vispy import scene, color
from vispy.scene import visuals, cameras
from vispy.color import Colormap

from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QCursor, QAction
from PySide6.QtWidgets import (
    QDoubleSpinBox,
    QCheckBox,
    QLabel,
    QHBoxLayout,
    QVBoxLayout,
    QWidget,
    QMenu,
    QWidgetAction,
    QPushButton,
)

import importlib

from catan.gui.plots import BasePlot
from catan.gui.plots.helper import FootprintSlider
from catan.gui.interaction import click_events

from catan.gui.structures.state import NeuronComponent
from catan.gui.GUI_elements.utils.menu_creation import (
    get_colored_label,
    add_label_to_menu,
)

from catan.core.image_correlation import calculate_img_correlation

importlib.reload(FootprintSlider)


@dataclass
class FootprintRecord:
    key: Tuple[int, int]  # e.g. (session_id, footprint_id)

    vertex_start: int
    vertex_stop: int
    face_start: int
    face_stop: int

    pick_points: np.ndarray  # (K, 3), usually same as vertices or downsampled
    center_xyz: np.ndarray  # (3,)


@dataclass
class NeuronRecord:
    mesh_vertices: np.ndarray  # (N, 3)
    mesh_faces: np.ndarray  # (M, 3)
    mesh_colors_default: np.ndarray  # (N, 4)

    thr: float  # threshold used for this footprint (store to check if changed)


class Display(BasePlot.BaseCanvas):

    def __init__(self, display_section, controls, config=None):
        super().__init__(display_section, controls, config)

        ## should be cleaned up / tested if necessary here (grid needed if no axis, ...?)
        self.unfreeze()
        self.grid = self.central_widget.add_grid(spacing=0)
        self.view = self.grid.add_view(row=0, col=0)
        self.view.stretch = (1, 1)  # expand a lot
        self.view.camera = cameras.TurntableCamera()

        self.changes_on_click = "highlighted"

        self.plot_root = scene.Node(parent=self.view.scene)

        self.build_overlays()
        self.clear()
        self.camera_set = False

        self.freeze()

    def clear(self):

        self.clear_visuals()
        self.plotting["data"] = {}

        # # reset root to clear properly (needed?)
        # old_root = self.plot_root
        # old_root.parent = None
        # self.plot_root = scene.Node(parent=self.view.scene)

    def clear_visuals(self):
        for key, footprint_visual in self.plotting["visuals"].items():
            footprint_visual.visible = False
            footprint_visual.parent = None
        self.plotting["visuals"] = {}
        self.clear_highlights()

    def clear_highlights(self):

        # for key in ["hovered", "focused", "highlighted"]:
        #     self.plotting["overlays"][key].visible = False

        for key, overlay in self.plotting["overlays"].items():
            if overlay is None:
                continue
            overlay.visible = False
            # self.plotting["overlays"][key].visible = False

    def build_overlays(self):

        # for key in ["hovered", "focused", "highlighted"]:
        for key in ["hovered", "focused", "highlighted"]:
            self.plotting["overlays"][key] = visuals.Mesh(parent=self.plot_root)
            self.plotting["overlays"][key].visible = False
            self.plotting["overlays"][key].set_gl_state(
                blend=True,
                depth_test=False,
                blend_func=("src_alpha", "one_minus_src_alpha"),
            )
        self.plotting["overlays"]["hovered"].order = 100
        self.plotting["overlays"]["focused"].order = 80
        self.plotting["overlays"]["highlighted"].order = 90

    def plot_neurons(self, single=False, reset=False):

        # self.state.logger.debug(
        #     f"Plotting footprints for neurons: {self.state.selected_components}"
        # )

        self.state.timeit()

        if reset:
            self.clear()
        else:
            self.clear_visuals()

        if self.data is None or self.state.selected_components is None:
            return

        z_stretch = self.controls["parameter"].z_stretch_spin.value()
        self.state.timeit("Initial setup")

        ## identify the neurons thata should be plotted
        this_neuron = self.state.focused_component.neuron_id
        if single:
            session_only = self.controls["parameter"].checkbox_session_only.isChecked()

            ## find closeby neurons
            adjacent_radius = self.controls["parameter"].adj_radius_spin.value()
            to_plot_neurons = []
            if adjacent_radius > 1e-2:
                # union_centroids = np.nanmean(self.data.neurons.centroids, axis=1)
                union_centroids = self.data.union.centroids
                distances = np.linalg.norm(
                    union_centroids - union_centroids[this_neuron],
                    axis=1,
                )

                adjacent_neurons = np.where(distances <= adjacent_radius)[0]
                to_plot_neurons.extend(adjacent_neurons)
        else:
            session_only = False
            to_plot_neurons = [c.neuron_id for c in self.state.selected_components]

        self.state.timeit("Found neurons to plot and calculated centroid ranges")

        ## iterate through each neuron that should be plotted
        for neuron in to_plot_neurons:
            key = "focused" if neuron == this_neuron else "default"
            alpha = 0.8 if key == "focused" else self.styles.opts[key]["alpha"]
            # alpha = self.styles.opts[key]["alpha"]

            ## create new data if not yet present
            if neuron not in self.plotting["data"]:
                self.add_footprint_data(
                    neuron,
                    alpha=alpha,
                    z_stretch=z_stretch,
                )

            if neuron != this_neuron and session_only:
                ## select subset of data for session-only display ...

                session_id = self.controls["parameter"].session_index_spin.value()
                component = NeuronComponent(session_id, neuron)

                rec = self.plotting["data"].get(component.id, None)
                if rec is None:
                    continue

                verts = self.plotting["data"][neuron].mesh_vertices[
                    rec.vertex_start : rec.vertex_stop
                ]
                faces = (
                    self.plotting["data"][neuron].mesh_faces[
                        rec.face_start : rec.face_stop
                    ]
                    - rec.vertex_start
                )
                cols = self.plotting["data"][neuron].mesh_colors_default[
                    rec.vertex_start : rec.vertex_stop
                ]

            else:
                ## ... or select all, to display for all sessions
                verts = self.plotting["data"][neuron].mesh_vertices
                faces = self.plotting["data"][neuron].mesh_faces
                cols = self.plotting["data"][neuron].mesh_colors_default

            # cols[:, 3] = alpha  ## apply alpha (to account for selection changes)

            plot_options = self.styles.get_plot_options(
                key, "mesh", verts[:, 2], alpha=alpha, colors=cols
            )

            ## plot the actual data
            mesh = visuals.Mesh(
                vertices=verts,
                faces=faces,
                **plot_options,
                shading="smooth",
                parent=self.plot_root,
            )

            mesh.set_gl_state(
                blend=True,
                depth_test=True,
                blend_func=("src_alpha", "one_minus_src_alpha"),
            )
            mesh.order = 10
            self.state.timeit("Added mesh")

            self.plotting["visuals"][neuron] = mesh

        self.state.timeit("Plotted neurons")
        self.reset_camera(full=not self.camera_set)

    def reset_camera(self, full=False):

        if not self.plotting["visuals"]:
            return

        ## obtain currently displayed neurons and their centroids
        displayed_neurons = list(self.plotting["visuals"].keys())
        # union_centroids = np.nanmean(
        #     self.data.neurons.centroids[displayed_neurons, ...], axis=1
        # )
        union_centroids = self.data.union.centroids[displayed_neurons, ...]

        centroid_min = np.nanmin(union_centroids, axis=0)
        centroid_max = np.nanmax(union_centroids, axis=0)

        distance_threshold = self.controls["parameter"].distance_spin.value()

        ## calculate ranges
        x_range = np.clip(
            np.array([centroid_min[0], centroid_max[0]])
            + np.array([-distance_threshold, distance_threshold]),
            0,
            self.data.current_session.dims[1],
        )
        y_range = np.clip(
            np.array([centroid_min[1], centroid_max[1]])
            + np.array([-distance_threshold, distance_threshold]),
            0,
            self.data.current_session.dims[0],
        )

        z_stretch = self.controls["parameter"].z_stretch_spin.value()
        z_range = (0, len(self.data.sessions) * z_stretch)

        if full:
            ## not working well, yet. rotation is somewhat preserved
            self.view.camera.set_range(
                x=x_range,
                y=y_range,
                z=z_range,
            )
            self.view.camera.center = (
                0.5 * (x_range[0] + x_range[1]),
                0.5 * (y_range[0] + y_range[1]),
                0.5 * (z_range[0] + z_range[1]),
            )
            self.camera_set = True
        else:
            ## merely move camera to new center (but keep zoom/ranges)
            self.view.camera.center = (
                0.5 * (x_range[0] + x_range[1]),
                0.5 * (y_range[0] + y_range[1]),
                self.view.camera.center[2],
            )

        self.update()

    def add_footprint_data(self, neuron: int, alpha=0.7, z_stretch=5.0):

        thr = self.controls["parameter"].threshold_spin.value()

        self.state.timeit()

        self.state.logger.debug(
            f"Plotting neuron {neuron} with threshold {thr} and alpha {alpha}"
        )

        vertex_offset = 0
        face_offset = 0

        # or self.plotting["data"][neuron].mesh_vertices is None
        vertices_all = []
        faces_all = []
        colors_all = []
        for session_id in range(len(self.data.sessions)):

            component = NeuronComponent(session_id, neuron)

            fp_id = self.state.get_footprint_from_component(component)
            if fp_id is None or fp_id < 0:
                continue

            rgba = self.state.session_colors[session_id]
            rgba[3] = alpha

            z_offset = float(session_id) * z_stretch

            v, f, c, pick_points = footprint_to_mesh(
                self.data.sessions[session_id].A[:, fp_id],
                self.data.sessions[session_id].dims,
                z_offset=z_offset,
                z_scale=z_stretch,
                z_thr=thr,
                rgba=rgba,
                base_vertex_offset=vertex_offset,
            )

            v0 = vertex_offset
            v1 = vertex_offset + v.shape[0]

            f0 = face_offset
            f1 = face_offset + f.shape[0]

            self.plotting["data"][component.id] = FootprintRecord(
                key=component.id,
                vertex_start=v0,
                vertex_stop=v1,
                face_start=f0,
                face_stop=f1,
                pick_points=pick_points,  # local/world coords, same as vertices
                center_xyz=np.nanmean(v, axis=0),
            )

            vertex_offset += v.shape[0]
            vertices_all.append(v)
            faces_all.append(f)
            colors_all.append(c)

            vertex_offset = v1
            face_offset = f1

        self.plotting["data"][neuron] = NeuronRecord(
            mesh_vertices=np.vstack(vertices_all),
            mesh_faces=np.vstack(faces_all),
            mesh_colors_default=np.vstack(colors_all),
            thr=thr,
        )

        self.state.timeit("Added surfaces")

        # return mesh

    def find_closest_component(self, mouse_pos) -> Optional[NeuronComponent]:

        if not self.plotting["visuals"]:
            return None

        center_radius_px = 80
        point_radius_px = 10

        keys = [
            key
            for key in self.plotting["data"].keys()
            if isinstance(key, tuple) and key[1] in self.plotting["visuals"]
        ]
        centers = np.asarray(
            [self.plotting["data"][key].center_xyz for key in keys],
            dtype=np.float32,
        )

        screen_centers = click_events.visual_to_canvas(
            self.plotting["visuals"][keys[0][1]], centers
        )

        dx = screen_centers[:, 0] - mouse_pos[0]
        dy = screen_centers[:, 1] - mouse_pos[1]
        d2 = dx * dx + dy * dy

        candidate_idx = np.where(d2 <= center_radius_px**2)[0]
        if len(candidate_idx) == 0:
            candidate_idx = np.argsort(d2)[:5]  # fallback nearest centers

        best_key = None
        best_d2 = point_radius_px**2

        for i in candidate_idx:
            key = keys[int(i)]
            rec = self.plotting["data"][key]

            screen = click_events.visual_to_canvas(
                self.plotting["visuals"][keys[int(i)][1]], rec.pick_points
            )

            ddx = screen[:, 0] - mouse_pos[0]
            ddy = screen[:, 1] - mouse_pos[1]
            local_d2 = ddx * ddx + ddy * ddy

            m = float(np.nanmin(local_d2))
            if m < best_d2:
                best_d2 = m
                best_key = key

        return NeuronComponent(*best_key) if best_key is not None else None

    def find_nearby_neurons(
        self, component: NeuronComponent
    ) -> tuple[np.ndarray, np.ndarray]:
        # union_centroids = np.nanmean(self.data.neurons.centroids, axis=1)
        union_centroids = self.data.union.centroids
        neuron_id = component.neuron_id
        d = np.linalg.norm(union_centroids - union_centroids[neuron_id], axis=1)
        try:
            adj_radius = self.controls["parameter"].adj_radius_spin.value()
        except Exception:
            adj_radius = 15.0
        neuron_ids = np.where(d <= adj_radius)[0]
        neuron_ids = [n for n in neuron_ids if n != neuron_id]

        self.state.logger.debug(
            f"Neuron {neuron_id} has {len(neuron_ids)} adjacent neurons within radius {adj_radius}: {neuron_ids}"
        )

        return np.array(neuron_ids), np.array(d[neuron_ids])

    def update_style(self, component, style="default"):

        if style == "selected":
            return

        if component is None:
            self.plotting["overlays"][style].visible = False
            self.update()
            return

        rec = self.plotting["data"].get(component.id, None)
        if rec is None:
            self.plotting["overlays"][style].visible = False
            self.update()
            return

        verts = self.plotting["data"][component.neuron_id].mesh_vertices[
            rec.vertex_start : rec.vertex_stop
        ]
        faces = (
            self.plotting["data"][component.neuron_id].mesh_faces[
                rec.face_start : rec.face_stop
            ]
            - rec.vertex_start
        )

        cols = self.styles.get_color_array(style, verts[:, 2], alpha=0.6)
        self.plotting["overlays"][style].set_data(
            vertices=verts,
            faces=faces,  # .astype(np.uint32),
            vertex_colors=cols,
        )
        self.plotting["overlays"][style].visible = True

        self.update()

    def on_mouse_release(self, event):

        if event.button == 2:  # right click in VisPy
            key = self.find_closest_component(event.pos)
            if key is not None:
                self.open_footprint_context_menu(key)
            event.handled = True
        super().on_mouse_release(event)

    def calculate_tracking_statistics(
        self, this_component: NeuronComponent, ref_neuron: int, with_session: str | int
    ) -> tuple[Optional[float], Optional[float], Optional[float], Optional[int]]:

        this_fp_id = self.state.get_footprint_from_component(this_component)

        ## find, where compared neuron has footprints
        fp_ids = self.state.assignments[ref_neuron, :]
        sessions_detected = np.where(fp_ids >= 0)[0]
        if len(sessions_detected) == 0:
            self.state.logger.debug(
                f"No footprints found for neuron {ref_neuron} to compare with."
            )
            return None, None, None, None

        ## find the reference session to compare with
        if with_session == "previous":
            prev_sessions = sessions_detected[
                sessions_detected < this_component.session_id
            ]
            if len(prev_sessions) == 0:
                self.state.logger.debug(
                    f"No previous session found for neuron {ref_neuron} to compare with."
                )
                return None, None, None, None
            ref_session_id = prev_sessions[-1]
        elif with_session == "next":
            next_sessions = sessions_detected[
                sessions_detected > this_component.session_id
            ]
            if len(next_sessions) == 0:
                self.state.logger.debug(
                    f"No next session found for neuron {ref_neuron} to compare with."
                )
                return None, None, None, None
            ref_session_id = next_sessions[0]
        elif isinstance(with_session, int):
            if with_session not in sessions_detected:
                self.state.logger.debug(
                    f"Neuron {ref_neuron} does not have a footprint in session {with_session}."
                )
                return None, None, None, None
            ref_session_id = with_session
        else:
            raise ValueError(f"Invalid with_session value: {with_session}")

        ## decided on the reference component to compare with
        ref_component = NeuronComponent(ref_session_id, ref_neuron)
        ref_fp_id = self.state.get_footprint_from_component(ref_component)

        ## calculate the statistics
        similarity, _, shift = calculate_img_correlation(
            A1=self.data.sessions[this_component.session_id].A[:, this_fp_id],
            A2=self.data.sessions[ref_component.session_id].A[:, ref_fp_id],
            dims=self.data.sessions[this_component.session_id].dims,
            crop=True,
            binary=False,
            shift=True,
            mode="cosine_union",
            gamma=0.1,
            shift_optimized=True,
        )
        d_shift = np.sqrt(np.sum([s**2 for s in shift]))
        p_same = self.data.f_same(d_shift, similarity)[0]
        return similarity, d_shift, p_same, ref_session_id

    def add_info_str(
        self,
        menu,
        this_component: NeuronComponent,
        candidate: int,
        which: str = "previous",
        cmap=Colormap(["green", "black", "red"]),
    ):
        similarity, shift, p_same, session_id = self.calculate_tracking_statistics(
            this_component, candidate, with_session=which
        )
        if similarity is None or shift is None or p_same is None or session_id is None:
            add_label_to_menu(
                menu,
                f"no {which} session found",
                enabled=False,
            )
            return
        ds = abs(this_component.session_id - session_id)

        arrow = "\u25bc" if which == "previous" else "\u25b2"
        html_probability = get_colored_label(f"p={p_same:.2f}", 1 - p_same, cmap, ["b"])
        html_session = (
            f"\u0394s="
            + get_colored_label(f"{ds}", ds / 10.0, cmap)
            + f"({session_id})"
        )
        html_similarity = "c=" + get_colored_label(
            f"{similarity:.2f}", (1 - similarity) / 2.0, cmap
        )
        html_shift = "shift=" + get_colored_label(f"{shift:.2f}px", shift / 10.0, cmap)

        add_label_to_menu(
            menu,
            f"{arrow} {html_probability} - {html_session},\t{html_similarity},\t{html_shift}",
            enabled=False,
        )

    def open_footprint_context_menu(self, this_component: NeuronComponent):

        ## find adjacent neurons
        neuron_candidates, distances = self.find_nearby_neurons(this_component)
        sorted_idxs = np.argsort(distances)

        menu = QMenu(self.native)
        menu.addSection(
            f"Neuron {this_component.neuron_id} in Session {this_component.session_id}"
        )

        ## === set focus ===
        act_focus = QAction("Set focus", menu)
        act_focus.triggered.connect(lambda: self.set_focused_footprint(this_component))
        menu.addAction(act_focus)

        menu.addSeparator()

        ## === start footprint change submenu ===
        if len(self.data.sessions) > 1:
            submenu_change = QMenu("Change neuron assignment…", menu)

            ## display current match
            this_neuron = this_component.neuron_id

            add_label_to_menu(
                submenu_change,
                f"<b>Current assignment (neuron {this_neuron}, session {this_component.session_id})</b>",
                enabled=False,
            )
            # act_candidate = QAction(
            #     f"Current assignment (neuron {this_neuron}, session {this_component.session_id})",
            #     submenu_change,
            # )
            # submenu_change.addAction(act_candidate)

            self.add_info_str(submenu_change, this_component, this_neuron, "next")
            self.add_info_str(submenu_change, this_component, this_neuron, "previous")
            submenu_change.addSeparator()

            ## display potential other matches
            for candidate in neuron_candidates[sorted_idxs]:

                act_candidate = QAction(f"... to neuron {candidate}", submenu_change)
                act_candidate.triggered.connect(
                    lambda checked, new_neuron=candidate: self.change_neuron_assignment_dialog(
                        this_component, new_neuron
                    )
                )
                submenu_change.addAction(act_candidate)

                self.add_info_str(submenu_change, this_component, candidate, "next")
                self.add_info_str(submenu_change, this_component, candidate, "previous")
                submenu_change.addSeparator()

            ## Option to create new neuron for this footprint
            act_candidate = QAction("... to new neuron", submenu_change)
            act_candidate.triggered.connect(
                lambda: self.change_neuron_assignment_dialog(this_component)
            )
            submenu_change.addAction(act_candidate)
            menu.addMenu(submenu_change)
            menu.addSeparator()

        ## === flag options ===
        submenu_flag = QMenu("Flag neuron ...", menu)

        submenu_flag.addAction(QAction("... as uncertain", submenu_flag))
        submenu_flag.addAction(QAction("... as certain", submenu_flag))
        submenu_flag.addSeparator()
        submenu_flag.addAction(QAction("... for merge", submenu_flag))
        submenu_flag.addAction(QAction("... for split", submenu_flag))
        submenu_flag.addAction(QAction("... for removal", submenu_flag))
        menu.addMenu(submenu_flag)

        if not self.data.sessions[this_component.session_id].status["traces_loaded"]:

            load_action = QAction("Load trace for this session", menu)
            menu.addAction(load_action)

            load_action.triggered.connect(
                lambda: self.data.change_trace_presence(this_component.session_id, True)
            )

        menu.exec(QCursor.pos())

    def change_neuron_assignment_dialog(
        self, component: NeuronComponent, new_neuron: Optional[int] = None
    ):
        # Implement a dialog to change the neuron assignment for the given footprint
        # This is a placeholder for the actual implementation
        # print(
        #     f"Change neuron assignment for footprint: {old_component} to {new_component}"
        # )
        """
        [1h] find all adjacent neurons (within distance threshold) and show a dialog to select one
            * [x] calculate and display distance + similarity + probability
            * [x] sort neurons according to calculated probability

        [2h] on selection, update the previous neuron and the new neuron
            * [x] update assignments (in state)
            * [x] remove & insert footprint data in old & new data.neurons
            * [x] recalculate neuron statistics and union A & centroid for both neurons
            * remove old neuron if empty

        [2h] update plots:
            * overview: union shape of both neurons, display table?
            * [x] footprints: update plotting data for both neurons, update selection/focus if needed

        further things:
            * [10h] calculate general neuron/tracking statistics & session-to-session data:
                * [x] distance, similarity, probability
                * [x] shift
                * # components / neuron
                * overall matching goodness (high prob everywhere? few shifts?, few size differences?)
                    * and highlight / allow filter
            * [x] implement "display session x only" for footprints
            * [1h] implement moving up/down sessions with arrow keys for highlighted footprint
            * [x] properly check alpha options for footprints
            * [x] change alpha on already registered components (when selecting others, etc)
            * [30min] enable zooming out when changing display distaance (without full camera reset)
            * [x] implement "unassign footprint" (create new neuron for this footprint)

        """
        session_id = component.session_id
        fp_id = self.state.get_footprint_from_component(component)

        if new_neuron is None:
            ## append new neuron
            new_neuron = self.state.assignments.shape[0]
            self.state.logger.debug(
                f"Creating new neuron {new_neuron} for footprint {fp_id} in session {session_id}"
            )
            self.state.assignments = np.pad(
                self.state.assignments,
                ((0, 1), (0, 0)),
                mode="constant",
                constant_values=-1,
            )

            # self.data.neurons.centroids = np.pad(
            #     self.data.neurons.centroids,
            #     ((0, 1), (0, 0), (0, 0)),
            #     mode="constant",
            #     constant_values=np.nan,
            # )
            ## union footprints is dict - no need to pad

        ## change assignments array
        self.state.assignments[new_neuron, session_id] = fp_id
        self.state.assignments[component.neuron_id, session_id] = -1

        ## update neuron data for both neurons
        # self.data.rebuild_neuron(new_neuron, self.state.assignments)
        # self.data.rebuild_neuron(component.neuron_id, self.state.assignments)

        # self.data.rebuild_neuron(new_neuron, self.state.assignments)
        # self.data.rebuild_neuron(component.neuron_id, self.state.assignments)

        if np.all(self.state.assignments[component.neuron_id, :] < 0):
            print(f"Neuron {component.neuron_id} is now empty and will be removed.")
            print(
                f"CAREFUL!! all references are f**cked up now, need to update all neurons"
            )

        #### for now hardcoded true single mode, asa it can only be done in single mode
        self.plot_neurons(single=True, reset=True)

    def set_focused_footprint(self, component: NeuronComponent):
        self.state.focused_component = component


class Controller(BasePlot.CanvasController):

    def disconnect_signals(self):
        super().disconnect_signals()
        self.controls["parameter"].data_parameter_changed.disconnect()
        self.controls["parameter"].display_parameter_changed.disconnect()

    def build_controls(self):
        super().build_controls()

        self.controls["slider"] = FootprintSlider.FootprintSliderController(
            self.section
        )
        self.section.x_options_layout.addWidget(self.controls["slider"])
        # print("Added footprint slider to controls")

        self.controls["parameter"] = FootprintParametersController(
            self.section, single_mode=self.single_mode
        )
        self.section.y_options_layout.addWidget(self.controls["parameter"])

        self.controls["parameter"].data_parameter_changed.connect(
            lambda: self.replot_neurons()
        )
        self.controls["parameter"].display_parameter_changed.connect(
            lambda: self.update_neuron_selection()
        )
        self.controls["parameter"].reset_camera_requested.connect(
            lambda: self.canvas.reset_camera(full=True)
        )
        # self.controls["parameter"].session_only_changed.connect(
        #     lambda: self._on_session_only_changed()
        # )

    def _on_data_changed(self, input: Tuple[str, int]):
        if input[0] == "assignments":
            self.replot_neurons()

    def initialize_display(self):
        if self.single_mode:
            self.controls["parameter"].session_only_setup()

        super().initialize_display()

    def _on_session_changed(self):
        if self.single_mode:
            self.controls["parameter"].session_only_setup()

    def _on_session_only_changed(self):
        # self.controls["parameter"]._on_session_only_changed()
        self.update_neuron_selection()

    def _on_selection_changed(self):
        self.controls["slider"].update_setup()
        super()._on_selection_changed()

    def _on_focus_changed(self):
        self.controls["slider"].adjust_id()
        super()._on_focus_changed()

    def update_neuron_selection(self):
        self.canvas.plot_neurons(single=self.single_mode)
        self.update_styles()

    def replot_neurons(self):
        self.canvas.plot_neurons(single=self.single_mode, reset=True)
        self.update_styles()


class FootprintParametersController(QWidget):

    data_parameter_changed = Signal()
    display_parameter_changed = Signal()
    session_only_changed = Signal()
    reset_camera_requested = Signal()

    def __init__(self, parent, single_mode=False):
        super().__init__(parent)

        self.state = parent.state

        layout = QVBoxLayout(self)
        layout.addWidget(QLabel("Parameters"))

        ## add footprint threshold control
        initial_threshold = 0.1
        self.threshold_spin = QDoubleSpinBox()
        self.threshold_spin.setDecimals(3)
        self.threshold_spin.setRange(0.0, 1.0)
        self.threshold_spin.setSingleStep(0.01)
        self.threshold_spin.setValue(initial_threshold)

        layout.addWidget(QLabel("Threshold"))
        layout.addWidget(self.threshold_spin)

        ## add distance threshold control
        initial_distance = 25
        self.distance_spin = QDoubleSpinBox()
        self.distance_spin.setDecimals(0)
        self.distance_spin.setRange(0.0, 100.0)
        self.distance_spin.setSingleStep(5.0)
        self.distance_spin.setValue(initial_distance)

        layout.addWidget(QLabel("Distance (px)"))
        layout.addWidget(self.distance_spin)

        ## z_stretch control (for 3D visualization)
        initial_z_stretch = 5.0
        self.z_stretch_spin = QDoubleSpinBox()
        self.z_stretch_spin.setDecimals(1)
        self.z_stretch_spin.setRange(1.0, 10.0)
        self.z_stretch_spin.setSingleStep(0.5)
        self.z_stretch_spin.setValue(initial_z_stretch)

        layout.addWidget(QLabel("Z Stretch"))
        layout.addWidget(self.z_stretch_spin)

        ## add reset camera button
        self.reset_camera_button = QPushButton("Reset Camera")
        self.reset_camera_button.clicked.connect(
            lambda: self.reset_camera_requested.emit()
        )
        layout.addWidget(self.reset_camera_button)

        ## add adjacency radius control
        if single_mode:
            initial_adj_radius = 15
            self.adj_radius_spin = QDoubleSpinBox()
            self.adj_radius_spin.setDecimals(1)
            self.adj_radius_spin.setRange(0.0, 50.0)
            self.adj_radius_spin.setSingleStep(1.0)
            self.adj_radius_spin.setValue(initial_adj_radius)

            layout.addWidget(QLabel("Adjacency radius (px)"))
            layout.addWidget(self.adj_radius_spin)

            ## checkbox for toggling single session display
            self.checkbox_session_only = QCheckBox("Show single session only")
            self.checkbox_session_only.setCheckable(True)
            self.checkbox_session_only.setChecked(False)

            layout.addWidget(self.checkbox_session_only)

            ## spinbox for selecting session index
            self.session_index_spin = QDoubleSpinBox()
            self.session_index_spin.setDecimals(0)
            self.session_index_spin.setRange(0, 1)
            self.session_index_spin.setSingleStep(1)
            self.session_index_spin.setValue(0)

            self.session_only_label = QLabel("Session index")
            layout.addWidget(self.session_only_label)
            layout.addWidget(self.session_index_spin)

            self._on_session_only_changed()  # initial visibility

        layout.addStretch()

        self.threshold_spin.valueChanged.connect(
            lambda: self.data_parameter_changed.emit()
        )
        self.distance_spin.valueChanged.connect(
            lambda: self.display_parameter_changed.emit()
        )
        self.z_stretch_spin.valueChanged.connect(
            lambda: self.data_parameter_changed.emit()
        )
        if single_mode:
            self.adj_radius_spin.valueChanged.connect(
                lambda: self.display_parameter_changed.emit()
            )
            self.checkbox_session_only.toggled.connect(
                lambda: self._on_session_only_changed()
            )
            self.session_index_spin.valueChanged.connect(
                lambda: self._on_session_only_changed()
            )

        # self.controller_layout = layout

    def _on_session_only_changed(self):

        self.session_index_spin.setVisible(self.checkbox_session_only.isChecked())
        self.session_index_spin.setEnabled(self.checkbox_session_only.isChecked())

        self.session_only_label.setVisible(self.checkbox_session_only.isChecked())

    def session_only_setup(self):

        if (
            not hasattr(self.state, "assignments")
            or self.state.current_session_id is None
        ):
            return
        self.session_index_spin.setRange(0, self.state.assignments.shape[1] - 1)
        self.session_index_spin.setValue(self.state.current_session_id)

        # if self.checkbox_session_only.isChecked():
        #     if hasattr(self, "session_index_spin"):
        #         return  # already exists

        #     ## spinbox for selecting session index
        #     self.session_index_spin = QDoubleSpinBox()
        #     self.session_index_spin.setDecimals(0)
        #     self.session_index_spin.setRange(0, self.state.assignments.shape[1])
        #     self.session_index_spin.setSingleStep(1)
        #     self.session_index_spin.setValue(self.state.current_session_id or 0)

        #     self.session_only_label = QLabel("Session index")
        #     self.controller_layout.addWidget(self.session_only_label)
        #     self.controller_layout.addWidget(self.session_index_spin)

        #     self.session_index_spin.valueChanged.connect(
        #         lambda: self.session_only_changed.emit()
        #     )
        # else:
        #     if hasattr(self, "session_index_spin"):
        #         self.controller_layout.removeWidget(self.session_only_label)
        #         self.session_only_label.deleteLater()
        #         del self.session_only_label
        #         self.controller_layout.removeWidget(self.session_index_spin)
        #         self.session_index_spin.deleteLater()
        #         del self.session_index_spin


def add_surface_to_mesh(vertices_all, faces_all, colors_all, X, Y, Z, rgba):
    H, W = Z.shape

    verts = np.column_stack(
        [
            X.ravel(),
            Y.ravel(),
            Z.ravel(),
        ]
    ).astype(np.float32)

    colors = np.tile(np.asarray(rgba, dtype=np.float32), (verts.shape[0], 1))

    faces = []
    for y in range(H - 1):
        for x in range(W - 1):
            i0 = y * W + x
            i1 = i0 + 1
            i2 = i0 + W
            i3 = i2 + 1
            faces.append([i0, i1, i2])
            faces.append([i1, i3, i2])

    faces = np.asarray(faces, dtype=np.uint32)

    offset = sum(v.shape[0] for v in vertices_all)
    faces += offset

    vertices_all.append(verts)
    faces_all.append(faces)
    colors_all.append(colors)


from scipy import sparse


def footprint_to_mesh(
    footprint: sparse.csc_matrix,
    dim: Tuple[int, int],
    *,
    z_offset=0.0,
    z_scale=1.0,
    z_thr=0.1,
    rgba=(0.2, 0.8, 0.2, 0.6),
    base_vertex_offset=0,
):
    """
    Convert sparse footprint pixels into a mesh made of small square tiles.

    x, y, z:
        1D arrays of equal length.
        x/y are pixel coordinates.
        z is footprint intensity.

    Returns:
        vertices: (4*N, 3)
        faces:    (2*N, 3)
        colors:   (4*N, 4)
    """
    y, x = np.unravel_index(footprint.indices, dim)
    z = np.asarray(footprint.data / footprint.data.max(), dtype=np.float32)

    x = x[z > z_thr]
    y = y[z > z_thr]
    z = z[z > z_thr]
    peak_points = z > 0.5

    # Crop to local bounding box
    xmin, xmax = x.min(), x.max()
    ymin, ymax = y.min(), y.max()

    W = xmax - xmin + 1
    H = ymax - ymin + 1

    lx = x - xmin
    ly = y - ymin

    # Local vertex index grid: -1 means no vertex
    idx = np.full((H, W), -1, dtype=np.int32)
    idx[ly, lx] = np.arange(len(z), dtype=np.int32)

    z = z_offset + 0.7 * z_scale * z

    vertices = np.column_stack([x, y, z]).astype(np.float32)
    pick_points = vertices[peak_points]  # for now, use the same points for picking

    # Find all complete 2x2 blocks in one vectorized operation
    i00 = idx[:-1, :-1]
    i10 = idx[:-1, 1:]
    i01 = idx[1:, :-1]
    i11 = idx[1:, 1:]

    valid = (i00 >= 0) & (i10 >= 0) & (i01 >= 0) & (i11 >= 0)

    if np.any(valid):
        a = i00[valid].astype(np.uint32) + base_vertex_offset
        b = i10[valid].astype(np.uint32) + base_vertex_offset
        c = i01[valid].astype(np.uint32) + base_vertex_offset
        d = i11[valid].astype(np.uint32) + base_vertex_offset

        faces = np.empty((2 * len(a), 3), dtype=np.uint32)
        faces[0::2] = np.column_stack([a, b, c])
        faces[1::2] = np.column_stack([b, d, c])
    else:
        faces = np.zeros((0, 3), dtype=np.uint32)

    colors = np.tile(
        np.asarray(rgba, dtype=np.float32),
        (vertices.shape[0], 1),
    )

    return vertices, faces, colors, pick_points
