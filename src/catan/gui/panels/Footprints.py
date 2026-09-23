import numpy as np
from dataclasses import dataclass
from typing import Optional, Tuple, Dict, Literal

from vispy import scene, color
from vispy.scene import visuals, cameras
from vispy.color import Colormap
from vispy.scene.visuals import Mesh

from PySide6.QtCore import Qt, Signal

from PySide6.QtGui import (
    QCursor,
    QAction,
)

from PySide6.QtWidgets import (
    QDoubleSpinBox,
    QCheckBox,
    QLabel,
    QVBoxLayout,
    QHBoxLayout,
    QWidget,
    QMenu,
    QPushButton,
    QFrame,
    QSlider,
)

import importlib

from catan.core.structures import NeuronComponent, sessiondata_type
from catan.gui.panels import BasePlot
from catan.gui.panels.helper import ReviewStatusFilter, ControlPanel

from catan.gui.interaction import click_events

from catan.gui.structures import request_handler
from catan.gui.GUI_elements.utils.menu_creation import (
    get_colored_label,
    add_label_to_menu,
)
from catan.gui.GUI_elements.fragments.ResetViewButton import ResetViewButton
from catan.tracking.structures import ReviewStatus
from catan.core.image_correlation import calculate_img_correlation

# importlib.reload(FootprintSlider)
importlib.reload(request_handler)

CAMERA_PADDING_PX = 25.0


@dataclass
class FootprintRecord:
    key: Tuple[int, int]  # e.g. (session_id, footprint_id)
    neuron: int
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

    main_visual = Mesh
    main_visual_name = "mesh"

    overlays = ["focused", "highlighted", "hovered"]

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

        self.reset_view_button = ResetViewButton(
            self.native, (6, 50), lambda: self.reset_camera(full=True)
        )

        self._build_review_status_tag()

        self.parameter_overlay = None
        self.request_display = RequestDisplay(self.native)
        self.state.current_request = None

        self.events.resize.connect(self._on_canvas_resize)

        self.clear()
        self.camera_set = False

        self.register_actions: dict[str, QAction] = {}

        # self._setup_review_shortcuts()

        self.freeze()

    def _on_canvas_resize(
        self,
        event=None,
    ):
        self._position_overlay_controls()
        self._position_request_display()
        self._position_review_status_tag()

    def _build_review_status_tag(self):
        self.review_status_label = QLabel(self.native)
        self.review_status_label.setObjectName("reviewStatusTag")
        self.review_status_label.setAlignment(Qt.AlignmentFlag.AlignCenter)

        self.review_status_label.setStyleSheet("""
            QLabel#reviewStatusTag {
                color: white;
                background-color: #666666;
                border: 1px solid #808080;
                border-radius: 6px;
                padding: 4px 10px;
                font-weight: 600;
            }
        """)

        self.review_status_label.hide()

    def _position_review_status_tag(self):

        if not self.review_status_label.isVisible():
            return

        margin = 8

        self.review_status_label.move(margin, margin)

        self.review_status_label.raise_()

    def update_review_status_tag(self):

        component = self.state.focused_component

        if component is None or self.data is None or self.data.assignments is None:
            self.review_status_label.hide()
            return

        neuron_id = int(component.neuron_id)

        if neuron_id >= len(self.data.assignments.review_status):
            self.review_status_label.hide()
            return

        status = ReviewStatus(self.data.assignments.review_status[neuron_id])

        if status == ReviewStatus.REVIEWED:
            background = "#357a4f"
            border = "#58a873"

        elif status == ReviewStatus.UNCERTAIN:
            background = "#9a672c"
            border = "#c98a3e"

        else:  # PENDING
            background = "#555b63"
            border = "#747b85"

        self.review_status_label.setText(status.label)

        self.review_status_label.setStyleSheet(f"""
            QLabel#reviewStatusTag {{
                color: #ffffff;
                background-color: {background};
                border: 1px solid {border};
                border-radius: 6px;
                padding: 4px 10px;
                font-weight: 600;
            }}
            """)

        self.review_status_label.adjustSize()
        self.review_status_label.show()
        self.review_status_label.raise_()

        self._position_review_status_tag()

    def _position_overlay_controls(self):

        if self.parameter_overlay is None:
            return

        margin = 8
        spacing = 6

        palette = self.parameter_overlay

        palette.adjustSize()

        x = self.native.width() - palette.width() - margin

        palette.move(max(margin, x), margin)

        palette.raise_()

    def _position_request_display(self):

        if self.request_display is None:
            return

        margin = 8

        display = self.request_display
        display.adjustSize()

        x = margin
        y = self.native.height() - display.height() - margin

        display.move(x, max(margin, y))

        display.raise_()

    def attach_parameter_overlay(self):
        """
        Attach the Footprints parameter widgets to the canvas after
        Controller.build_controls() has created them.
        """

        parameter_overlay = self.controls.get("parameter")

        if parameter_overlay is None:
            return

        self.parameter_overlay = parameter_overlay

        self.parameter_overlay.setParent(self.native)
        self.parameter_overlay.show()
        self.parameter_overlay.raise_()

        self.parameter_overlay.overlay_layout_changed.connect(
            self._position_overlay_controls
        )

        # if hasattr(self.parameter_overlay, "session_filter_overlay"):
        #     self.session_filter_overlay = self.parameter_overlay.session_filter_overlay

        #     self.session_filter_overlay.setParent(self.native)

        #     # Respect current filter state.
        #     self.session_filter_overlay.setVisible(
        #         self.parameter_overlay.checkbox_session_only.isChecked()
        #     )

        #     self.session_filter_overlay.raise_()

        self._position_overlay_controls()

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

        for style, visuals in self.plotting["overlays"].items():
            if visuals is None:
                continue
            for visual in visuals:
                if visual is None:
                    continue
                visual.visible = False

    def plot_neurons(self, reset=False):

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

        self.state.timeit("Initial setup")

        ## identify the neurons thata should be plotted
        this_neuron = self.state.focused_component.neuron_id

        display_scope = self.controls["parameter"].display_scope
        if display_scope == "adjacent":

            ## find closeby neurons
            union_centroids = self.data.assignments.union.centroids
            distances = np.linalg.norm(
                union_centroids - union_centroids[this_neuron], axis=1
            )
            to_plot_neurons = np.where(distances <= self.state.adjacency_radius)[0]

        elif display_scope == "selection":
            to_plot_neurons = [c.neuron_id for c in self.state.selected_components]
        else:
            raise ValueError(f"Unknown footprint display scope: " f"{display_scope!r}")

        show_excluded = self.controls["parameter"].checkbox_show_excluded.isChecked()

        if not show_excluded:
            included = self.data.assignments.union.included
            to_plot_neurons = [neuron for neuron in to_plot_neurons if included[neuron]]

        to_plot_neurons = np.asarray(to_plot_neurons, dtype=int)

        review_mask = ReviewStatusFilter.neuron_mask(
            to_plot_neurons,
            self.data.assignments.review_status,
            (self.controls["parameter"].review_filter.visible_statuses),
            focused_neuron_id=this_neuron,
            keep_focused=True,
        )

        to_plot_neurons = to_plot_neurons[review_mask]

        session_filter = self.controls["parameter"].checkbox_session_only.isChecked()
        filter_focused = self.controls["parameter"].checkbox_filter_focused.isChecked()

        z_stretch = self.controls["parameter"].z_stretch_spin.value()

        self.state.timeit("Found neurons to plot and calculated centroid ranges")

        ## iterate through each neuron that should be plotted
        for neuron in to_plot_neurons:
            included = bool(self.data.assignments.union.included[neuron])

            selected_neurons = {
                component.neuron_id
                for component in (self.state.selected_components or [])
            }

            if not included:
                key = "default"
                alpha = 0.1

            elif neuron == this_neuron:
                key = "focused"
                alpha = self.styles.opts[key]["alpha"]

            elif neuron in selected_neurons:
                key = "selected"
                alpha = self.styles.opts[key]["alpha"]

            else:
                key = "default"
                alpha = self.styles.opts[key]["alpha"]

            ## create new data if not yet present
            if neuron not in self.plotting["data"]:
                self.add_footprint_data(
                    neuron,
                    alpha=alpha,
                    z_stretch=z_stretch,
                )

            if session_filter and (neuron != this_neuron or filter_focused):
                ## select subset of data for session-only display ...

                session_id = int(self.controls["session_filter"].slider.value())
                component = NeuronComponent(neuron, session_id)

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
            mesh = Mesh(
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
        union_centroids = self.data.assignments.union.centroids[displayed_neurons, ...]

        centroid_min = np.nanmin(union_centroids, axis=0)
        centroid_max = np.nanmax(union_centroids, axis=0)

        ## calculate ranges
        x_range = np.clip(
            np.array([centroid_min[0], centroid_max[0]])
            + np.array([-CAMERA_PADDING_PX, CAMERA_PADDING_PX]),
            0,
            self.data.current_session.dims[1],
        )
        y_range = np.clip(
            np.array([centroid_min[1], centroid_max[1]])
            + np.array([-CAMERA_PADDING_PX, CAMERA_PADDING_PX]),
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
        # print(f"Adding footprint data for neuron {neuron}")

        vertex_offset = 0
        face_offset = 0

        # or self.plotting["data"][neuron].mesh_vertices is None
        vertices_all = []
        faces_all = []
        colors_all = []
        for session in self.data.sessions:

            if not self.data.session_assigned(session.id):
                continue

            component = NeuronComponent(neuron, session.id)

            fp_id = self.state.get_footprint_from_component(component)

            if fp_id is None or fp_id < 0:
                continue

            rgba = np.array(
                self.state.session_colors[session.id],
                dtype=np.float32,
                copy=True,
            )

            rgba[3] = 1.0

            z_offset = float(session.id) * z_stretch

            v, f, c, pick_points = footprint_to_mesh(
                session.footprints[:, fp_id],
                session.dims,
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
                neuron=neuron,
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

    def find_closest_component(self, mouse_pos) -> Optional[NeuronComponent]:

        if not self.plotting["visuals"]:
            return None

        center_radius_px = 80
        point_radius_px = 10

        keys = [
            key
            for key in self.plotting["data"].keys()
            if isinstance(key, tuple) and key[0] in self.plotting["visuals"]
        ]
        centers = np.asarray(
            [self.plotting["data"][key].center_xyz for key in keys],
            dtype=np.float32,
        )

        screen_centers = click_events.visual_to_canvas(
            self.plotting["visuals"][keys[0][0]], centers
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
                self.plotting["visuals"][keys[int(i)][0]], rec.pick_points
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
        union_centroids = self.data.assignments.union.centroids
        neuron_id = component.neuron_id
        d = np.linalg.norm(union_centroids - union_centroids[neuron_id], axis=1)

        neuron_ids = np.where(d <= self.state.adjacency_radius)[0]
        neuron_ids = [n for n in neuron_ids if n != neuron_id]

        self.state.logger.debug(
            f"Neuron {neuron_id} has {len(neuron_ids)} adjacent neurons within radius {self.state.adjacency_radius}: {neuron_ids}"
        )

        return np.array(neuron_ids), np.array(d[neuron_ids])

    def plot_data_from_rec(self, rec, style: str) -> dict[str, np.ndarray]:

        neuron_rec = self.plotting["data"][rec.neuron]
        verts = neuron_rec.mesh_vertices[rec.vertex_start : rec.vertex_stop]
        faces = neuron_rec.mesh_faces[rec.face_start : rec.face_stop] - rec.vertex_start
        cols = self.styles.get_color_array(style, verts[:, 2], alpha=0.6)
        return {"vertices": verts, "faces": faces, "vertex_colors": cols}

    def on_mouse_release(self, event):

        if self.state.current_request is not None:

            if event.button == 1 and event.pos is not None:
                self._on_request_interaction(event)

            event.handled = True
            return

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
        ref_component = NeuronComponent(ref_neuron, ref_session_id)
        ref_fp_id = self.state.get_footprint_from_component(ref_component)

        ## calculate the statistics
        if self.data.model is not None and self.data.model.fitted:
            similarity, _, shift = calculate_img_correlation(
                A1=self.data.sessions[this_component.session_id].footprints[
                    :, this_fp_id
                ],
                A2=self.data.sessions[ref_component.session_id].footprints[
                    :, ref_fp_id
                ],
                dims=self.data.sessions[this_component.session_id].dims,
                crop=True,
                binary=False,
                shift=True,
                mode="cosine_union",
                gamma=0.1,
                shift_optimized=True,
            )
            d_shift = np.sqrt(np.sum([s**2 for s in shift]))
            p_same = self.data.model.f_same(d_shift, similarity)[0]
        else:
            similarity, d_shift, p_same = None, None, None
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
        if session_id is None:
            # similarity is None or shift is None or p_same is None or
            add_label_to_menu(
                menu,
                f"no {which} session found",
                enabled=False,
            )
            return
        ds = abs(this_component.session_id - session_id)

        arrow = "\u25bc" if which == "previous" else "\u25b2"
        html_session = (
            f"\u0394s="
            + get_colored_label(f"{ds}", ds / 10.0, cmap)
            + f"({session_id})"
        )
        html_probability = ""
        html_similarity = ""
        html_shift = ""
        if p_same is not None:
            html_probability = get_colored_label(
                f"p={p_same:.2f}", 1 - p_same, cmap, ["b"]
            )

        if similarity is not None:
            html_similarity = "c=" + get_colored_label(
                f"{similarity:.2f}", (1 - similarity) / 2.0, cmap
            )

        if shift is not None:
            html_shift = "shift=" + get_colored_label(
                f"{shift:.2f}px", shift / 10.0, cmap
            )

        add_label_to_menu(
            menu,
            f"{arrow} {html_probability} - {html_session},\t{html_similarity},\t{html_shift}",
            enabled=False,
        )

    def open_footprint_context_menu(self, this_component: NeuronComponent):

        if this_component.session_id is None:
            return

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

        ## === flag submenus ===
        submenu_neuron = self.build_neuron_tag_menu(this_component.neuron_id, menu)
        menu.addMenu(submenu_neuron)

        submenu_footprint = self.build_footprint_tag_menu(this_component, menu)
        menu.addMenu(submenu_footprint)

        menu.addSeparator()

        if not self.data.sessions[this_component.session_id].status["traces_loaded"]:

            load_action = QAction("Load trace for this session", menu)
            menu.addAction(load_action)

            load_action.triggered.connect(
                lambda: self.toggle_session_data(this_component.session_id, "traces"),
            )

        menu.exec(QCursor.pos())

    def build_neuron_tag_menu(self, neuron_id: int, menu: QMenu) -> QMenu:
        submenu = QMenu("Tag neuron ...", menu)

        for status in ReviewStatus:
            key = status.name.lower()

            self.register_actions[key] = QAction(f"... as {status.menu_label}", submenu)
            submenu.addAction(self.register_actions[key])

            self.register_actions[key].triggered.connect(
                lambda checked, status=status: self.change_review_status(
                    neuron_id, status
                )
            )

        submenu.addSeparator()

        self.register_actions["toggle_included_neuron"] = QAction(
            f"... as {"excluded" if self.data.is_included(neuron_id) else "included"}",
            submenu,
        )
        submenu.addAction(self.register_actions["toggle_included_neuron"])
        self.register_actions["toggle_included_neuron"].triggered.connect(
            lambda: self.toggle_included(neuron_id)
        )

        self.register_actions["remove_neuron"] = QAction("... as removed", submenu)
        submenu.addAction(self.register_actions["remove_neuron"])
        self.register_actions["remove_neuron"].triggered.connect(
            lambda: self.remove_component(neuron_id)
        )

        menu.addMenu(submenu)

        return submenu

    def build_footprint_tag_menu(
        self, component: NeuronComponent, menu: QMenu
    ) -> QMenu:

        submenu = QMenu("Tag footprint ...", menu)

        self.register_actions["merge"] = QAction("... for merge", submenu)
        submenu.addAction(self.register_actions["merge"])
        self.register_actions["merge"].triggered.connect(
            lambda: self.start_merge(component)
        )

        self.register_actions["split"] = QAction("... for split", submenu)
        submenu.addAction(self.register_actions["split"])
        self.register_actions["split"].triggered.connect(
            lambda: self.start_split(component)
        )

        fp_id = self.state.get_footprint_from_component(component)
        self.register_actions["toggle_included"] = QAction(
            f"... as {"excluded" if self.data.is_included(component) else "included"}",
            submenu,
        )
        submenu.addAction(self.register_actions["toggle_included"])
        self.register_actions["toggle_included"].triggered.connect(
            lambda: self.toggle_included(component)
        )

        self.register_actions["remove"] = QAction("... as removed", submenu)
        submenu.addAction(self.register_actions["remove"])
        self.register_actions["remove"].triggered.connect(
            lambda: self.remove_component(component)
        )

        return submenu

    def _on_request_status_changed(self):

        self.request_display.set_request(self.state.current_request)
        self._position_request_display()

    def sync_request_display(self):

        request = self.state.current_request

        self.request_display.set_request(request)

        if request is not None:
            self.state.update_highlighted_components(request.components)

        self._position_request_display()

    def start_merge(self, component: NeuronComponent):
        self.state.current_request = request_handler.RequestHandler("merge", component)
        self.state.update_highlighted_components(self.state.current_request.components)

    def start_split(self, component: NeuronComponent):
        self.state.current_request = request_handler.RequestHandler("split", component)
        self.state.update_highlighted_components(self.state.current_request.components)

    def cancel_request(self):
        self.state.current_request = None
        self.state.update_highlighted_components(None)

    def advance_or_confirm_request(self):

        request = self.state.current_request

        if request is None:
            return

        # ---------------------------------------------
        # Origin -> destination
        # ---------------------------------------------
        if request.stage == "origin":

            try:
                request.advance_stage()

            except ValueError as exc:
                self.request_display.set_error(str(exc))
                return

            self.state.notify_request_changed()
            return

        # ---------------------------------------------
        # Final confirmation
        # ---------------------------------------------
        if not request.is_complete:
            self.request_display.set_error("The request is not complete yet.")
            return

        try:
            self.data.process_component_request(request)
        except Exception as exc:
            self.request_display.set_error(str(exc))
            return

        self.state.current_request = None
        self.state.update_highlighted_components(None)

    def _on_request_interaction(self, event):

        request = self.state.current_request

        if request is None:
            return

        component = self.find_closest_component(event.pos)

        if component is None:
            return

        current_components = (
            request.origin if request.stage == "origin" else request.destination
        )

        if component in current_components:

            request.remove_component(component)

            self.state.update_highlighted_components(request.components)
            self.state.notify_request_changed()
            return

        error = request.validation_error(component)

        if error is not None:
            self.request_display.set_error(error)
            return

        request.add_component(component)

        self.state.update_highlighted_components(request.components)
        self.state.notify_request_changed()

    def toggle_included(self, component: NeuronComponent | int):
        # Implement the logic for handling the removal action
        if self.data.is_included(component):
            self.data.exclude_component(component)
        else:
            self.data.reinclude_component(component)

    def remove_component(self, component: NeuronComponent | int):
        self.data.remove_component(component)

    def toggle_session_data(
        self, session_id: int, which: Optional[sessiondata_type] = None
    ):
        session = self.data.sessions[session_id]
        self.state.tasks.start(
            "loading",
            f"Loading {which} data for {session.name}",
            lambda: self.data.toggle_session_data(session_id, which),
            # finished=self.refresh_rows,
        )

    def change_neuron_assignment_dialog(
        self, component: NeuronComponent, new_neuron: Optional[int] = None
    ):

        session_id = component.session_id
        fp_id = self.state.get_footprint_from_component(component)

        if new_neuron is None:
            ## append new neuron
            new_neuron = self.state.assignments.shape[0]
            self.state.logger.debug(
                f"Creating new neuron {new_neuron} for footprint {fp_id} in session {session_id}"
            )
            self.data.assignments.pad_empty(n_neurons=1, n_sessions=0)

        ## change assignments array
        self.state.assignments[new_neuron, session_id] = fp_id
        self.state.assignments[component.neuron_id, session_id] = -1

        self.data.rebuild_union_neurons([new_neuron, component.neuron_id])

        if np.all(self.state.assignments[component.neuron_id, :] < 0):
            print(f"Neuron {component.neuron_id} is now empty and will be removed.")
            print(
                f"CAREFUL!! all references are f**cked up now, need to update all neurons"
            )

        self.plot_neurons(reset=True)

    def set_focused_footprint(self, component: NeuronComponent):
        self.state.focused_component = component

    def change_review_status(self, neuron_id: int, status: ReviewStatus):
        self.data.change_review_status(neuron_id, status)


class Controller(BasePlot.CanvasController):

    request_status_changed = Signal()

    def disconnect_signals(self):
        super().disconnect_signals()
        self.controls["parameter"].data_parameter_changed.disconnect()
        self.controls["parameter"].display_parameter_changed.disconnect()
        self.state.request_status_changed.disconnect(self._on_request_status_changed)

    def build_controls(self):
        super().build_controls()

        self.state.adjacency_radius_changed.connect(self.replot_neurons)
        self.controls["parameter"] = FootprintParametersController(self.section)
        self.canvas.attach_parameter_overlay()

        self.controls["parameter"].data_parameter_changed.connect(
            lambda: self.replot_neurons()
        )
        self.controls["parameter"].display_parameter_changed.connect(
            lambda: self.update_neuron_selection()
        )

        self.controls["session_filter"] = SessionFilterControl(self.section)

        self.controls["session_filter"].hide()

        self.section.y_options_layout.addWidget(
            self.controls["session_filter"],
            stretch=1,
            alignment=Qt.AlignmentFlag.AlignHCenter,
        )

        self.controls["parameter"].session_only_changed.connect(
            self._on_session_filter_toggled
        )

        self.controls["session_filter"].valueChanged.connect(
            lambda _: self.update_neuron_selection()
        )

        self.state.request_status_changed.connect(self._on_request_status_changed)

        self.canvas.request_display.cancel_requested.connect(self.canvas.cancel_request)

        self.canvas.request_display.confirm_requested.connect(
            self.canvas.advance_or_confirm_request
        )

        # Important for hot reload:
        # there may already be a pending request.
        self._on_request_status_changed()

    def _on_data_changed(self, input: Tuple[str, int]):

        if input[0] in ["sessions", "assignments"]:
            self.replot_neurons()

        elif input[0] == "review_status":
            self.canvas.update_review_status_tag()

    def initialize_display(self):
        self._setup_session_filter()
        super().initialize_display()
        self.canvas.update_review_status_tag()

    def _on_session_changed(self):
        self._setup_session_filter()

        super()._on_session_changed()

    def _on_session_only_changed(self):
        # self.controls["parameter"]._on_session_only_changed()
        self.update_neuron_selection()

    def _on_session_filter_toggled(self, active: bool):

        self.controls["session_filter"].setVisible(active)
        self.controls["session_filter"]._on_value_changed(self.state.current_session_id)

        self.update_neuron_selection()

    def _setup_session_filter(self):

        if (
            not hasattr(self.state, "assignments")
            or self.state.current_session_id is None
        ):
            return

        self.controls["session_filter"].set_sessions(
            n_sessions=self.state.assignments.shape[1],
            current_session=self.state.current_session_id,
        )

    def _on_selection_changed(self):
        # self.controls["slider"].update_setup()
        super()._on_selection_changed()

    def _on_focus_changed(self):
        # self.controls["slider"].adjust_id()
        self.canvas.update_review_status_tag()
        super()._on_focus_changed()

    def update_neuron_selection(self):
        self.canvas.plot_neurons()
        self.update_styles()

    def replot_neurons(self):
        self.canvas.plot_neurons(reset=True)
        self.update_styles()

    def _on_request_status_changed(self):
        self.canvas.sync_request_display()


class SessionFilterControl(QWidget):

    valueChanged = Signal(int)

    def __init__(self, parent=None):
        super().__init__(parent)

        self.data = parent.data

        self.setFixedWidth(50)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(2, 4, 2, 4)
        layout.setSpacing(4)

        self.title = QLabel("Session")
        self.title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(self.title)

        self.value_label = QLabel("")
        self.value_label.setAlignment(Qt.AlignmentFlag.AlignCenter)

        self.slider = QSlider(Qt.Orientation.Vertical)

        self.slider.setRange(0, 0)
        self.slider.setSingleStep(1)
        self.slider.setPageStep(1)

        # Important:
        # session 0 at the BOTTOM,
        # increasing session IDs upward.
        self.slider.setInvertedAppearance(False)
        self.slider.setInvertedControls(False)

        self.zero_label = QLabel("0")
        self.zero_label.setAlignment(Qt.AlignmentFlag.AlignCenter)

        layout.addWidget(self.value_label)
        layout.addWidget(
            self.slider,
            stretch=1,
        )
        layout.addWidget(self.zero_label)

        self.slider.valueChanged.connect(self._on_value_changed)

    def _on_value_changed(self, value: int):
        self.value_label.setText(f"{value:d} / {len(self.data.sessions)}")
        self.value_label.setToolTip(f"{self.data.sessions[value].name}")
        self.valueChanged.emit(value)

    def set_sessions(
        self,
        n_sessions: int,
        current_session: int | None = None,
    ):
        if n_sessions <= 0:
            self.slider.setRange(0, 0)
            return

        self.slider.setRange(0, n_sessions - 1)

        if current_session is not None:
            self.slider.setValue(int(current_session))


class RequestDisplay(QFrame):

    cancel_requested = Signal()
    confirm_requested = Signal()

    def __init__(self, parent):
        super().__init__(parent)

        self.setMaximumWidth(parent.width() / 2)

        root_layout = QVBoxLayout(self)
        root_layout.setContentsMargins(10, 8, 10, 8)
        root_layout.setSpacing(5)

        self.title = QLabel("")
        self.title.setObjectName("requestTitle")
        self.title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        root_layout.addWidget(self.title)

        self.origin_label = QLabel("")
        self.origin_label.setObjectName("requestStatus")
        self.origin_label.setAlignment(Qt.AlignmentFlag.AlignLeft)
        root_layout.addWidget(self.origin_label)

        self.destination_label = QLabel("")
        self.destination_label.setObjectName("requestStatus")
        self.destination_label.setAlignment(Qt.AlignmentFlag.AlignLeft)
        root_layout.addWidget(self.destination_label)

        self.error_label = QLabel("")
        self.error_label.setObjectName("requestError")
        self.error_label.setWordWrap(True)
        self.error_label.hide()
        root_layout.addWidget(self.error_label)

        button_layout = QHBoxLayout()

        self.cancel_button = QPushButton("Cancel")
        self.cancel_button.setObjectName("requestCancelButton")

        self.confirm_button = QPushButton("Confirm")
        self.confirm_button.setObjectName("requestConfirmButton")

        button_layout.addStretch()
        button_layout.addWidget(self.cancel_button)
        button_layout.addWidget(self.confirm_button)

        root_layout.addLayout(button_layout)

        self.cancel_button.clicked.connect(self.cancel_requested.emit)

        self.confirm_button.clicked.connect(self.confirm_requested.emit)

        # self.root_layout = root_layout
        self.setObjectName("RequestDisplayOverlay")

        self.setStyleSheet("""
            QFrame#RequestDisplayOverlay {
                background-color: rgba(30, 34, 40, 245);
                border: 1px solid #6b7480;
                border-radius: 7px;
            }

            QFrame#RequestDisplayOverlay QLabel {
                background: transparent;
                border: none;
                color: #e8eaed;
            }

            QLabel#requestTitle {
                color: #ffffff;
                font-size: 13px;
                font-weight: 600;
                padding-bottom: 3px;
            }

            QLabel#requestStatus {
                color: #d5d9df;
                font-size: 11px;
                padding: 1px 2px;
            }

            QLabel#requestError {
                color: #ff8a8a;
                font-size: 11px;
                font-weight: 500;
                padding: 3px 2px;
            }

            QFrame#RequestDisplayOverlay QPushButton {
                color: #e8eaed;
                background-color: #3b414a;
                border: 1px solid #66707c;
                border-radius: 4px;
                padding: 5px 12px;
                min-width: 32px;
            }

            QFrame#RequestDisplayOverlay QPushButton:hover {
                background-color: #4a525d;
                border-color: #8793a1;
            }

            QPushButton#requestCancelButton:pressed {
                background-color: #30353c;
            }

            QPushButton#requestConfirmButton {
                background-color: #355f4b;
                border-color: #57906f;
                color: #ffffff;
                font-weight: 600;
            }

            QPushButton#requestConfirmButton:hover {
                background-color: #40735a;
            }

            QPushButton#requestConfirmButton:disabled {
                background-color: #2b3036;
                border-color: #474e57;
                color: #777f89;
                font-weight: normal;
            }
        """)

        self.setVisible(False)

    def set_request(
        self,
        request_handler: request_handler.RequestHandler | None,
    ):

        if request_handler is None:
            self.clear_request()
            return

        self.setVisible(True)

        self.title.setText(f"{request_handler.type.capitalize()} request")
        self.origin_label.setText(request_handler.origin_status_text())
        self.destination_label.setText(request_handler.destination_status_text())

        if request_handler.stage == "origin":
            self.confirm_button.setText("Continue to destination")

        else:
            self.confirm_button.setText(f"Confirm {request_handler.type}")

        self.confirm_button.setEnabled(request_handler.stage_complete)

        self.clear_error()

    def set_error(self, message: str):
        self.error_label.setText(message)
        self.error_label.show()
        self.adjustSize()

    def clear_error(self):
        self.error_label.clear()
        self.error_label.hide()

    def clear_request(self):

        self.setVisible(False)

        self.title.clear()
        self.origin_label.clear()
        self.destination_label.clear()

        self.confirm_button.setEnabled(False)

        self.clear_error()


class FootprintParametersController(ControlPanel.ControlPanel):

    data_parameter_changed = Signal()

    session_only_changed = Signal(bool)
    reset_camera_requested = Signal()

    def __init__(self, parent):
        super().__init__(parent)

        scope_selector = self._build_display_scope_selection(
            {
                "adjacent": {
                    "label": "Nearby",
                    "position": "left",
                },
                "selection": {
                    "label": "Selected",
                    "position": "right",
                },
            }
        )
        self.form.addRow("Show", scope_selector)

        review_selector = self._build_review_selector()
        self.form.addRow("Review status", review_selector)

        ## add footprint threshold control
        initial_threshold = 0.1
        self.threshold_spin = QDoubleSpinBox()
        self.threshold_spin.setDecimals(3)
        self.threshold_spin.setRange(0.0, 1.0)
        self.threshold_spin.setSingleStep(0.01)
        self.threshold_spin.setValue(initial_threshold)
        self.threshold_spin.valueChanged.connect(
            lambda: self.data_parameter_changed.emit()
        )
        self.form.addRow("Threshold", self.threshold_spin)

        ## z_stretch control (for 3D visualization)
        initial_z_stretch = 5.0
        self.z_stretch_spin = QDoubleSpinBox()
        self.z_stretch_spin.setDecimals(1)
        self.z_stretch_spin.setRange(1.0, 10.0)
        self.z_stretch_spin.setSingleStep(0.5)
        self.z_stretch_spin.setValue(initial_z_stretch)
        self.z_stretch_spin.valueChanged.connect(
            lambda: self.data_parameter_changed.emit()
        )
        self.form.addRow("Z stretch", self.z_stretch_spin)

        for spin in (
            self.threshold_spin,
            self.z_stretch_spin,
        ):
            spin.setFixedWidth(72)

        ## checkbox for toggling single session display
        self.checkbox_session_only = QCheckBox("Session filter")
        self.checkbox_session_only.setChecked(False)
        self.checkbox_session_only.toggled.connect(self._on_session_only_changed)
        self.form.addRow(self.checkbox_session_only)

        self.checkbox_filter_focused = QCheckBox("Filter focused neuron")
        self.checkbox_filter_focused.setChecked(False)
        self.checkbox_filter_focused.setEnabled(False)
        self.checkbox_filter_focused.toggled.connect(
            lambda: self.display_parameter_changed.emit()
        )
        self.form.addRow(self.checkbox_filter_focused)

        self.checkbox_show_excluded = QCheckBox("Show excluded")
        self.checkbox_show_excluded.setChecked(True)
        self.checkbox_show_excluded.toggled.connect(
            lambda: self.display_parameter_changed.emit()
        )
        self.form.addRow(self.checkbox_show_excluded)

    def _on_session_only_changed(self):

        active = self.checkbox_session_only.isChecked()

        self.checkbox_filter_focused.setEnabled(active)
        self.session_only_changed.emit(active)

        super()._on_session_only_changed()


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
