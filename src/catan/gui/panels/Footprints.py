import time

import numpy as np
from dataclasses import dataclass, field
from typing import Optional, Tuple, Dict, Literal

from vispy import scene, color
from vispy.scene import visuals, cameras
from vispy.color import Colormap
from vispy.scene.visuals import Mesh

from PySide6.QtCore import Qt, Signal, QTimer, QEvent, QSignalBlocker

from PySide6.QtGui import (
    QCursor,
    QColor,
    QIcon,
    QPixmap,
    QKeySequence,
    QShortcut,
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
    QComboBox,
    QStyle,
    QToolTip,
    QApplication,
    QToolButton,
    QMessageBox,
    QScrollArea,
    QButtonGroup,
    QListWidget,
    QListWidgetItem,
)

import importlib

from catan.gui.utils.session_labels import session_label, session_labels, session_tag
from catan.core.changes import (
    ChangeKind as C,
    DataChange,
    SESSION_STRUCTURE_CHANGES,
)
from catan.core.structures import NeuronComponent, sessiondata_type
from catan.core.redetection import infer_aligned_footprint
from catan.core.structures.remap import Remapping
from catan.core.footprint_alignment import move_points as _move_points
from catan.gui.panels import BasePlot
from catan.gui.GUI_elements.utils.tooltips import show_tooltip
from catan.gui.GUI_elements.utils.enter_key import EnterKeyButton
from catan.gui.panels.helper import ReviewStatusFilter, ControlPanel
from catan.gui.interaction import click_events
from catan.gui.data.assignment_colors import assignment_label

from catan.gui.structures import request_handler
from catan.gui.GUI_elements.utils.menu_creation import (
    add_menu_section,
    get_colored_label,
    fit_message_box_buttons,
)
from catan.gui.GUI_elements.fragments.ResetViewButton import ResetViewButton
from catan.gui.panels.helper.footprint_coordinates import FootprintCoordinates
from catan.tracking.structures import ReviewStatus
from catan.core.image_correlation import calculate_img_correlation

# importlib.reload(FootprintSlider)
importlib.reload(request_handler)

CAMERA_PADDING_PX = 25.0
# Coordinate floor below the lowest session, in session spacings.
FLOOR_DEPTH = 0.8
ROTATE_ON_FOCUS_KEY = "footprints/rotate_on_focus"
ROTATION_DURATION_KEY = "footprints/rotation_duration_s"
SHOW_COORDINATES_KEY = "footprints/show_coordinates"
HOVER_NEURON_KEY = "footprints/hover_neuron"
# The hovered footprint's other footprints: faint hover colour, fading out.
HOVER_NEURON_ALPHA = 0.35
HOVER_NEURON_FADE_S = 2.0
# Picking a target neuron: the neurons that are no candidate fade to this.
PICK_FADED_ALPHA = 0.08

INFERRED_RGBA = np.array(
    [0.0, 166 / 255, 178 / 255, 1.0],
    dtype=np.float32,
)

# Good → intermediate → bad; bright throughout for dark menus.
MATCH_QUALITY_CMAP = Colormap(
    [
        "#72E6A1",  # mint green
        "#FFE08A",  # pale yellow
        "#FF8585",  # coral red
    ]
)


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
    inferred: bool = False


@dataclass
class NeuronRecord:
    mesh_vertices: np.ndarray  # (N, 3)
    mesh_faces: np.ndarray  # (M, 3)
    mesh_colors_default: np.ndarray  # (N, 4)

    thr: float  # threshold used for this footprint (store to check if changed)


@dataclass
class TargetPick:
    """Picking the target neuron of a reassignment / move (Display.
    start_reassign_pick / start_move_pick); shown by TargetPickerBar."""

    kind: str  # "reassign" or "move"
    component: NeuronComponent  # the right-clicked footprint
    source_id: int
    moved: list  # sessions of the source's footprints that move
    radius: float  # candidates within (px), widened by "Load more"
    rows: dict = field(default_factory=dict)  # neuron -> (label, html lines)
    candidates: set = field(default_factory=set)
    sections: list = field(default_factory=list)  # (title, [neuron])
    title: str = ""
    summary: list = field(default_factory=list)  # html lines
    note: str = ""
    more_text: str = ""
    allow_new: bool = True
    new_tooltip: str = ""
    moved_union: object = None
    previous_highlight: tuple = None


class Display(BasePlot.BaseCanvas):

    main_visual = Mesh
    main_visual_name = "mesh"

    overlays = [
        "selected",
        "focused",
        "highlighted",
        "highlighted_source",
        "highlighted_target",
        "hovered_neuron",
        "hovered",
    ]

    def __init__(self, display_section, controls, config=None):
        super().__init__(display_section, controls, config)

        ## should be cleaned up / tested if necessary here (grid needed if no axis, ...?)
        self.unfreeze()

        # None means follow the globally active assignments.
        self._assignment_name = (config or {}).get("assignments_name")

        # Fading "hovered_neuron" overlay (show_hovered_neuron).
        self._hover_fade = []
        self._hover_alpha = {}
        self._hover_fade_timer = None
        self._hover_fade_clock = None

        self.grid = self.central_widget.add_grid(spacing=0)
        self.view = self.grid.add_view(row=0, col=0)
        self.view.stretch = (1, 1)  # expand a lot
        self.view.camera = cameras.TurntableCamera()
        self._rotation = None
        self._rotation_timer = None
        self._rotated_neuron = None

        self.changes_on_click = "highlighted"

        self.plot_root = scene.Node(parent=self.view.scene)

        self.reset_view_button = ResetViewButton(
            self.native, (6, 50), lambda: self.reset_camera(full=True)
        )

        self._build_review_status_tag()

        self.parameter_overlay = None
        self.request_display = RequestDisplay(self.native, self.data)
        self.pick_bar = TargetPickerBar(self.native)

        # Why the focused neuron is not drawn (_update_not_drawn_hint).
        self.not_drawn_hint = QLabel(self.native)
        self.not_drawn_hint.setWordWrap(False)
        self.not_drawn_hint.setStyleSheet(
            "QLabel { color: #e8eaed; background-color: rgba(30, 34, 40, 220);"
            " border: 1px solid #c87920; border-radius: 6px; padding: 4px 8px; }"
        )
        self.not_drawn_hint.hide()
        self._pick = None

        self.events.resize.connect(self._on_canvas_resize)

        self._mesh_context = None
        self._inference_errors = set()

        self.coordinates = FootprintCoordinates(
            self, self.view, self.plot_root, lambda: self.data.units
        )
        self._hover_component = None

        self.clear()
        self.camera_set = False


        # self._setup_review_shortcuts()

        self.freeze()

    def _on_canvas_resize(
        self,
        event=None,
    ):
        self._position_overlay_controls()
        self._position_request_display()
        self._position_review_status_tag()
        self.coordinates.place()

    def _build_review_status_tag(self):
        # The focused neuron's tag; its arrow menu sets the review status.
        self.review_status_label = QToolButton(self.native)
        self.review_status_label.setObjectName("reviewStatusTag")
        self.review_status_label.setPopupMode(
            QToolButton.ToolButtonPopupMode.InstantPopup
        )
        self.review_status_label.setCursor(Qt.CursorShape.PointingHandCursor)

        self.review_status_menu = QMenu(self.review_status_label)
        self.review_status_menu.aboutToShow.connect(self._fill_review_status_menu)

        self.review_status_label.hide()

    def _fill_review_status_menu(self):
        menu = self.review_status_menu
        menu.clear()

        component = self.display_focused_component
        if component is None or not self.displays_active_assignments:
            return

        neuron_id = int(component.neuron_id)
        current = ReviewStatus(self.display_assignments.review_status[neuron_id])

        add_menu_section(menu, f"Neuron {neuron_id}", title=True)
        for status in ReviewStatus:
            action = menu.addAction(status.menu_label)
            action.setCheckable(True)
            action.setChecked(status == current)
            action.triggered.connect(
                lambda checked=False, status=status: self.change_review_status(
                    neuron_id, status
                )
            )

    def _position_review_status_tag(self):
        if not self.review_status_label.isVisible():
            return

        margin = 8
        reset_rect = self.reset_view_button.geometry()

        self.review_status_label.move(
            margin,
            reset_rect.bottom() + margin,
        )
        self.review_status_label.raise_()

    def update_review_status_tag(self):

        component = self.display_focused_component

        if component is None or self.data is None or self.display_assignments is None:
            self.review_status_label.hide()
            return

        neuron_id = int(component.neuron_id)

        if neuron_id >= len(self.display_assignments.review_status):
            self.review_status_label.hide()
            return

        status = ReviewStatus(self.display_assignments.review_status[neuron_id])

        if status == ReviewStatus.REVIEWED:
            background = "#357a4f"
            border = "#58a873"

        elif status == ReviewStatus.UNCERTAIN:
            background = "#9a672c"
            border = "#c98a3e"

        else:  # PENDING
            background = "#555b63"
            border = "#747b85"

        # Only the active assignments' status can be changed here.
        editable = self.displays_active_assignments
        self.review_status_label.setMenu(self.review_status_menu if editable else None)
        self.review_status_label.setToolTip(
            "Set the review status" if editable else ""
        )
        self.review_status_label.setText(
            f"Neuron {neuron_id} · {assignment_label(self.assignment_name)} · "
            f"{status.label}" + (" ▾" if editable else "")
        )

        self.review_status_label.setStyleSheet(f"""
            QToolButton#reviewStatusTag {{
                color: #ffffff;
                background-color: {background};
                border: 1px solid {border};
                border-radius: 6px;
                padding: 4px 10px;
                font-weight: 600;
            }}
            QToolButton#reviewStatusTag:hover {{
                border-color: #ffffff;
            }}
            QToolButton#reviewStatusTag::menu-indicator {{
                image: none;
                width: 0px;
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
        if self.request_display is not None:
            self.request_display.reposition()

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

        self._inference_errors.clear()

    def clear_visuals(self):
        for key, footprint_visual in self.plotting["visuals"].items():
            footprint_visual.visible = False
            footprint_visual.parent = None
        self.plotting["visuals"] = {}
        self.clear_highlights()
        # Shown again by reset_camera for the new plot.
        self.coordinates.hide()

    def clear_highlights(self):

        for style, visuals in self.plotting["overlays"].items():
            if visuals is None:
                continue
            for visual in visuals:
                if visual is None:
                    continue
                visual.visible = False

    @property
    def assignment_name(self):
        name = self._assignment_name

        if name is None or name not in self.data.available_assignments:
            return self.data.current_assignments

        return name

    @property
    def display_assignments(self):
        # Another set: without its synthetic components (Data.inactive_view).
        return self.data.inactive_view(self.data._assignments.get(self.assignment_name))

    @property
    def displays_active_assignments(self):
        return self.assignment_name == self.data.current_assignments

    def components_from_active(self, components, *, whole_neurons=False):
        return self.data.map_assignment_components(
            components,
            self.data.current_assignments,
            self.assignment_name,
            whole_neurons=whole_neurons,
        )

    def component_to_active(self, component):

        if self.is_inferred_component(component):
            component = NeuronComponent(
                component.neuron_id,
                None,
            )

        mapped = self.data.map_assignment_components(
            component,
            self.assignment_name,
            self.data.current_assignments,
        )
        return mapped[0] if mapped else None

    @property
    def display_focused_component(self):
        matches = self.components_from_active(
            self.state.focused_component,
            whole_neurons=True,
        )
        # Partners are ordered by shared assignments, overlap and distance.
        return matches[0] if matches else None

    @property
    def display_selected_components(self):
        return self.components_from_active(
            self.state.selected_components,
            whole_neurons=True,
        )

    def display_footprint_id(self, component):
        assignments = self.display_assignments

        if assignments is None or component is None:
            return None

        neuron = component.neuron_id
        session = component.session_id

        if (
            session is None
            or not 0 <= neuron < assignments.ids.shape[0]
            or not 0 <= session < assignments.ids.shape[1]
        ):
            return None

        footprint = int(assignments.ids[neuron, session])
        return footprint if footprint >= 0 else None

    def is_inferred_component(self, component):
        if component is None:
            return False

        record = self.plotting["data"].get(component.id)
        return bool(getattr(record, "inferred", False))

    def session_levels(self):
        """Stack level of each displayed session (session ID -> level).

        Collapsed: only sessions in which the focused neuron has a
        footprint (plus the session-filter session), evenly spaced.
        """
        n_sessions = len(self.data.sessions)
        parameters = self.controls["parameter"]
        assignments = self.display_assignments
        focused = self.display_focused_component

        if (
            not parameters.collapse_sessions_button.isChecked()
            or assignments is None
            or focused is None
            or not 0 <= focused.neuron_id < len(assignments.ids)
        ):
            # Inactive sessions are not displayed (their level stays empty).
            return {
                sid: sid for sid in range(n_sessions) if self.data.session_is_active(sid)
            }

        row = assignments.ids[focused.neuron_id]
        shown = set(np.flatnonzero(row >= 0).tolist())

        if parameters.checkbox_session_only.isChecked():
            shown.add(int(self.controls["session_filter"].slider.value()))

        shown = {
            sid for sid in shown if sid < n_sessions and self.data.session_is_active(sid)
        }
        return {sid: level for level, sid in enumerate(sorted(shown))}

    def plot_neurons(self, reset=False):

        # self.state.logger.debug(
        #     f"Plotting footprints for neurons: {self.state.selected_components}"
        # )

        self.state.timeit()

        parameters = self.controls["parameter"]

        infer = (
            parameters.checkbox_session_only.isChecked()
            and parameters.checkbox_show_inferred.isChecked()
        )

        filter_focused = parameters.checkbox_filter_focused.isChecked()

        focused = (
            self.display_focused_component if infer and not filter_focused else None
        )

        context = (
            self.assignment_name,
            id(self.display_assignments),
            infer,
            (int(self.controls["session_filter"].slider.value()) if infer else None),
            filter_focused if infer else False,
            None if focused is None else focused.neuron_id,
            parameters.threshold_spin.value(),
            parameters.z_stretch_spin.value(),
            tuple(sorted(self.session_levels().items())),
            # Meshes are cached per (neuron, session index): a reordering
            # (e.g. swapping two sessions) must rebuild them.
            tuple(id(session) for session in self.data.sessions),
        )

        if context != self._mesh_context:
            reset = True
            self._mesh_context = context

        if reset:
            self.clear()
        else:
            self.clear_visuals()

        self.plotting["alignment_preview"] = self.alignment_preview()

        assignments = self.display_assignments
        focused = self.display_focused_component
        selected = self.display_selected_components

        if assignments is None or assignments.union is None or focused is None:
            self._update_not_drawn_hint(None)
            return

        self.state.timeit("Initial setup")

        this_neuron = focused.neuron_id
        n_neurons = assignments.ids.shape[0]

        if not 0 <= this_neuron < n_neurons:
            self._update_not_drawn_hint(None)
            return

        selected_neurons = {
            int(component.neuron_id)
            for component in selected
            if 0 <= component.neuron_id < n_neurons
        }

        # Explicit inspection/manipulation (highlighted footprints, e.g.
        # curator sources and targets, and the current request) takes
        # precedence: their neurons are always shown, however far from the
        # focused neuron, and regardless of the scope, exclusion and
        # review-status filters.
        required_components = list(self.state.highlighted_components or ())

        request = self.state.current_request
        if request is not None:
            required_components.extend(request.components)

        required_display_components = self.components_from_active(required_components)

        required_neurons = {
            int(component.neuron_id)
            for component in required_display_components
            if 0 <= component.neuron_id < n_neurons
        }

        display_scope = self.controls["parameter"].display_scope
        if display_scope == "adjacent":

            ## find closeby neurons
            union_centroids = self.display_assignments.union.centroids
            distances = np.linalg.norm(
                union_centroids - union_centroids[this_neuron], axis=1
            )
            to_plot_neurons = np.where(distances <= self.state.adjacency_radius)[0]

        elif display_scope == "selection":
            to_plot_neurons = sorted(selected_neurons)
        else:
            raise ValueError(f"Unknown footprint display scope: " f"{display_scope!r}")

        show_excluded = self.controls["parameter"].checkbox_show_excluded.isChecked()

        if not show_excluded:
            included = self.display_assignments.union.included
            to_plot_neurons = [neuron for neuron in to_plot_neurons if included[neuron]]

        in_scope = {int(neuron) for neuron in to_plot_neurons}
        # The focused neuron is always drawn: excluded (flipping through
        # curator results) or without a union centroid (no distance to
        # itself), it was dropped and the panel showed only its tag.
        to_plot_neurons = np.union1d(
            np.asarray(to_plot_neurons, dtype=int),
            np.asarray(sorted(required_neurons | {int(this_neuron)}), dtype=int),
        )

        review_mask = ReviewStatusFilter.neuron_mask(
            to_plot_neurons,
            self.display_assignments.review_status,
            (self.controls["parameter"].review_filter.visible_statuses),
            focused_neuron_id=this_neuron,
            keep_focused=True,
        )

        # Shown only because they are required (re-plot when no longer).
        self.plotting["required_only"] = required_neurons - (
            in_scope & {int(neuron) for neuron in to_plot_neurons[review_mask]}
        )

        review_mask |= np.isin(
            to_plot_neurons,
            list(required_neurons),
        )

        to_plot_neurons = to_plot_neurons[review_mask]

        # Picking a target: the candidates are always shown (without fitting
        # the view to them), every other neuron fades.
        pick = self._pick
        picked = None
        if pick is not None:
            picked = {pick.source_id, *pick.candidates}
            to_plot_neurons = np.union1d(
                to_plot_neurons,
                np.asarray(
                    [n for n in pick.candidates if 0 <= n < n_neurons], dtype=int
                ),
            )

        session_filter = self.controls["parameter"].checkbox_session_only.isChecked()
        filter_focused = self.controls["parameter"].checkbox_filter_focused.isChecked()

        z_stretch = self.controls["parameter"].z_stretch_spin.value()

        self.state.timeit("Found neurons to plot and calculated centroid ranges")

        ## iterate through each neuron that should be plotted
        for neuron in to_plot_neurons:
            included = bool(self.display_assignments.union.included[neuron])

            if neuron == this_neuron:
                key = "focused"
                alpha = self.styles.opts[key]["alpha"] * (1.0 if included else 0.5)

            elif not included:
                key = "default"
                alpha = 0.1

            elif neuron in selected_neurons and display_scope == "selection":
                # "Nearby": selected neurons look like any other neighbour.
                key = "selected"
                alpha = self.styles.opts[key]["alpha"]

            else:
                key = "default"
                alpha = self.styles.opts[key]["alpha"]

            if picked is not None and neuron not in picked:
                key = "default"
                alpha = PICK_FADED_ALPHA

            ## create new data if not yet present
            if neuron not in self.plotting["data"]:
                self.add_footprint_data(
                    neuron,
                    alpha=alpha,
                    z_stretch=z_stretch,
                )

            if neuron not in self.plotting["data"]:
                # No drawable footprint remains for this neuron.
                continue

            if session_filter and (neuron != this_neuron or filter_focused):
                ## select subset of data for session-only display ...

                session_id = int(self.controls["session_filter"].slider.value())
                component = NeuronComponent(neuron, session_id)

                rec = self.plotting["data"].get(component.id, None)
                if rec is None:
                    continue

                verts = self.neuron_vertices(neuron)[
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
                verts = self.neuron_vertices(neuron)
                faces = self.plotting["data"][neuron].mesh_faces
                cols = self.plotting["data"][neuron].mesh_colors_default

            # cols[:, 3] = alpha  ## apply alpha (to account for selection changes)

            plot_options = self.styles.get_plot_options(
                key, "mesh", verts[:, 2], alpha=alpha, colors=cols
            )

            if (
                session_filter
                and (neuron != this_neuron or filter_focused)
                and rec.inferred
            ):
                # Preserve cyan even for selected/focused neurons.
                inferred_colors = np.array(cols, copy=True)
                inferred_colors[:, 3] = alpha
                plot_options["vertex_colors"] = inferred_colors

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
        # Fit the view when distant highlighted neurons were added.
        self.reset_camera(
            full=not self.camera_set or bool(self.plotting.get("required_only"))
        )
        self._update_not_drawn_hint(int(this_neuron))

    def _update_not_drawn_hint(self, neuron):
        """Below the neuron tag: why the focused neuron is not drawn (an
        empty panel otherwise looked broken)."""
        reason = None
        if neuron is not None and neuron not in self.plotting["visuals"]:
            reason = self._not_drawn_reason(neuron)

        label = self.not_drawn_hint
        label.setVisible(reason is not None)
        if reason is None:
            return
        label.setText(reason)
        label.adjustSize()
        anchor = (
            self.review_status_label.geometry()
            if self.review_status_label.isVisible()
            else self.reset_view_button.geometry()
        )
        label.move(8, anchor.bottom() + 6)
        label.raise_()

    def _not_drawn_reason(self, neuron):
        ids = self.display_assignments.ids
        sessions = [int(s) for s in np.flatnonzero(ids[neuron] >= 0)]
        if not sessions:
            return f"Neuron {neuron} has no footprints."

        def tags(session_ids):
            return ", ".join(session_tag(s) for s in session_ids[:6]) + (
                " …" if len(session_ids) > 6 else ""
            )

        empty = [
            s for s in sessions
            if self.data.sessions[s].footprints[:, int(ids[neuron, s])].nnz == 0
        ]
        if len(empty) == len(sessions):
            return (
                f"Neuron {neuron}: its footprint{'s are' if len(empty) > 1 else ' is'} "
                f"empty ({tags(empty)}) — nothing to draw."
            )

        shown = [s for s in sessions if self.data.session_is_active(s)]
        if not shown:
            return f"Neuron {neuron}: only in inactive sessions ({tags(sessions)})."

        parameters = self.controls["parameter"]
        if (
            parameters.checkbox_session_only.isChecked()
            and parameters.checkbox_filter_focused.isChecked()
        ):
            session_id = int(self.controls["session_filter"].slider.value())
            if session_id not in sessions:
                return (
                    f"Neuron {neuron} has no footprint in {session_tag(session_id)} "
                    f"(session filter); it is in {tags(sessions)}."
                )

        return (
            f"Neuron {neuron}: nothing above the display threshold "
            f"({tags(sessions)})."
        )

    def alignment_preview(self):
        """(session ID, key, move) for the session being aligned, while an
        alignment editor is open; None otherwise. ``move`` maps displayed
        (applied alignment) x/y points to their draft-alignment position,
        flow included; ``key`` identifies the preview. Display only."""
        draft = self.state.alignment_draft

        if (
            draft is None
            or getattr(draft, "external", False)
            or not getattr(self.state, "alignment_editors", None)
        ):
            return None

        session_id = draft.session_id

        if not (
            0 <= session_id < len(self.data.sessions)
            and self.data.sessions[session_id] is draft.session
            and np.isfinite(draft.values).all()
        ):
            return None

        dx, dy, angle = draft.values
        target = Remapping._rigid_matrix(draft.dims, (dy, dx), angle)
        target_flow = draft.effective_flow

        # Displayed footprints carry the applied alignment
        # (Remapping.apply_remap: unchanged if unsuccessful).
        applied = draft.session.remap
        if applied is not None and applied.success:
            current = np.asarray(applied.matrix, dtype=float)
            current_flow = applied.flow
        else:
            current, current_flow = np.eye(3), None

        if target_flow is current_flow and np.allclose(
            target, current, rtol=0, atol=1e-6
        ):
            return None

        key = (
            session_id,
            tuple(map(float, draft.values)),
            id(target_flow),
            id(applied),
        )

        def move(points):
            return _move_points(points, current, current_flow, target, target_flow)

        return session_id, key, move

    def preview_points(self, key, points):
        """Points of a footprint record, moved by the alignment preview.
        Inferred footprints come from other sessions and stay."""
        preview = self.plotting.get("alignment_preview")
        record = self.plotting["data"].get(key)

        if (
            preview is None
            or key[1] != preview[0]
            or record is None
            or record.inferred
        ):
            return points

        points = np.array(points, dtype=np.float32, copy=True)
        points[:, :2] = preview[2](points[:, :2])
        return points

    def neuron_vertices(self, neuron):
        """Mesh vertices of a neuron, with the alignment preview applied."""
        vertices = self.plotting["data"][neuron].mesh_vertices
        preview = self.plotting.get("alignment_preview")

        if preview is None:
            return vertices

        key = (neuron, preview[0])
        record = self.plotting["data"].get(key)

        if record is None or record.inferred:
            return vertices

        vertices = vertices.copy()
        part = slice(record.vertex_start, record.vertex_stop)
        vertices[part] = self.preview_points(key, vertices[part])
        return vertices

    def reset_camera(self, full=False):

        if not self.plotting["visuals"]:
            self.coordinates.hide()
            return

        ## obtain currently displayed neurons and their centroids
        displayed_neurons = list(self.plotting["visuals"].keys())
        # union_centroids = np.nanmean(
        #     self.data.neurons.centroids[displayed_neurons, ...], axis=1
        # )
        union_centroids = np.asarray(
            self.display_assignments.union.centroids[displayed_neurons, ...], dtype=float
        )[:, :2]
        union_centroids = union_centroids[np.isfinite(union_centroids).all(axis=1)]

        if not len(union_centroids):
            # No union centroid (e.g. a one-off whose union is empty): the
            # drawn footprints' own centres. A NaN center blanked the panel
            # for good (the camera's z center is kept between resets).
            union_centroids = np.asarray(
                [
                    record.center_xyz[:2]
                    for key, record in self.plotting["data"].items()
                    if isinstance(key, tuple) and key[0] in self.plotting["visuals"]
                ],
                dtype=float,
            ).reshape(-1, 2)
            union_centroids = union_centroids[np.isfinite(union_centroids).all(axis=1)]

        if not len(union_centroids):
            return

        centroid_min = union_centroids.min(axis=0)
        centroid_max = union_centroids.max(axis=0)

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
        # Levels can have gaps (inactive sessions): use the highest one.
        z_range = (0, (max(self.session_levels().values(), default=-1) + 1) * z_stretch)

        # Coordinate floor under the lowest session, over the shown area.
        self.coordinates.set_extent(x_range, y_range, -FLOOR_DEPTH * z_stretch)

        # A camera left without a finite center gets a full reset.
        if not np.all(np.isfinite(self.view.camera.center)):
            full = True

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
        self.rotate_on_focus()

    # ---------------------------------------------
    # Rotation on focus
    # ---------------------------------------------
    ROTATION_INTERVAL_MS = 30

    def rotate_on_focus(self, force=False):
        """One full turn around the focused neuron's union centroid, once
        per newly focused neuron (``force``: also for the same one)."""
        parameters = self.controls.get("parameter")
        focused = self.display_focused_component

        if (
            parameters is None
            or not parameters.checkbox_rotate_on_focus.isChecked()
            or focused is None
        ):
            return

        neuron = int(focused.neuron_id)
        if neuron == self._rotated_neuron and not force:
            return
        self._rotated_neuron = neuron

        centroids = self.display_assignments.union.centroids
        if not 0 <= neuron < len(centroids) or not np.all(
            np.isfinite(centroids[neuron][:2])
        ):
            return

        camera = self.view.camera
        camera.center = (
            float(centroids[neuron][0]),
            float(centroids[neuron][1]),
            camera.center[2],
        )

        self._rotation = {
            "start": time.perf_counter(),
            "azimuth": float(camera.azimuth),
            "expected": float(camera.azimuth),
            "duration": parameters.rotation_duration_spin.value(),
        }
        if self._rotation_timer is None:
            self._rotation_timer = QTimer(self.native)
            self._rotation_timer.setInterval(self.ROTATION_INTERVAL_MS)
            self._rotation_timer.timeout.connect(self._rotation_step)
        self._rotation_timer.start()

    def stop_rotation(self):
        if self._rotation_timer is not None:
            self._rotation_timer.stop()
        self._rotation = None

    def _rotation_step(self):
        rotation = self._rotation
        camera = self.view.camera

        if rotation is None:
            self.stop_rotation()
            return

        # The user turned the view: leave it to them.
        drift = (float(camera.azimuth) - rotation["expected"] + 180.0) % 360.0 - 180.0
        if abs(drift) > 0.5:
            self.stop_rotation()
            return

        fraction = (time.perf_counter() - rotation["start"]) / rotation["duration"]

        if fraction >= 1.0:
            camera.azimuth = rotation["azimuth"]
            self.stop_rotation()
        else:
            camera.azimuth = rotation["azimuth"] + 360.0 * fraction

        rotation_now = self._rotation
        if rotation_now is not None:
            rotation_now["expected"] = float(camera.azimuth)
        self.update()

    def style_records(self, selection, style):
        if selection is None:
            return []

        if isinstance(selection, NeuronComponent):
            selection = [selection]

        parameters = self.controls["parameter"]
        session_only = parameters.checkbox_session_only.isChecked()
        filter_focused = parameters.checkbox_filter_focused.isChecked()
        selected_session = int(self.controls["session_filter"].slider.value())

        focused = self.display_focused_component
        focused_neuron = None if focused is None else focused.neuron_id

        records = []
        seen = set()

        for component in selection:

            # Whole-neuron selection determines which stacks are displayed.
            # Only concrete footprint references receive these overlays.

            for key, record in self.plotting["data"].items():
                if not isinstance(key, tuple):
                    continue

                if (
                    style in {"selected", "focused", "hovered"}
                    and component.session_id is None
                    and not record.inferred
                ):
                    continue

                neuron, session = key

                if neuron != component.neuron_id:
                    continue

                if component.session_id is not None and session != component.session_id:
                    continue

                if neuron not in self.plotting["visuals"]:
                    continue

                if (
                    session_only
                    and session != selected_session
                    and not (neuron == focused_neuron and not filter_focused)
                ):
                    continue

                if key not in seen:
                    seen.add(key)
                    records.append(record)

        return records

    def add_footprint_data(
        self,
        neuron: int,
        alpha=0.7,
        z_stretch=5.0,
    ):
        parameters = self.controls["parameter"]
        threshold = parameters.threshold_spin.value()

        assignments = self.display_assignments
        selected_session = int(self.controls["session_filter"].slider.value())

        focused = self.display_focused_component
        focused_neuron = None if focused is None else focused.neuron_id

        infer_here = (
            parameters.checkbox_session_only.isChecked()
            and parameters.checkbox_show_inferred.isChecked()
            and (
                neuron != focused_neuron
                or parameters.checkbox_filter_focused.isChecked()
            )
        )

        vertices_all = []
        faces_all = []
        colors_all = []
        vertex_offset = 0
        face_offset = 0

        levels = self.session_levels()

        for session in self.data.sessions:
            if (
                session.id >= assignments.ids.shape[1]
                or not session.status["spatial_loaded"]
                or session.id not in levels
            ):
                continue

            component = NeuronComponent(neuron, session.id)
            footprint_id = self.display_footprint_id(component)
            inferred = footprint_id is None

            if inferred:
                if not infer_here or session.id != selected_session:
                    continue

                try:
                    footprint = infer_aligned_footprint(
                        self.data.sessions,
                        assignments.ids,
                        assignments.union,
                        session.id,
                        neuron,
                    )
                except ValueError as exc:
                    error_key = (neuron, session.id)

                    if error_key not in self._inference_errors:
                        self._inference_errors.add(error_key)
                        self.state.logger.warning(
                            f"Cannot infer neuron {neuron} in "
                            f"session {session.id}: {exc}"
                        )
                    continue

                if footprint is None:
                    continue

                rgba = INFERRED_RGBA.copy()

            else:
                # A whole excluded neuron is drawn only when focused or
                # required (plot_neurons): then with all its footprints.
                if (
                    not parameters.checkbox_show_excluded.isChecked()
                    and not session.included[footprint_id]
                    and assignments.union.included[neuron]
                ):
                    continue

                footprint = session.footprints[:, footprint_id]
                rgba = np.array(
                    self.state.session_colors[session.id],
                    dtype=np.float32,
                    copy=True,
                )
                rgba[3] = 1.0

            vertices, faces, colors, pick_points = footprint_to_mesh(
                footprint,
                session.dims,
                z_offset=float(levels[session.id]) * z_stretch,
                z_scale=z_stretch,
                z_thr=threshold,
                rgba=rgba,
                base_vertex_offset=vertex_offset,
            )

            if not len(vertices) or not len(faces):
                continue

            vertex_stop = vertex_offset + len(vertices)
            face_stop = face_offset + len(faces)

            self.plotting["data"][component.id] = FootprintRecord(
                key=component.id,
                neuron=neuron,
                vertex_start=vertex_offset,
                vertex_stop=vertex_stop,
                face_start=face_offset,
                face_stop=face_stop,
                pick_points=pick_points,
                center_xyz=np.nanmean(vertices, axis=0),
                inferred=inferred,
            )

            vertices_all.append(vertices)
            faces_all.append(faces)
            colors_all.append(colors)

            vertex_offset = vertex_stop
            face_offset = face_stop

        if not vertices_all:
            return

        self.plotting["data"][neuron] = NeuronRecord(
            mesh_vertices=np.vstack(vertices_all),
            mesh_faces=np.vstack(faces_all),
            mesh_colors_default=np.vstack(colors_all),
            thr=threshold,
        )

    def find_closest_component(self, mouse_pos) -> Optional[NeuronComponent]:

        if not self.plotting["visuals"]:
            return None

        center_radius_px = 80
        point_radius_px = 10

        parameters = self.controls["parameter"]
        session_filter = parameters.checkbox_session_only.isChecked()
        filter_focused = parameters.checkbox_filter_focused.isChecked()
        session_id = int(self.controls["session_filter"].slider.value())

        focused = self.display_focused_component
        focused_neuron = None if focused is None else focused.neuron_id

        keys = [
            key
            for key in self.plotting["data"]
            if (
                isinstance(key, tuple)
                and key[0] in self.plotting["visuals"]
                and (
                    not session_filter
                    or key[1] == session_id
                    or (key[0] == focused_neuron and not filter_focused)
                )
            )
        ]

        if not keys:
            return None
        centers = np.asarray(
            [
                self.preview_points(key, self.plotting["data"][key].center_xyz[None])[0]
                for key in keys
            ],
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
                self.plotting["visuals"][keys[int(i)][0]],
                self.preview_points(key, rec.pick_points),
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
        self, component: NeuronComponent, radius=None
    ) -> tuple[np.ndarray, np.ndarray]:
        # union_centroids = np.nanmean(self.data.neurons.centroids, axis=1)
        union_centroids = self.display_assignments.union.centroids
        neuron_id = component.neuron_id
        d = np.linalg.norm(union_centroids - union_centroids[neuron_id], axis=1)

        if radius is None:
            radius = min(self.REASSIGN_RADIUS_PX, self.state.adjacency_radius)
        neuron_ids = np.where(d <= radius)[0]
        neuron_ids = [n for n in neuron_ids if n != neuron_id]

        self.state.logger.debug(
            f"Neuron {neuron_id} has {len(neuron_ids)} adjacent neurons within radius {self.state.adjacency_radius}: {neuron_ids}"
        )

        return np.array(neuron_ids), np.array(d[neuron_ids])

    def plot_data_from_rec(self, rec, style: str) -> dict[str, np.ndarray]:

        neuron_rec = self.plotting["data"][rec.neuron]
        verts = self.preview_points(
            rec.key, neuron_rec.mesh_vertices[rec.vertex_start : rec.vertex_stop]
        )
        faces = neuron_rec.mesh_faces[rec.face_start : rec.face_stop] - rec.vertex_start
        cols = self.styles.get_color_array(
            style,
            verts[:, 2],
            alpha=HOVER_NEURON_ALPHA if style == "hovered_neuron" else 0.6,
        )

        if rec.inferred:
            rgba = INFERRED_RGBA.copy()

            # Highlight through brightness while retaining cyan.
            lightening = 0.35 if style == "hovered" else 0.15
            rgba[:3] += lightening * (1.0 - rgba[:3])

            # Keep the existing overlay opacity.
            rgba[3] = 0.6
            cols = np.tile(rgba, (len(verts), 1))

        return {"vertices": verts, "faces": faces, "vertex_colors": cols}

    def show_hovered_neuron(self, components):
        """The other footprints of the hovered footprint's neuron in a faint
        hover colour, fading out over HOVER_NEURON_FADE_S; None hides.

        All footprints form one mesh (one upload per hover change) and the
        fade only changes an alpha uniform (no uploads while fading): one
        overlay per footprint made large datasets lag (60 sessions)."""
        from PySide6.QtCore import QElapsedTimer, QTimer
        from vispy.visuals.filters import Alpha

        if "hovered_neuron" not in self.plotting["overlays"]:
            self.add_overlay("hovered_neuron")
        visuals = self.plotting["overlays"].get("hovered_neuron") or []
        for visual in visuals:
            visual.visible = False

        timer = self._hover_fade_timer
        if timer is None:
            timer = self._hover_fade_timer = QTimer(self.native)
            timer.setInterval(40)
            timer.timeout.connect(self._step_hover_fade)
            self._hover_fade_clock = QElapsedTimer()
        timer.stop()
        self._hover_fade = []

        records = self.style_records(components, "hovered_neuron") if components else []
        parts = [self.plot_data_from_rec(rec, "hovered_neuron") for rec in records]
        parts = [part for part in parts if len(part["faces"])]
        if not visuals or not parts:
            self.update()
            return

        offsets = np.cumsum([0] + [len(part["vertices"]) for part in parts[:-1]])
        visual = visuals[0]
        visual.set_data(
            vertices=np.concatenate([part["vertices"] for part in parts]),
            faces=np.concatenate(
                [part["faces"] + offset for part, offset in zip(parts, offsets)]
            ),
            vertex_colors=np.concatenate([part["vertex_colors"] for part in parts]),
        )
        # Per visual (vispy visuals are frozen: no new attributes).
        fade = self._hover_alpha.get(id(visual))
        if fade is None or fade[0] is not visual:
            fade = (visual, Alpha(1.0))
            visual.attach(fade[1])
            self._hover_alpha = {id(visual): fade}
        fade = fade[1]
        fade.alpha = 1.0
        visual.visible = True
        self._hover_fade = [(visual, fade)]

        self._hover_fade_clock.start()
        timer.start()
        self.update()

    def _step_hover_fade(self):
        remaining = 1.0 - self._hover_fade_clock.elapsed() / (1000.0 * HOVER_NEURON_FADE_S)

        for visual, fade in self._hover_fade:
            if remaining <= 0:
                visual.visible = False
            else:
                fade.alpha = remaining

        if remaining <= 0:
            self._hover_fade = []
            self._hover_fade_timer.stop()
        self.update()

    def on_mouse_release(self, event):

        if self._pick is not None:
            self._on_pick_release(event)
            event.handled = True
            return

        component = (
            self.find_closest_component(event.pos) if event.pos is not None else None
        )

        if self.is_inferred_component(component):
            event.handled = True

            # No real session footprint exists to manipulate.
            if self.state.current_request is not None:
                return

            if event.button == 2:
                self.open_footprint_context_menu(component)

            elif event.button == 1 and not self.is_drag(event.pos):
                mapped = self.component_to_active(component)

                if mapped is None:
                    return

                if self.changes_on_click == "selected":
                    self.state.update_selected_components(mapped, event.modifiers)
                elif self.changes_on_click == "focused":
                    self.state.focused_component = mapped
                else:
                    self.state.update_highlighted_components(
                        mapped,
                        event.modifiers,
                        max_components=3,
                    )

            return

        if not self.displays_active_assignments:
            if event.pos is None:
                return

            # A pending manipulation owns global interaction state.
            if self.state.current_request is not None:
                event.handled = True
                return

            component = self.find_closest_component(event.pos)

            if event.button == 2:
                if component is not None:
                    self.open_footprint_context_menu(component)

            elif event.button == 1 and not self.is_drag(event.pos):
                mapped = self.component_to_active(component)

                if component is not None and mapped is None:
                    self.handle_tooltip(component)
                elif self.changes_on_click == "highlighted":
                    self.state.update_highlighted_components(
                        mapped,
                        event.modifiers,
                        max_components=3,
                    )
                elif self.changes_on_click == "selected":
                    self.state.update_selected_components(
                        mapped,
                        event.modifiers,
                    )
                elif self.changes_on_click == "focused":
                    self.state.focused_component = mapped

            event.handled = True
            return

        request = self.state.current_request

        # Allow inspection and conflict resolution during neuron merging.
        if (
            event.button == 2
            and event.pos is not None
            and (request is None or request.type == "neuron_merge")
        ):
            component = self.find_closest_component(event.pos)
            if component is not None:
                self.open_footprint_context_menu(component)

            event.handled = True
            return

        if request is not None:
            if (
                event.button == 1
                and event.pos is not None
                and not self.is_drag(event.pos)
            ):
                self._on_request_interaction(event)

            # The request continues to own the highlighted components.
            event.handled = True
            return

        super().on_mouse_release(event)

    def on_mouse_press(self, event):
        self.native.setFocus(Qt.FocusReason.MouseFocusReason)
        super().on_mouse_press(event)

    def update_hover(self, component):
        self._hover_component = component
        if self._pick is not None:
            self._update_pick_hover(component)
            return
        self.handle_tooltip(component)
        self.state.update_hovered_components(self.component_to_active(component))

    def on_mouse_move(self, event):
        super().on_mouse_move(event)
        self._update_coordinate_readout(event)

    def _update_coordinate_readout(self, event):
        """Point under the cursor on the hovered footprint's session plane
        (FootprintCoordinates)."""
        component = self._hover_component
        if isinstance(component, list):
            component = component[0] if component else None
        if (
            event.is_dragging
            or event.pos is None
            or component is None
            or component.session_id is None
            or not self.plotting["visuals"]
        ):
            self.coordinates.clear_hover()
            return
        level = self.session_levels().get(int(component.session_id))
        if level is None:
            self.coordinates.clear_hover()
            return
        z_stretch = self.controls["parameter"].z_stretch_spin.value()
        self.coordinates.show_hover(
            event.pos,
            float(level) * z_stretch,
            session_label(self.data, int(component.session_id)),
        )

    def handle_tooltip(self, component=None):
        if isinstance(component, list):
            component = component[0] if component else None

        if component is None:
            self._tooltip_statistics.clear()
            QToolTip.hideText()
            return

        inferred = self.is_inferred_component(component)

        if self.displays_active_assignments and not inferred:
            return super().handle_tooltip(component)
        self._tooltip_statistics.clear()

        lines = [
            f"Assignments: {assignment_label(self.assignment_name)}",
            f"Neuron ID: {component.neuron_id}",
        ]

        if component.session_id is not None:
            lines.append(f"Session: {session_label(self.data, component.session_id)}")

            if inferred:
                lines.extend(
                    [
                        "Inferred footprint — not detected in this session",
                        "Union shape at an interpolated centroid",
                    ]
                )
            else:
                lines.append(f"Footprint ID: {self.display_footprint_id(component)}")

        if not self.displays_active_assignments:
            counterpart = self.component_to_active(component)

            lines.append(
                "No corresponding neuron in active assignments."
                if counterpart is None
                else f"Active counterpart: neuron {counterpart.neuron_id}"
            )

        show_tooltip(
            QCursor.pos(),
            "\n".join(lines),
            self.native,
        )

    def calculate_tracking_statistics(
        self, this_component: NeuronComponent, ref_neuron: int, with_session: str | int
    ) -> tuple[Optional[float], Optional[float], Optional[float], Optional[int]]:

        this_fp_id = self.display_footprint_id(this_component)

        ## find, where compared neuron has footprints
        fp_ids = self.display_assignments.ids[ref_neuron, :]
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
        ref_fp_id = self.display_footprint_id(ref_component)

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

    def info_html(
        self,
        this_component: NeuronComponent,
        candidate: int,
        which: str = "previous",
        cmap=MATCH_QUALITY_CMAP,
    ):
        """Match quality of a footprint vs. the candidate's footprint in the
        previous / next session it has one (coloured html)."""
        similarity, shift, p_same, session_id = self.calculate_tracking_statistics(
            this_component, candidate, with_session=which
        )
        if session_id is None:
            return f"no {which} session found"
        ds = abs(this_component.session_id - session_id)

        arrow = "\u25bc" if which == "previous" else "\u25b2"
        html_session = (
            f"\u0394s="
            + get_colored_label(f"{ds}", float(np.clip(ds / 10.0, 0.0, 1.0)), cmap)
            + f"({session_tag(session_id)})"
        )
        html_probability, html_similarity, html_shift = match_quality_html(
            similarity, shift, p_same, cmap, units=self.data.units
        )
        return f"{arrow} {html_probability} - {html_session}, {html_similarity}, {html_shift}"
    def footprint_set_statistics(self, union_footprint, neuron_id):
        """Similarity, shift (px) and model probability between a footprint
        (e.g. the union of moved footprints) and a neuron's union footprint."""
        other = self.display_assignments.union.footprints[:, neuron_id]
        return self._set_statistics_between(union_footprint, other)

    def _set_statistics_between(self, footprint, other):
        """footprint_set_statistics() between two footprint columns."""
        if footprint is None or other is None or footprint.nnz == 0 or other.nnz == 0:
            return None, None, None

        similarity, _, shift = calculate_img_correlation(
            A1=footprint,
            A2=other,
            dims=self.display_assignments.union.dims,
            crop=True,
            binary=False,
            shift=True,
            mode="cosine_union",
            gamma=0.1,
            shift_optimized=True,
        )
        d_shift = float(np.sqrt(np.sum([s**2 for s in shift])))

        p_same = None
        if self.data.model is not None and self.data.model.fitted:
            p_same = self.data.model.f_same(d_shift, similarity)[0]

        return similarity, d_shift, p_same

    def set_info_html(self, union_footprint, neuron_id, cmap=MATCH_QUALITY_CMAP):
        """info_html() for a set of footprints vs. a neuron's union."""
        similarity, shift, p_same = self.footprint_set_statistics(
            union_footprint, neuron_id
        )

        if similarity is None:
            return "no union footprint to compare"

        html_probability, html_similarity, html_shift = match_quality_html(
            similarity, shift, p_same, cmap, units=self.data.units
        )
        return f"\u25c6 {html_probability or 'p=–'} - union, {html_similarity}, {html_shift}"
    def _activate_display_assignments(self, component):
        if self.data.change_assignments(self.assignment_name):
            self.state.focused_component = component

    def open_footprint_context_menu(self, this_component: NeuronComponent):

        if self.is_inferred_component(this_component):
            menu = QMenu(self.native)
            add_menu_section(
                menu,
                f"Inferred footprint — neuron {this_component.neuron_id}",
                title=True,
            )

            mapped = self.component_to_active(this_component)

            focus = menu.addAction("Focus corresponding neuron")
            focus.setEnabled(mapped is not None)
            focus.triggered.connect(
                lambda checked=False: self.set_focused_footprint(this_component)
            )

            if not self.displays_active_assignments:
                activate = menu.addAction(
                    "Make these assignments active and focus neuron"
                )
                activate.triggered.connect(
                    lambda checked=False: self._activate_display_assignments(
                        NeuronComponent(
                            this_component.neuron_id,
                            None,
                        )
                    )
                )

            menu.exec(QCursor.pos())
            menu.deleteLater()
            return

        if not self.displays_active_assignments:
            menu = QMenu(self.native)
            add_menu_section(
                menu,
                f"{assignment_label(self.assignment_name)}: "
                f"neuron {this_component.neuron_id}",
                title=True,
            )

            counterpart = self.component_to_active(this_component)

            focus_action = menu.addAction(
                "Focus corresponding neuron in active assignments"
            )
            focus_action.setEnabled(counterpart is not None)
            focus_action.triggered.connect(
                lambda checked=False: self.set_focused_footprint(this_component)
            )

            activate_action = menu.addAction(
                "Make these assignments active and focus this footprint"
            )
            activate_action.triggered.connect(
                lambda checked=False: self._activate_display_assignments(this_component)
            )

            menu.exec(QCursor.pos())
            menu.deleteLater()
            return

        if this_component.session_id is None:
            return

        menu = QMenu(self.native)
        menu.setToolTipsVisible(True)
        add_menu_section(
            menu,
            f"Neuron {this_component.neuron_id} · "
            f"{session_label(self.data, this_component.session_id)}",
            title=True,
        )
        neuron_id = int(this_component.neuron_id)
        no_request = self.state.current_request is None

        ## === inspect ===
        add_menu_section(menu, "Inspect")
        menu.addAction("Set focus").triggered.connect(
            lambda: self.set_focused_footprint(this_component)
        )
        self.add_neuron_highlight_menu(menu, neuron_id)
        self.add_compare_menu(menu, this_component)
        self.add_preset_check_menu(menu, this_component)

        if not self.data.sessions[this_component.session_id].status["traces_loaded"]:
            menu.addAction("Load trace for this session").triggered.connect(
                lambda: self.toggle_session_data(this_component.session_id, "traces"),
            )

        ## === assignment (which neuron) ===
        add_menu_section(menu, "Assignment")
        reassign = menu.addAction("Reassign this footprint…")
        reassign.setEnabled(len(self.data.sessions) > 1)
        reassign.setToolTip(
            "Then click the new neuron in the view or in the list below it."
        )
        reassign.triggered.connect(lambda: self.start_reassign_pick(this_component))
        self.add_move_footprints_menu(menu, this_component)

        ## === shape of this footprint ===
        add_menu_section(menu, "Shape")
        split = menu.addAction("Split this footprint…")
        split.setEnabled(no_request)
        split.setToolTip("Then click the source footprints (other sessions).")
        split.triggered.connect(lambda: self.start_split(this_component))

        merge = menu.addAction("Merge with other footprints…")
        merge.setEnabled(no_request)
        merge.setToolTip(
            "Then click the other target footprints of this session, "
            "continue and click the source."
        )
        merge.triggered.connect(lambda: self.start_merge(this_component))

        self.add_cut_action(menu, this_component, no_request)
        self.add_replace_with_union_menu(menu, this_component)

        if self.data.shape_decisions():
            from catan.gui.GUI_elements.fragments.shape_decision_dialog import (
                open_decisions_list,
            )

            menu.addAction("Shape decisions…").triggered.connect(
                lambda: open_decisions_list(self.native, self.state, self.data)
            )

        ## === inclusion ===
        add_menu_section(menu, "Inclusion")
        menu.addAction(
            "Exclude this footprint"
            if self.data.is_included(this_component)
            else "Include this footprint"
        ).triggered.connect(lambda: self.toggle_included(this_component))
        menu.addAction(
            f"Exclude neuron {neuron_id}"
            if self.data.is_included(neuron_id)
            else f"Include neuron {neuron_id}"
        ).triggered.connect(lambda: self.toggle_included(neuron_id))

        submenu_remove = QMenu("Remove", menu)
        submenu_remove.addAction("this footprint").triggered.connect(
            lambda: self.remove_component(this_component)
        )
        submenu_remove.addAction(f"neuron {neuron_id}").triggered.connect(
            lambda: self.remove_component(neuron_id)
        )
        menu.addMenu(submenu_remove)

        menu.exec(QCursor.pos())
        menu.deleteLater()

    # Reassignment candidates: within this radius (px), "Load more" adds
    # this much each time.
    REASSIGN_RADIUS_PX = 10.0
    REASSIGN_RADIUS_STEP_PX = 10.0

    # Reassignment candidates: within this radius (px), "Load more" adds
    # this much each time.
    REASSIGN_RADIUS_PX = 10.0
    REASSIGN_RADIUS_STEP_PX = 10.0

    # ---------------------------------------------------------------
    # Picking a target neuron (reassign a footprint, move footprints):
    # the context menu only starts it; the candidates are listed in the
    # bar at the bottom of the view (TargetPickerBar) and can be chosen
    # there or by clicking a neuron in the view, which stays rotatable.
    # ---------------------------------------------------------------

    def start_reassign_pick(self, component):
        """"Reassign this footprint…": pick its new neuron."""
        self._start_pick(
            TargetPick(
                "reassign",
                component,
                int(component.neuron_id),
                [int(component.session_id)],
                min(self.REASSIGN_RADIUS_PX, self.state.adjacency_radius),
            )
        )

    def start_move_pick(self, component, moved):
        """"Move footprints of neuron N ▸ <scope>…": pick the target neuron
        of the footprints in sessions ``moved``."""
        if self.state.current_request is not None:
            return
        self._start_pick(
            TargetPick(
                "move",
                component,
                int(component.neuron_id),
                [int(s) for s in moved],
                max(self.MERGE_RADIUS_PX, self.state.adjacency_radius),
            )
        )

    @property
    def picking(self):
        return self._pick is not None

    def _start_pick(self, pick):
        if not self.displays_active_assignments:
            return
        self.cancel_pick()

        # Restored when the pick ends.
        pick.previous_highlight = (
            list(self.state.highlighted_components or ()),
            dict(self.state.highlight_roles),
        )
        self._pick = pick
        self._fill_pick(pick)

        # What is moved: violet (source).
        moved = [NeuronComponent(pick.source_id, s) for s in pick.moved]
        self.state.update_highlighted_components(
            moved, roles={c: "source" for c in moved}
        )
        self.pick_bar.show_pick(pick)
        self.native.setFocus(Qt.FocusReason.OtherFocusReason)
        self.pick_bar.pick_changed.emit()

    def cancel_pick(self, restore=True):
        pick = self._pick
        if pick is None:
            return
        self._pick = None
        self.pick_bar.clear_pick()
        self.state.update_hovered_components(None)
        QToolTip.hideText()
        if restore:
            self._restore_highlight(pick.previous_highlight)
        self.pick_bar.pick_changed.emit()

    def widen_pick(self):
        """"Load more": a wider radius for the candidates."""
        pick = self._pick
        if pick is None:
            return
        pick.radius += (
            self.MERGE_RADIUS_STEP_PX
            if pick.kind == "move"
            else self.REASSIGN_RADIUS_STEP_PX
        )
        self._fill_pick(pick)
        self.pick_bar.show_pick(pick)
        self.pick_bar.pick_changed.emit()

    def choose_pick_target(self, neuron):
        """Apply the pick: ``neuron`` the target, None a new neuron."""
        pick = self._pick
        if pick is None:
            return
        ids = self.data.assignments.ids
        if neuron is not None:
            neuron = int(neuron)
            if neuron == pick.source_id or not 0 <= neuron < ids.shape[0]:
                self.pick_bar.set_message(
                    f"Neuron {neuron} is the one being moved: pick another neuron."
                )
                return
        elif not pick.allow_new:
            return

        self.cancel_pick()
        if pick.kind == "move":
            self.start_neuron_merge(
                pick.source_id, neuron, sessions=np.asarray(pick.moved, dtype=int)
            )
        else:
            self.change_neuron_assignment_dialog(pick.component, neuron)

    def hover_pick_target(self, neuron):
        """Highlight all footprints of ``neuron`` (None: nothing)."""
        ids = self.data.assignments.ids
        if neuron is None or not 0 <= neuron < ids.shape[0]:
            self.state.update_hovered_components(None)
            return
        self.state.update_hovered_components(
            [NeuronComponent(int(neuron), int(s)) for s in np.flatnonzero(ids[neuron] >= 0)]
            or None
        )

    def _on_pick_release(self, event):
        if event.button == 2:
            self.cancel_pick()
            return
        if event.button != 1 or event.pos is None or self.is_drag(event.pos):
            return
        component = self.find_closest_component(event.pos)
        if component is not None:
            self.choose_pick_target(int(component.neuron_id))

    def _update_pick_hover(self, component):
        """Hovering a neuron in the view: all its footprints, its row in the
        bar and its match quality as tooltip."""
        pick = self._pick
        neuron = None if component is None else int(component.neuron_id)
        self.hover_pick_target(neuron)
        self.pick_bar.mark(neuron)
        if neuron is None:
            QToolTip.hideText()
            return
        if neuron == pick.source_id:
            text = f"<b>Neuron {neuron}</b><br>being moved"
        else:
            label, lines = self._pick_row(pick, neuron)
            text = (
                f"<b>Into {label}</b><br>"
                + "<br>".join(lines)
                + "<br><i>Click to choose · Esc cancels</i>"
            )
        show_tooltip(QCursor.pos(), text, self.native)

    def _union_distance(self, first, second):
        centroids = self.display_assignments.union.centroids
        distance = float(np.linalg.norm(centroids[first] - centroids[second]))
        return distance if np.isfinite(distance) else np.inf

    def _pick_row(self, pick, neuron, distance=None):
        """(label, html lines) of a candidate; computed once per pick."""
        if neuron in pick.rows:
            return pick.rows[neuron]

        ids = self.display_assignments.ids
        units = self.data.units
        if distance is None:
            distance = self._union_distance(pick.source_id, neuron)
        details = [f"{int(np.count_nonzero(ids[neuron] >= 0))} footprints"]
        details.append(
            f"{units.to_display(distance, 'distance'):.1f} "
            f"{units.label('distance')} apart"
            if np.isfinite(distance)
            else "no union"
        )

        if pick.kind == "move":
            if not self.display_assignments.union.included[neuron]:
                details.append("excluded")
            lines = [self.set_info_html(pick.moved_union, neuron)]
        else:
            session = pick.moved[0]
            if ids[neuron, session] >= 0:
                details.append(f"has footprint {int(ids[neuron, session])}")
            lines = [
                self.info_html(pick.component, neuron, "next"),
                self.info_html(pick.component, neuron, "previous"),
            ]

        pick.rows[neuron] = (f"neuron {neuron} ({', '.join(details)})", lines)
        return pick.rows[neuron]

    def _fill_pick(self, pick):
        """Title, summary and candidate sections of the pick."""
        units = self.data.units
        ids = self.display_assignments.ids
        source_id = pick.source_id
        focused = self.state.focused_component
        focused_neuron = None if focused is None else int(focused.neuron_id)
        radius_text = (
            f"{units.to_display(pick.radius, 'distance'):.0f} {units.label('distance')}"
        )
        sections = []
        listed = {source_id}
        pick.note = ""

        def section(title, neurons):
            rows = []
            for neuron, distance in neurons:
                if neuron in listed:
                    continue
                listed.add(neuron)
                self._pick_row(pick, neuron, distance)
                rows.append(neuron)
            if rows:
                sections.append((title, rows))
            return rows

        nearby, distances = self.find_nearby_neurons(pick.component, radius=pick.radius)
        nearby = sorted(zip((int(n) for n in nearby), distances), key=lambda x: x[1])

        # Before any row: the move rows compare against it.
        if pick.kind == "move" and pick.moved_union is None:
            pick.moved_union = self.data.partial_union_footprint(source_id, pick.moved)

        ## the focused neuron is always offered (no distance limit: choosing
        ## it is deliberate)
        if focused_neuron is not None and 0 <= focused_neuron < ids.shape[0]:
            section("Focused neuron", [(focused_neuron, None)])

        if pick.kind == "reassign":
            session = pick.moved[0]
            pick.title = (
                f"Reassign neuron {source_id} · {session_label(self.data, session)}"
            )
            pick.summary = [
                "<b>Current assignment</b>",
                self.info_html(pick.component, source_id, "next"),
                self.info_html(pick.component, source_id, "previous"),
            ]
            ## neurons that already have a footprint in this session come
            ## last (reassigning asks how to resolve the conflict)
            section(
                f"Within {radius_text}",
                [(n, d) for n, d in nearby if ids[n, session] < 0],
            )
            section(
                f"Already have a footprint in {session_label(self.data, session)}",
                [(n, d) for n, d in nearby if ids[n, session] >= 0],
            )
            pick.allow_new = True
            pick.new_tooltip = ""

        else:
            moved = np.asarray(pick.moved, dtype=int)
            sessions = np.flatnonzero(ids[source_id] >= 0)
            remaining = np.setdiff1d(sessions, moved)
            pick.title = (
                f"Move footprints of neuron {source_id}: "
                f"{session_labels(self.data, moved, short=True)}"
            )
            if len(remaining):
                # Moved footprints vs. the footprints that would remain.
                rest = self.data.partial_union_footprint(source_id, remaining)
                similarity, shift, p_same = self._set_statistics_between(
                    pick.moved_union, rest
                )
                html = match_quality_html(
                    similarity, shift, p_same, MATCH_QUALITY_CMAP, units=units
                )
                pick.summary = [
                    f"\u25c6 {html[0] or 'p=–'} - {len(remaining)} remaining "
                    f"footprints, {html[1]}, {html[2]}"
                ]
            else:
                pick.summary = ["the whole neuron (nothing remains)"]

            ## selected neurons (Overview / navigation bar) within a wider
            ## radius than the neighbours, nearest first
            selected = [
                n
                for n in dict.fromkeys(
                    int(c.neuron_id) for c in self.state.selected_components or ()
                )
                if n not in listed and 0 <= n < ids.shape[0] and (ids[n] >= 0).any()
            ]
            if selected:
                selected_radius = max(self.MERGE_SELECTED_RADIUS_PX, pick.radius)
                near = sorted(
                    (
                        (n, d)
                        for n in selected
                        if (d := self._union_distance(source_id, n)) <= selected_radius
                    ),
                    key=lambda x: x[1],
                )
                n_far = len(selected) - len(near)
                section("Selected neurons", near[: self.MERGE_MAX_SELECTED])
                if n_far:
                    pick.note = (
                        f"{n_far} more selected beyond "
                        f"{units.to_display(selected_radius, 'distance'):.0f} "
                        f"{units.label('distance')}"
                    )

            section(f"Within {radius_text}", nearby)
            pick.allow_new = bool(len(remaining))
            pick.new_tooltip = (
                "" if pick.allow_new
                else "All footprints into a new neuron would change nothing."
            )

        if not any(title.startswith("Within") for title, _ in sections):
            sections.append((f"Within {radius_text}", []))
        pick.sections = sections
        pick.candidates = listed - {source_id}
        step = (
            self.MERGE_RADIUS_STEP_PX if pick.kind == "move" else self.REASSIGN_RADIUS_STEP_PX
        )
        pick.more_text = (
            f"Load more (within "
            f"{units.to_display(pick.radius + step, 'distance'):.0f} "
            f"{units.label('distance')})"
        )

    def _restore_highlight(self, previous):
        """Highlight from before a request / pick (its footprints that still
        exist)."""
        if previous is None or not previous[0]:
            self.state.update_highlighted_components(None)
            return

        components, roles = previous
        ids = self.data.assignments.ids
        components = [
            c
            for c in components
            if c.neuron_id < len(ids)
            and (c.session_id is None or ids[c.neuron_id, c.session_id] >= 0)
        ]
        self.state.update_highlighted_components(
            components or None,
            roles={c: roles[c] for c in components if c in roles},
        )
    def add_preset_check_menu(self, menu: QMenu, component: NeuronComponent):
        """Submenu: check the footprint's neuron against a curator preset."""
        from catan.gui.data.curation_filter_presets import (
            PRESET_CATEGORIES,
            UNCATEGORIZED_LABEL,
            shared_preset_store,
        )
        from catan.gui.GUI_elements.fragments.preset_check_dialog import (
            WORKING_FILTER_KEY,
        )

        store = shared_preset_store(self.state)
        submenu = QMenu("Check against preset", menu)
        submenu.setToolTipsVisible(True)
        submenu.setToolTip(
            "Does this neuron match the preset? Shows its values against "
            "every threshold of the preset."
        )

        working = "Curator's current filter"
        if store.working_dirty:
            working += " (unsaved changes)"
        submenu.addAction(working).triggered.connect(
            lambda checked=False: self.open_preset_check(component, WORKING_FILTER_KEY)
        )

        category = object()
        for preset in store.presets():
            if preset.category != category:
                category = preset.category
                add_menu_section(
                    submenu, PRESET_CATEGORIES.get(category, UNCATEGORIZED_LABEL)
                )
            submenu.addAction(preset.name).triggered.connect(
                lambda checked=False, key=preset.key: self.open_preset_check(component, key)
            )

        menu.addMenu(submenu)

    def add_compare_menu(self, menu: QMenu, component: NeuronComponent):
        """"Compare <this footprint> with ▸": the highlighted footprints, the
        focused footprint, the selected neurons and its own neuron (always
        there, so the entry is never empty)."""

        def label(c):
            if c.session_id is None:
                return f"neuron {c.neuron_id}"
            return f"N{c.neuron_id} S{c.session_id}"

        def footprint(c):
            return c is not None and c.session_id is not None and c != component

        sections = []
        highlighted = [
            c for c in dict.fromkeys(self.state.highlighted_components or ()) if footprint(c)
        ]
        if highlighted:
            sections.append(("Highlighted", highlighted))
        focused = self.state.focused_component
        if footprint(focused) and focused not in highlighted:
            sections.append(("Focused", [focused]))
        selected = list(
            dict.fromkeys(
                NeuronComponent(int(c.neuron_id), None)
                for c in self.state.selected_components or ()
                if int(c.neuron_id) != int(component.neuron_id)
            )
        )[:10]
        own = NeuronComponent(int(component.neuron_id), None)
        sections.append(("Neurons", [own, *selected]))

        submenu = QMenu(f"Compare {label(component)} with", menu)
        submenu.setToolTipsVisible(True)
        for title, partners in sections:
            add_menu_section(submenu, title)
            for partner in partners:
                text = label(partner)
                if partner == own:
                    text += " (own neuron)"
                submenu.addAction(text).triggered.connect(
                    lambda checked=False, b=partner: self.open_pair_statistics(component, b)
                )
        menu.addMenu(submenu)

    def open_pair_statistics(self, first: NeuronComponent, second: NeuronComponent):
        from catan.gui.GUI_elements.fragments.pair_statistics_dialog import (
            PairStatisticsDialog,
        )

        dialog = PairStatisticsDialog(
            self.state, self.data, first, second, parent=self.native.window()
        )
        dialog.setWindowFlag(Qt.WindowType.Tool, True)
        dialog.show()

    def open_preset_check(self, component: NeuronComponent, preset_key: str):
        from catan.gui.GUI_elements.fragments.preset_check_dialog import (
            PresetCheckDialog,
        )

        dialog = PresetCheckDialog(
            self.state,
            self.data,
            component.neuron_id,
            component,
            preset_key,
            parent=self.native.window(),
        )
        dialog.setWindowFlag(Qt.WindowType.Tool, True)
        dialog.show()

    def _on_request_status_changed(self):

        self.request_display.set_request(self.state.current_request)
        self._position_request_display()

    def sync_request_display(self):

        if not self.displays_active_assignments:
            self.request_display.set_request(None)
            self._position_request_display()
            return

        request = self.state.current_request
        if request is not None and request.type == "neuron_merge":
            if request.assignments is not self.display_assignments:
                self.cancel_request()
                return
            request.refresh()

        self.request_display.set_request(request)

        # Every request change re-applies its highlight (also after clicks
        # elsewhere replaced it).
        if request is not None:
            self._highlight_request(request)

        self._position_request_display()

    def add_neuron_highlight_menu(self, menu, neuron_id):
        """Highlight all footprints of a neuron as source or target (replacing
        that role's current highlight), or remove them from the highlight."""
        submenu = QMenu(f"Highlight all footprints of neuron {neuron_id}", menu)
        submenu.setEnabled(self.state.current_request is None)
        menu.addMenu(submenu)

        for role in ("source", "target"):
            action = submenu.addAction(f"as {role}")
            action.setIcon(self._role_icon(role))
            action.triggered.connect(
                lambda checked=False, role=role: self.highlight_neuron_footprints(
                    neuron_id, role
                )
            )

        if any(
            c.neuron_id == neuron_id for c in (self.state.highlighted_components or ())
        ):
            submenu.addSeparator()
            action = submenu.addAction("remove from highlight")
            action.triggered.connect(
                lambda: self.highlight_neuron_footprints(neuron_id, None)
            )

    def highlight_neuron_footprints(self, neuron_id, role):
        """Highlight all footprints of a neuron as "source"/"target", replacing
        the current highlight of that role (plain highlights count as targets;
        the other role is kept), or remove them from the highlight (None)."""
        ids = self.data.assignments.ids
        footprints = [
            NeuronComponent(int(neuron_id), int(session))
            for session in np.flatnonzero(ids[neuron_id] >= 0)
        ]

        current_roles = self.state.highlight_roles

        def replaced(component):
            if component.neuron_id == neuron_id:
                return True
            if role is None:
                return False
            return current_roles.get(component, "target") == role

        components = [
            c for c in (self.state.highlighted_components or ()) if not replaced(c)
        ]
        roles = {c: r for c, r in current_roles.items() if c in components}

        if role is not None:
            components.extend(footprints)
            roles.update({c: role for c in footprints})

        self.state.update_highlighted_components(
            components or None, roles=roles, edited=True
        )

    def _role_icon(self, role):
        """Small swatch in the highlight colour of a source/target role."""
        if role not in ("source", "target"):
            return None

        rgba = self.styles.opts[f"highlighted_{role}"]["cmap"].map(np.array([0.5]))[0]
        pixmap = QPixmap(12, 12)
        pixmap.fill(QColor.fromRgbF(*(float(v) for v in rgba[:3]), 1.0))
        return QIcon(pixmap)

    def highlighted_footprints_of(self, component):
        """Sessions of the highlighted footprints of the component's neuron.

        With highlight roles (e.g. curator sources / targets), only those
        sharing the role of the component, if it is highlighted itself.
        """
        neuron = component.neuron_id
        highlighted = [
            c
            for c in (self.state.highlighted_components or ())
            if c.neuron_id == neuron and c.session_id is not None
        ]
        roles = self.state.highlight_roles
        role = roles.get(component)

        if role is not None:
            highlighted = [c for c in highlighted if roles.get(c) == role]

        return sorted({int(c.session_id) for c in highlighted})

    # Moving footprints into the selected (focused) neuron: the unions of the
    # moved footprints and the target within this distance (px).
    # Merge / move targets: neighbours within this radius (px; at least the
    # navigation bar's adjacency radius), "Load more" adds the step.
    MERGE_RADIUS_PX = 20.0
    MERGE_RADIUS_STEP_PX = 10.0
    MERGE_MAX_SELECTED = 10
    # Selected neurons are offered within this radius (px; "Load more"
    # widens it once the neighbour radius is larger).
    MERGE_SELECTED_RADIUS_PX = 50.0

    def add_move_footprints_menu(self, menu, component):
        """"Move footprints of neuron N": the scope (all / this and previous /
        this and following / highlighted), then the target neuron. Every
        scope is listed; one that does not apply is disabled with the
        reason as tooltip. Choosing a scope starts picking the target
        (start_move_pick)."""
        source_id = int(component.neuron_id)
        session_id = int(component.session_id)
        sessions = np.flatnonzero(self.display_assignments.ids[source_id] >= 0)
        highlighted = np.asarray(self.highlighted_footprints_of(component), dtype=int)
        role = self.state.highlight_roles.get(component)

        submenu = QMenu(f"Move footprints of neuron {source_id}", menu)
        submenu.setToolTipsVisible(True)
        submenu.setEnabled(self.state.current_request is None)
        menu.addMenu(submenu)

        tag = session_tag(session_id)
        earlier = sessions[sessions <= session_id]
        later = sessions[sessions >= session_id]
        single = "Only this footprint: use “Reassign this footprint”."
        described = f"Highlighted {role} footprints" if role else "Highlighted footprints"

        scopes = (
            (f"All footprints ({len(sessions)})", sessions, None),
            (
                f"This and previous (≤ {tag}, {len(earlier)})",
                earlier,
                None if len(earlier) > 1 else "No earlier footprint. " + single,
            ),
            (
                f"This and following (≥ {tag}, {len(later)})",
                later,
                None if len(later) > 1 else "No later footprint. " + single,
            ),
            (
                f"{described} ({len(highlighted)})",
                highlighted,
                None
                if len(highlighted)
                else f"No footprint of neuron {source_id} is highlighted "
                "(Ctrl + click it, or “Highlight all footprints of neuron "
                f"{source_id}”).",
            ),
        )

        for label, moved, reason in scopes:
            scope = submenu.addAction(f"{label}…")
            if moved is highlighted and role is not None:
                scope.setIcon(self._role_icon(role))

            if reason is not None:
                scope.setEnabled(False)
                scope.setToolTip(reason)
                continue

            scope.setToolTip("Then click the target neuron in the view or in the list.")
            scope.triggered.connect(
                lambda checked=False, moved=moved: self.start_move_pick(component, moved)
            )

    def add_replace_with_union_menu(self, menu, component):
        """Shapes from the union of the neuron's other footprints (e.g. odd
        shapes before seeded redetection); activity is kept."""
        neuron_id, session_id = int(component.neuron_id), int(component.session_id)
        included = [
            int(s)
            for s in np.flatnonzero(self.display_assignments.ids[neuron_id] >= 0)
            if self.data.is_included(NeuronComponent(neuron_id, int(s)))
        ]
        highlighted = self.highlighted_footprints_of(component)
        role = self.state.highlight_roles.get(component)

        submenu = QMenu("Replace with union shape", menu)
        submenu.setToolTipsVisible(True)
        submenu.setEnabled(self.state.current_request is None)
        submenu.menuAction().setToolTip(
            "Each footprint gets the union shape of the neuron's other "
            "footprints, placed where those footprints put it (interpolated "
            "between the nearest earlier and later one, as for seeds); its "
            "activity is kept."
        )
        menu.addMenu(submenu)

        this = submenu.addAction("This footprint")
        others = [s for s in included if s != session_id]
        this.setEnabled(bool(others))
        if not others:
            this.setToolTip("The neuron has no other included footprint.")
        this.triggered.connect(
            lambda checked=False: self._replace_with_union(neuron_id, [session_id])
        )

        described = f"Highlighted {role} footprints" if role else "Highlighted footprints"
        action = submenu.addAction(f"{described} ({len(highlighted)})")
        if role is not None:
            action.setIcon(self._role_icon(role))
        rest = [s for s in included if s not in highlighted]
        action.setEnabled(bool(highlighted) and bool(rest))
        if not highlighted:
            action.setToolTip(f"No footprint of neuron {neuron_id} is highlighted.")
        elif not rest:
            action.setToolTip("No other footprint remains to build the union from.")
        action.triggered.connect(
            lambda checked=False: self._replace_with_union(neuron_id, highlighted)
        )

    def add_cut_action(self, menu, component, enabled):
        """Cut the footprint between its two lobes (no other sessions
        needed): one part stays, the other becomes a new neuron."""
        from catan.core.footprint_lobes import lobe_count

        cut = menu.addAction("Cut at the neck")
        reason = None
        if not self.data.is_included(component):
            reason = "Excluded footprints cannot be cut."
        else:
            session = self.data.sessions[component.session_id]
            footprint_id = self.state.get_footprint_from_component(component)
            if footprint_id is None or lobe_count(
                session.footprints[:, int(footprint_id)], tuple(session.dims)
            ) < 2:
                reason = (
                    "The footprint has no two lobes to cut apart "
                    "(see the statistic Footprint lobes)."
                )
        cut.setEnabled(enabled and reason is None)
        cut.setToolTip(
            reason
            or "Cut between its two lobes: the part matching this neuron's "
            "other footprints stays, the other becomes a new neuron."
        )
        cut.triggered.connect(lambda: self._cut_footprint(component))

    def _cut_footprint(self, component):
        try:
            self.data.cut_footprint(component)
        except ValueError as exc:
            QMessageBox.warning(self.native, "Footprint not cut", str(exc))

    def _replace_with_union(self, neuron_id, sessions):
        try:
            self.data.replace_footprints_with_union(
                [NeuronComponent(int(neuron_id), int(s)) for s in sessions]
            )
        except ValueError as exc:
            QMessageBox.warning(self.native, "Shapes not replaced", str(exc))

    def start_neuron_merge(self, source_id, target_id, sessions=None):
        if self.state.current_request is not None:
            return

        request = request_handler.NeuronMergeRequest(
            self.display_assignments,
            source_id,
            target_id,
            sessions=sessions,
        )

        # Restored when the request is cancelled.
        request.previous_highlight = (
            list(self.state.highlighted_components or ()),
            dict(self.state.highlight_roles),
        )

        self.state.current_request = request
        self.plot_neurons(reset=True)

    def change_request_highlight(self, mode):
        request = self.state.current_request
        if request is None or request.type != "neuron_merge":
            return

        request.highlight_mode = mode
        self.state.notify_request_changed()

    def _highlight_request(self, request):
        """Targets orange, sources violet."""
        roles = request.roles
        self.state.update_highlighted_components(list(roles) or None, roles=roles)

    def start_merge(self, component: NeuronComponent):
        self.state.current_request = request_handler.RequestHandler("merge", component)
        self._highlight_request(self.state.current_request)

    def start_split(self, component: NeuronComponent):
        request = request_handler.RequestHandler("split", component)
        # A split has a single target: go straight to choosing the sources.
        request.advance_stage()
        self.state.current_request = request
        self._highlight_request(request)

    def cancel_request(self, restore=True):
        request = self.state.current_request
        previous = getattr(request, "previous_highlight", None) if restore else None

        self.state.current_request = None
        self._restore_highlight(previous)

    def advance_or_confirm_request(self):

        request = self.state.current_request

        if request is None:
            return

        if request.type == "neuron_merge":
            try:
                self.data.merge_neuron_request(request)
            except Exception as exc:
                self.sync_request_display()
                self.request_display.set_error(str(exc))
                return

            self.cancel_request(restore=False)
            return

        # ---------------------------------------------
        # Targets -> sources
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

        resolution = self._resolve_request_conflicts(request)
        if resolution is None:
            return
        included, occupied = resolution

        try:
            self.data.process_component_request(request, **occupied)
        except Exception as exc:
            # Nothing applied: the re-included endpoints are excluded again.
            self.data.restore_inclusion(included)
            self.request_display.set_error(str(exc))
            return

        self.state.current_request = None
        self.state.update_highlighted_components(None)

    def _resolve_request_conflicts(self, request):
        """Ask about conflicts of a split / merge request before applying:
        excluded endpoints (included again) and destinations that already
        hold a footprint (as the curator: exclude existing / keep as a
        separate neuron). Returns (inclusion before, see
        Data.inclusion_state; resolution keywords for prepare()) or None
        when cancelled."""
        from catan.gui.data.curation_actions import (
            OccupiedDestination,
            occupied_destinations,
        )

        sources = tuple(request.destination)
        targets = tuple(request.origin)
        ids = self.data.assignments.ids

        union_included = self.data.assignments.union.included
        excluded = [
            c
            for c in (*sources, *targets)
            if c.session_id is not None
            and ids[c.neuron_id, c.session_id] >= 0
            and not (self.data.is_included(c) and union_included[c.neuron_id])
        ]
        slots = occupied_destinations(self.data, request.type, sources, targets)

        if not excluded and not slots:
            return (), {}

        word = {"split": "split", "merge": "merge"}.get(request.type, "change")
        lines = []
        if excluded:
            lines.append("Excluded footprints in this " + word + ":")
            lines += [
                f"Neuron {c.neuron_id}, {session_label(self.data, c.session_id)}: "
                f"footprint {int(ids[c.neuron_id, c.session_id])}"
                for c in excluded
            ]
        if slots:
            if lines:
                lines.append("")
            lines.append(str(OccupiedDestination(slots, request.type)))

        info = []
        if excluded:
            info.append(
                "The excluded footprints are included again (and their "
                "neurons, as with Include footprint) before the " + word + "."
            )
        if slots:
            info.append(
                "Exclude: the existing footprint is excluded (kept as an "
                "excluded single-footprint neuron) and the result takes its "
                "place.\nKeep: the existing footprint moves to a separate, "
                "included neuron; nothing is excluded."
            )

        dialog = QMessageBox(self.native)
        dialog.setIcon(QMessageBox.Icon.Warning)
        dialog.setWindowTitle(f"Conflicts in this {word}")
        dialog.setText("\n".join(lines))
        dialog.setInformativeText("\n\n".join(info))

        buttons = {}
        prefix = "Include, " if excluded else ""
        if slots:
            buttons["exclude"] = dialog.addButton(
                prefix + "exclude existing && apply", QMessageBox.ButtonRole.AcceptRole
            )
            buttons["detach"] = dialog.addButton(
                prefix + "keep as separate neuron && apply",
                QMessageBox.ButtonRole.AcceptRole,
            )
        else:
            buttons["include"] = dialog.addButton(
                "Include && apply", QMessageBox.ButtonRole.AcceptRole
            )
        cancel_button = dialog.addButton(QMessageBox.StandardButton.Cancel)
        dialog.setDefaultButton(cancel_button)

        fit_message_box_buttons(dialog)
        dialog.exec()

        clicked = dialog.clickedButton()
        choice = next((key for key, button in buttons.items() if button is clicked), None)
        if choice is None:
            return None

        previous = self.data.inclusion_state(excluded)
        for component in excluded:
            self.data.reinclude_component(component)

        occupied = {}
        if choice == "exclude":
            occupied = {"exclude_occupied": True}
        elif choice == "detach":
            occupied = {"detach_occupied": True}
        return previous, occupied

    def _on_request_interaction(self, event):

        request = self.state.current_request

        if request is None:
            return

        component = self.find_closest_component(event.pos)

        if component is None or self.is_inferred_component(component):
            return

        if request.type == "neuron_merge":
            # Edit the set of moved footprints.
            error = request.toggle(component)

            if error is not None:
                self.request_display.set_error(error)
                return

            self.state.notify_request_changed()
            return

        current_components = (
            request.origin if request.stage == "origin" else request.destination
        )

        if component in current_components:

            request.remove_component(component)

            self._highlight_request(request)
            self.state.notify_request_changed()
            return

        error = request.validation_error(component)

        if error is not None:
            self.request_display.set_error(error)
            return

        request.add_component(component)

        self._highlight_request(request)
        self.state.notify_request_changed()

    def toggle_included(self, component: NeuronComponent | int):
        # Implement the logic for handling the removal action
        if self.data.is_included(component):
            self.data.exclude_component(component, keep_alone_in_place=True)
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
            f"Loading {which} data for {session_label(session)}",
            lambda: self.data.toggle_session_data(session_id, which),
            # finished=self.refresh_rows,
        )

    def change_neuron_assignment_dialog(
        self,
        component: NeuronComponent,
        new_neuron: Optional[int] = None,
    ):
        occupied = None

        if new_neuron is not None:
            existing = int(self.data.assignments.ids[new_neuron, component.session_id])

            if existing >= 0:
                occupied = self._ask_occupied_resolution(component, new_neuron, existing)
                if occupied is None:
                    return

        # The resulting data change replots (Controller._on_data_changed).
        try:
            self.data.reassign_footprint(component, new_neuron, occupied=occupied)
        except ValueError as exc:
            QMessageBox.warning(self.native, "Reassignment not applied", str(exc))

    def _ask_occupied_resolution(self, component, new_neuron, existing):
        """"exclude", "detach" or None (cancelled)."""
        session = session_label(self.data, component.session_id)

        dialog = QMessageBox(self.native)
        dialog.setIcon(QMessageBox.Icon.Warning)
        dialog.setWindowTitle("Neuron already has a footprint")
        dialog.setText(
            f"Neuron {new_neuron} already has footprint {existing} in {session}."
        )
        dialog.setInformativeText(
            "Exclude: the existing footprint is excluded (kept as an excluded "
            "single-footprint neuron) and this footprint takes its place.\n"
            "Keep: the existing footprint moves to a separate, included "
            "neuron; nothing is excluded."
        )

        exclude_button = dialog.addButton(
            "Exclude existing && reassign", QMessageBox.ButtonRole.AcceptRole
        )
        detach_button = dialog.addButton(
            "Keep as separate neuron && reassign", QMessageBox.ButtonRole.AcceptRole
        )
        cancel_button = dialog.addButton(QMessageBox.StandardButton.Cancel)
        dialog.setDefaultButton(cancel_button)

        fit_message_box_buttons(dialog)
        dialog.exec()

        clicked = dialog.clickedButton()
        if clicked is exclude_button:
            return "exclude"
        if clicked is detach_button:
            return "detach"
        return None

    def set_focused_footprint(self, component):
        mapped = self.component_to_active(component)

        if mapped is not None:
            self.state.focused_component = mapped

    def change_review_status(self, neuron_id: int, status: ReviewStatus):
        self.data.change_review_status(neuron_id, status)


class Controller(BasePlot.CanvasController):

    request_status_changed = Signal()

    def _restore_rotation_preference(self):
        """"Rotate on focus" and its turn duration are user preferences
        (QSettings), shared by all footprint panels and kept across runs."""
        parameters = self.controls["parameter"]
        settings = self.state.settings
        if settings is not None:
            parameters.checkbox_rotate_on_focus.setChecked(
                settings.value(ROTATE_ON_FOCUS_KEY, False, type=bool)
            )
            parameters.rotation_duration_spin.setValue(
                settings.value(ROTATION_DURATION_KEY, 4.0, type=float)
            )

        def store(*_):
            if self.state.settings is None:
                return
            self.state.settings.setValue(
                ROTATE_ON_FOCUS_KEY, parameters.checkbox_rotate_on_focus.isChecked()
            )
            self.state.settings.setValue(
                ROTATION_DURATION_KEY, parameters.rotation_duration_spin.value()
            )

        parameters.checkbox_rotate_on_focus.toggled.connect(store)
        parameters.rotation_duration_spin.valueChanged.connect(store)

    def _update_hovered_neuron(self, hovered):
        """Hovering a footprint flashes its neuron's other footprints
        ("Show neuron on hover"); restarted only when the hover changes."""
        if isinstance(hovered, NeuronComponent):
            hovered = [hovered]
        hovered = [c for c in hovered or () if c.session_id is not None]
        key = tuple(sorted((int(c.neuron_id), int(c.session_id)) for c in hovered))
        if key == getattr(self, "_hovered_neuron_key", None):
            return
        self._hovered_neuron_key = key

        parameters = self.controls.get("parameter")
        enabled = parameters is None or parameters.checkbox_hover_neuron.isChecked()
        siblings = []
        if enabled:
            hovered_keys = set(key)
            neurons = {neuron for neuron, _ in key}
            siblings = [
                NeuronComponent(*data_key)
                for data_key in self.canvas.plotting["data"]
                if isinstance(data_key, tuple)
                and data_key[0] in neurons
                and data_key not in hovered_keys
            ]
        self.canvas.show_hovered_neuron(siblings)

    def _restore_coordinates_preference(self):
        """"Coordinate grid" (user preference, QSettings)."""
        parameters = self.controls["parameter"]
        settings = self.state.settings
        if settings is not None:
            parameters.checkbox_coordinates.setChecked(
                settings.value(SHOW_COORDINATES_KEY, True, type=bool)
            )
        self.canvas.coordinates.set_enabled(parameters.checkbox_coordinates.isChecked())

        def apply(on):
            self.canvas.coordinates.set_enabled(on)
            if self.state.settings is not None:
                self.state.settings.setValue(SHOW_COORDINATES_KEY, bool(on))

        parameters.show_coordinates_changed.connect(apply)

        # "Show neuron on hover" (user preference).
        if settings is not None:
            parameters.checkbox_hover_neuron.setChecked(
                settings.value(HOVER_NEURON_KEY, True, type=bool)
            )

        def store_hover(on):
            if self.state.settings is not None:
                self.state.settings.setValue(HOVER_NEURON_KEY, bool(on))
            if not on:
                self._hovered_neuron_key = None
                self.canvas.show_hovered_neuron(None)

        parameters.checkbox_hover_neuron.toggled.connect(store_hover)

    def build_controls(self):
        super().build_controls()

        self.connect_signal(self.state.adjacency_radius_changed, self.replot_neurons)
        self.controls["parameter"] = FootprintParametersController(self.section)
        self.canvas.attach_parameter_overlay()
        self._restore_rotation_preference()
        self._restore_coordinates_preference()
        self.controls["parameter"].rotate_on_focus_changed.connect(
            lambda on: self.canvas.rotate_on_focus(force=True)
            if on
            else self.canvas.stop_rotation()
        )

        self.assignments_selector = QComboBox()
        self.assignments_selector.setMinimumWidth(160)
        self.assignments_selector.setToolTip(
            "Assignments displayed by this footprint panel."
        )
        self.section.x_options_layout.addWidget(self.assignments_selector)

        self.assignments_selector.currentIndexChanged.connect(
            self._on_display_assignments_changed
        )
        self._refresh_assignments_selector()

        self.controls["parameter"].data_parameter_changed.connect(
            lambda: self.replot_neurons()
        )
        self.controls["parameter"].display_parameter_changed.connect(
            lambda: self.update_neuron_selection()
        )

        self.controls["session_filter"] = SessionFilterControl(self.section)
        self.canvas.native.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
        self.canvas.native.installEventFilter(self.controls["session_filter"])

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

        self.connect_signal(
            self.state.alignment_draft_changed,
            self._schedule_alignment_preview,
        )

        self.connect_signal(
            self.state.request_status_changed,
            self._on_request_status_changed,
            controls=True,
        )
        self.connect_signal(
            self.canvas.request_display.cancel_requested,
            self.canvas.cancel_request,
            controls=True,
        )
        self.connect_signal(
            self.canvas.request_display.confirm_requested,
            self.canvas.advance_or_confirm_request,
            controls=True,
        )
        self.connect_signal(
            self.canvas.request_display.highlight_changed,
            self.canvas.change_request_highlight,
            controls=True,
        )
        self._on_request_status_changed()

        bar = self.canvas.pick_bar
        self.connect_signal(bar.chosen, self.canvas.choose_pick_target, controls=True)
        self.connect_signal(bar.hovered, self.canvas.hover_pick_target, controls=True)
        self.connect_signal(bar.more_requested, self.canvas.widen_pick, controls=True)
        self.connect_signal(bar.cancel_requested, self.canvas.cancel_pick, controls=True)
        self.connect_signal(
            bar.pick_changed,
            lambda: self._queue_selection_update("selection"),
            controls=True,
        )

    def deactivate(self):
        canvas = getattr(self, "canvas", None)
        if not self._deactivated and canvas is not None:
            canvas.cancel_pick()
        super().deactivate()

    def _on_data_changed(self, event: DataChange):

        self._refresh_assignments_selector()

        # The candidates (and what moves) may be gone: start again.
        if self.canvas.picking and event.has(
            *SESSION_STRUCTURE_CHANGES,
            C.ASSIGNMENT_SET,
            C.ASSIGNMENT_MAPPING,
            C.INCLUSION,
        ):
            self.canvas.cancel_pick()

        if event.has(*SESSION_STRUCTURE_CHANGES, C.ASSIGNMENT_SET):
            self.initialize_display()

        elif event.has(
            C.ASSIGNMENT_MAPPING,
            C.UNION_GEOMETRY,
            C.FOOTPRINT_GEOMETRY,
            C.INCLUSION,
            C.SESSION_ACTIVITY,
            C.SESSION_METADATA,
        ) or event.has_availability("spatial"):
            self.replot_neurons()

        if event.has(
            C.REVIEW_STATUS,
            C.ASSIGNMENT_SET,
            C.ASSIGNMENT_MAPPING,
        ):
            self.canvas.update_review_status_tag()

        if (
            self.state.current_request is not None
            and self.state.current_request.type == "neuron_merge"
        ):
            self.canvas.sync_request_display()

    def initialize_display(self):

        self._refresh_assignments_selector()

        self._setup_session_filter()
        super().initialize_display()
        self.canvas.update_review_status_tag()

    def _refresh_assignments_selector(self):
        selector = self.assignments_selector
        chosen = self.canvas._assignment_name

        if chosen is not None and chosen not in self.data.available_assignments:
            self.canvas._assignment_name = None
            chosen = None

        blocked = selector.blockSignals(True)
        try:
            selector.clear()
            selector.addItem("Follow active assignments", None)

            names = self.data.available_assignments
            for name in names:
                selector.addItem(f"Assignments: {assignment_label(name)}", name)

            if chosen is not None and chosen not in names:
                selector.addItem(
                    f"Assignments: {assignment_label(chosen)} — unavailable", chosen
                )

            selector.setCurrentIndex(0 if chosen is None else selector.findData(chosen))
        finally:
            selector.blockSignals(blocked)

        # Only a choice with several assignment sets (or a chosen set that
        # is no longer available).
        loaded = [name for name in names if self.data._assignments.get(name) is not None]
        selector.setVisible(len(loaded) > 1 or (chosen is not None and chosen not in loaded))

    def _on_display_assignments_changed(self, index):
        name = self.assignments_selector.itemData(index)

        if name == self.canvas._assignment_name:
            return

        self.canvas.cancel_pick()
        self.canvas._assignment_name = name
        self.config["assignments_name"] = name
        self.canvas.camera_set = False

        self.canvas.sync_request_display()
        self.replot_neurons()
        self.canvas.update_review_status_tag()

    def _on_session_changed(self):
        self._setup_session_filter()

        super()._on_session_changed()

    def _on_session_only_changed(self):
        # self.controls["parameter"]._on_session_only_changed()
        self.update_neuron_selection()

    def _on_session_filter_toggled(self, active: bool):
        self.controls["session_filter"].setVisible(active)
        self._setup_session_filter()
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

    def _on_highlight_changed(self):
        # Highlighted neurons are always plotted (Display.plot_neurons):
        # re-plot if one is not drawn yet, otherwise restyle only.
        canvas = self.canvas
        drawn = canvas.plotting["visuals"]
        highlighted = {
            int(component.neuron_id)
            for component in canvas.components_from_active(
                list(self.state.highlighted_components or ())
            )
        }
        request = self.state.current_request
        if request is not None:
            highlighted |= {
                int(component.neuron_id)
                for component in canvas.components_from_active(request.components)
            }

        replot = any(neuron not in drawn for neuron in highlighted) or bool(
            canvas.plotting.get("required_only", set()) - highlighted
        )
        self._queue_selection_update("selection" if replot else "style")

    def update_styles(self):
        canvas = self.canvas

        hovered = canvas.components_from_active(self.state.hovered_components) or None
        canvas.update_style(hovered, "hovered")
        self._update_hovered_neuron(hovered)
        canvas.update_style(
            canvas.components_from_active(
                self.state.selected_components,
            )
            or None,
            "selected",
        )
        canvas.update_style(
            canvas.components_from_active(
                self.state.focused_component,
            )
            or None,
            "focused",
        )

        highlighted = self.state.highlighted_components or ()
        roles = self.state.highlight_roles

        for role, style in (
            (None, "highlighted"),
            ("source", "highlighted_source"),
            ("target", "highlighted_target"),
        ):
            components = [
                component for component in highlighted if roles.get(component) == role
            ]
            canvas.update_style(
                canvas.components_from_active(components) or None,
                style,
            )

    def replot_neurons(self):
        self.canvas.plot_neurons(reset=True)
        self.update_styles()

    def _schedule_alignment_preview(self):
        # Coalesce the draft changes of one interaction (dragging, spin boxes).
        if not getattr(self, "_alignment_preview_pending", False):
            self._alignment_preview_pending = True
            QTimer.singleShot(0, self._update_alignment_preview)

    def _update_alignment_preview(self):
        self._alignment_preview_pending = False

        if self._deactivated:
            return

        preview = self.canvas.alignment_preview()
        shown = self.canvas.plotting.get("alignment_preview")

        unchanged = (None if preview is None else preview[1]) == (
            None if shown is None else shown[1]
        )

        if not unchanged:
            self.update_neuron_selection()

    def _on_request_status_changed(self):
        self.canvas.sync_request_display()


class SessionFilterControl(QWidget):
    valueChanged = Signal(int)

    def __init__(self, parent=None):
        super().__init__(parent)

        self.data = parent.data
        self._session_count = 0

        self.setFixedWidth(50)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(2, 4, 2, 4)
        layout.setSpacing(4)

        self.title = QLabel("Session")
        self.title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(self.title)

        self.value_label = QLabel("")
        self.value_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(self.value_label)

        self.up_button = QToolButton()
        self.up_button.setArrowType(Qt.ArrowType.UpArrow)
        self.up_button.setToolTip("Next session — Up")
        self.up_button.setAutoRepeat(True)
        self.up_button.clicked.connect(lambda checked=False: self.step(1))
        layout.addWidget(
            self.up_button,
            alignment=Qt.AlignmentFlag.AlignHCenter,
        )

        self.slider = QSlider(Qt.Orientation.Vertical)
        self.slider.setRange(0, 0)
        self.slider.setSingleStep(1)
        self.slider.setPageStep(1)
        self.slider.setInvertedAppearance(False)
        self.slider.setInvertedControls(False)
        layout.addWidget(self.slider, stretch=1)

        self.down_button = QToolButton()
        self.down_button.setArrowType(Qt.ArrowType.DownArrow)
        self.down_button.setToolTip("Previous session — Down")
        self.down_button.setAutoRepeat(True)
        self.down_button.clicked.connect(lambda checked=False: self.step(-1))
        layout.addWidget(
            self.down_button,
            alignment=Qt.AlignmentFlag.AlignHCenter,
        )

        self.zero_label = QLabel("0")
        self.zero_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(self.zero_label)

        self.slider.valueChanged.connect(self._on_value_changed)

        for widget in (
            self.slider,
            self.up_button,
            self.down_button,
        ):
            widget.installEventFilter(self)

        self._refresh_labels()

    def step(self, direction):
        if not self.isVisible() or self._session_count <= 0:
            return

        value = self.slider.value() + int(direction)
        value = max(
            self.slider.minimum(),
            min(self.slider.maximum(), value),
        )
        self.slider.setValue(value)

    def eventFilter(self, watched, event):
        if (
            self.isVisible()
            and self._session_count > 0
            and QApplication.activePopupWidget() is None
            and event.type()
            in (
                QEvent.Type.ShortcutOverride,
                QEvent.Type.KeyPress,
            )
            and event.modifiers() == Qt.KeyboardModifier.NoModifier
            and event.key()
            in (
                Qt.Key.Key_Up,
                Qt.Key.Key_Down,
            )
        ):
            event.accept()

            if event.type() == QEvent.Type.KeyPress:
                self.step(1 if event.key() == Qt.Key.Key_Up else -1)

            return True

        return super().eventFilter(watched, event)

    def _refresh_labels(self):
        value = self.slider.value()
        available = self._session_count > 0

        self.slider.setEnabled(available)
        self.up_button.setEnabled(available and value < self.slider.maximum())
        self.down_button.setEnabled(available and value > self.slider.minimum())

        if not available:
            self.value_label.setText("—")
            self.value_label.setToolTip("")
            return

        self.value_label.setText(session_tag(value))
        self.value_label.setToolTip(
            f"{session_label(self.data, value)} — {self._session_count} sessions"
        )

    def _on_value_changed(self, value):
        self._refresh_labels()

        if self._session_count > 0:
            self.valueChanged.emit(value)

    def set_sessions(self, n_sessions, current_session=None):
        previous = self.slider.value()
        self._session_count = max(
            0,
            min(int(n_sessions), len(self.data.sessions)),
        )
        maximum = max(0, self._session_count - 1)

        value = previous if current_session is None else int(current_session)
        value = max(0, min(maximum, value))

        with QSignalBlocker(self.slider):
            self.slider.setRange(0, maximum)
            self.slider.setValue(value)

        self._refresh_labels()

        if self._session_count > 0 and value != previous:
            self.valueChanged.emit(value)


class RequestDisplay(QFrame):

    # The box fits its text; a long error (e.g. many conflicting slots)
    # scrolls beyond this height.
    ERROR_MAX_HEIGHT = 160

    cancel_requested = Signal()
    confirm_requested = Signal()

    highlight_changed = Signal(str)

    def __init__(self, parent, data):
        super().__init__(parent)

        # Session labels in the request summary.
        self.data = data

        self._position_timer = QTimer(self)
        self._position_timer.setSingleShot(True)
        self._position_timer.setInterval(0)
        self._position_timer.timeout.connect(self.reposition)
        parent.installEventFilter(self)

        self.setMaximumWidth(max(1, parent.width() // 2))

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
        self.origin_label.setWordWrap(True)
        root_layout.addWidget(self.origin_label)

        self.destination_label = QLabel("")
        self.destination_label.setObjectName("requestStatus")
        self.destination_label.setAlignment(Qt.AlignmentFlag.AlignLeft)
        self.destination_label.setWordWrap(True)
        root_layout.addWidget(self.destination_label)

        self.error_label = QLabel("")
        self.error_label.setObjectName("requestError")
        self.error_label.setWordWrap(True)
        self.error_label.setAlignment(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignTop)
        self.error_scroll = QScrollArea(self)
        self.error_scroll.setObjectName("requestErrorScroll")
        self.error_scroll.setFrameShape(QFrame.Shape.NoFrame)
        self.error_scroll.setWidgetResizable(True)
        self.error_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.error_scroll.setWidget(self.error_label)
        self.error_scroll.viewport().setAutoFillBackground(False)
        self.error_scroll.hide()
        self.error_icon = QLabel(self)
        self.error_icon.setPixmap(
            self.style()
            .standardIcon(QStyle.StandardPixmap.SP_MessageBoxWarning)
            .pixmap(22, 22)
        )
        self.error_icon.setFixedSize(24, 24)
        self.error_icon.hide()

        error_layout = QHBoxLayout()
        error_layout.setSpacing(7)
        error_layout.addWidget(
            self.error_icon,
            0,
            Qt.AlignmentFlag.AlignTop,
        )
        error_layout.addWidget(self.error_scroll, 1)
        root_layout.addLayout(error_layout)

        self.highlight_selector = QComboBox(self)
        self.highlight_selector.hide()
        root_layout.addWidget(self.highlight_selector)

        # activated: choosing the shown mode again re-applies its highlight.
        self.highlight_selector.activated.connect(
            lambda _: self.highlight_changed.emit(self.highlight_selector.currentData())
        )

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

        # Enter = Continue / Confirm / Apply.
        self._enter_key = EnterKeyButton(self.confirm_button)
        self.confirm_button.setToolTip("Enter")

        # self.root_layout = root_layout
        self.setObjectName("RequestDisplayOverlay")

        self.setStyleSheet("""
            QFrame#RequestDisplayOverlay {
                background-color: rgba(30, 34, 40, 245);
                border: 1px solid #6b7480;
                border-radius: 7px;
            }

            QFrame#RequestDisplayOverlay[needsAttention="true"] {
                border: 2px solid #c87920;
            }

            QScrollArea#requestErrorScroll,
            QScrollArea#requestErrorScroll > QWidget > QWidget,
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
                color: #ffd28a;
                font-size: 13px;
                font-weight: 600;
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

    def event(self, event):
        result = super().event(event)

        if event.type() in (
            QEvent.Type.LayoutRequest,
            QEvent.Type.Resize,
            QEvent.Type.Show,
        ) and hasattr(self, "_position_timer"):
            self._position_timer.start()

        return result

    def eventFilter(self, watched, event):
        if watched is self.parentWidget() and event.type() == QEvent.Type.Resize:
            self._position_timer.start()

        return super().eventFilter(watched, event)

    def reposition(self):
        parent = self.parentWidget()
        if parent is None or not self.isVisible():
            return

        margin = 8
        available_width = max(1, parent.width() - 2 * margin)
        max_width = min(available_width, max(280, parent.width() // 2))
        self.setMaximumWidth(max_width)

        # Width from the unwrapped text, height for that width (adjustSize
        # gets wrapped labels wrong and cut the error off).
        layout = self.layout()
        margins = layout.contentsMargins()
        error_extra = self.error_icon.width() + 7 if self.error_scroll.isVisible() else 0
        widths = [layout.sizeHint().width() - margins.left() - margins.right()]
        for label in (self.title, self.origin_label, self.destination_label, self.error_label):
            if label is self.error_label and not self.error_scroll.isVisible():
                continue
            metrics = label.fontMetrics()
            longest = max(
                (metrics.horizontalAdvance(line) for line in label.text().split("\n")),
                default=0,
            )
            widths.append(longest + 8 + (error_extra if label is self.error_label else 0))
        width = min(max_width, max(widths) + margins.left() + margins.right())

        if self.error_scroll.isVisible():
            label_width = width - margins.left() - margins.right() - error_extra
            text_height = self.error_label.heightForWidth(label_width)
            if text_height > self.ERROR_MAX_HEIGHT:
                scrollbar = self.error_scroll.verticalScrollBar().sizeHint().width()
                text_height = self.error_label.heightForWidth(label_width - scrollbar)
            self.error_scroll.setFixedHeight(min(text_height, self.ERROR_MAX_HEIGHT))

        layout.activate()
        height = self.heightForWidth(width) if self.hasHeightForWidth() else -1
        if height <= 0:
            height = self.sizeHint().height()
        height = min(height, max(1, parent.height() - 2 * margin))
        self.resize(width, height)
        self.move(
            margin,
            max(margin, parent.height() - self.height() - margin),
        )
        self.raise_()

    def set_request(
        self,
        request_handler: request_handler.RequestHandler | None,
    ):

        self.highlight_selector.hide()
        if request_handler is None:
            self.clear_request()
            return

        if request_handler.type == "neuron_merge":
            request = request_handler

            whole = request.target_id is not None and request.moves_whole_neuron
            self.title.setText("Merge neurons" if whole else "Move footprints")
            sessions = session_labels(self.data, [c.session_id for c in request.origin], short=True)
            self.origin_label.setText(
                f"Move {len(request.origin)} footprints of neuron "
                f"{request.source_id}"
                + (f" ({sessions})" if sessions else "")
                + "\nClick its footprints to add or remove them."
            )
            self.destination_label.setText(
                "Into a new neuron"
                if request.target_id is None
                else f"Into neuron {request.target_id}"
            )

            self.highlight_selector.blockSignals(True)
            try:
                self.highlight_selector.clear()
                self.highlight_selector.addItem("Highlight footprints", "footprints")
                if request.conflicts:
                    self.highlight_selector.addItem("Highlight conflicts", "conflicts")
                self.highlight_selector.setCurrentIndex(
                    max(
                        0,
                        self.highlight_selector.findData(request.highlight_mode),
                    )
                )
            finally:
                self.highlight_selector.blockSignals(False)

            self.highlight_selector.show()
            self.confirm_button.setText("Apply")
            self.confirm_button.setEnabled(request.is_complete)

            if request.error:
                self.set_error(request.error)
            else:
                self.clear_error()

            self.show()
            self._position_timer.start()
            return

        self.setVisible(True)

        stage = (
            "click the target footprints"
            if request_handler.stage == "origin"
            else "click the source footprints"
        )
        self.title.setText(f"{request_handler.type.capitalize()}: {stage}")
        self.origin_label.setText(request_handler.origin_status_text())
        self.destination_label.setText(request_handler.destination_status_text())

        if request_handler.stage == "origin":
            self.confirm_button.setText("Continue to source")

        else:
            self.confirm_button.setText(f"Confirm {request_handler.type}")

        self.confirm_button.setEnabled(request_handler.stage_complete)

        self.clear_error()

    def _set_attention(self, active):
        if self.property("needsAttention") == active:
            return

        self.setProperty("needsAttention", active)
        self.style().unpolish(self)
        self.style().polish(self)
        self.update()

    def set_error(self, message: str):
        self.error_label.setText(message)
        self.error_scroll.verticalScrollBar().setValue(0)
        self.error_scroll.show()
        self.error_icon.show()
        self._set_attention(True)
        self._position_timer.start()

    def clear_error(self):
        self.error_label.clear()
        self.error_scroll.hide()
        self.error_icon.hide()
        self._set_attention(False)
        self._position_timer.start()

    def clear_request(self):

        self.setVisible(False)

        self.title.clear()
        self.origin_label.clear()
        self.destination_label.clear()

        self.confirm_button.setEnabled(False)

        self.clear_error()


class TargetPickerBar(QFrame):
    """Bar at the bottom of the footprints view while a target neuron is
    picked (TargetPick): what moves, the candidates with their match
    quality (hovering a row highlights the neuron in the view, clicking
    chooses it), Load more, Into a new neuron, Cancel. Non-modal: the view
    stays rotatable and a neuron can be clicked there instead."""

    MAX_WIDTH = 640
    # Share of the view's height the list may take.
    LIST_HEIGHT_SHARE = 0.35

    chosen = Signal(object)  # neuron ID; None: a new neuron
    hovered = Signal(object)  # neuron ID or None
    more_requested = Signal()
    cancel_requested = Signal()
    # Candidates or pick state changed: replot.
    pick_changed = Signal()

    def __init__(self, parent):
        super().__init__(parent)
        self.setObjectName("TargetPickerBar")
        self._rows = {}

        self._position_timer = QTimer(self)
        self._position_timer.setSingleShot(True)
        self._position_timer.setInterval(0)
        self._position_timer.timeout.connect(self.reposition)
        parent.installEventFilter(self)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(10, 8, 10, 8)
        layout.setSpacing(4)

        title_row = QHBoxLayout()
        self.title = QLabel("")
        self.title.setObjectName("pickTitle")
        self.title.setWordWrap(True)
        title_row.addWidget(self.title, 1)
        self.list_toggle = QToolButton(self)
        self.list_toggle.setObjectName("pickListToggle")
        self.list_toggle.setCheckable(True)
        self.list_toggle.setChecked(True)
        self.list_toggle.setToolTip("Show / hide the list (more room for the view)")
        self.list_toggle.toggled.connect(self._on_list_toggled)
        title_row.addWidget(self.list_toggle, 0, Qt.AlignmentFlag.AlignTop)
        layout.addLayout(title_row)

        self.summary = QLabel("")
        self.summary.setObjectName("pickSummary")
        self.summary.setTextFormat(Qt.TextFormat.RichText)
        self.summary.setWordWrap(True)
        layout.addWidget(self.summary)

        self.hint = QLabel(
            "Click the target neuron in the view or below · right-click or Esc cancels"
        )
        self.hint.setObjectName("pickHint")
        self.hint.setWordWrap(True)
        layout.addWidget(self.hint)

        self.list = QListWidget(self)
        self.list.setObjectName("pickList")
        self.list.setMouseTracking(True)
        self.list.setVerticalScrollMode(QListWidget.ScrollMode.ScrollPerPixel)
        self.list.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.list.itemEntered.connect(self._on_item_entered)
        self.list.itemClicked.connect(self._on_item_clicked)
        self.list.viewport().installEventFilter(self)
        layout.addWidget(self.list)

        self.note = QLabel("")
        self.note.setObjectName("pickHint")
        layout.addWidget(self.note)

        buttons = QHBoxLayout()
        self.more_button = QPushButton("Load more")
        self.more_button.clicked.connect(self.more_requested.emit)
        self.new_button = QPushButton("Into a new neuron")
        self.new_button.clicked.connect(lambda: self.chosen.emit(None))
        self.cancel_button = QPushButton("Cancel")
        self.cancel_button.setToolTip("Esc")
        self.cancel_button.clicked.connect(self.cancel_requested.emit)
        buttons.addWidget(self.more_button)
        buttons.addWidget(self.new_button)
        buttons.addStretch()
        buttons.addWidget(self.cancel_button)
        layout.addLayout(buttons)

        # Esc in the view (or the bar) cancels, only while picking.
        self._escape = QShortcut(QKeySequence(Qt.Key.Key_Escape), parent)
        self._escape.setContext(Qt.ShortcutContext.WidgetWithChildrenShortcut)
        self._escape.activated.connect(self.cancel_requested.emit)
        self._escape.setEnabled(False)

        self.setStyleSheet("""
            QFrame#TargetPickerBar {
                background-color: rgba(30, 34, 40, 240);
                border: 1px solid #c87920;
                border-radius: 7px;
            }
            QFrame#TargetPickerBar QLabel {
                background: transparent;
                border: none;
                color: #e8eaed;
            }
            QLabel#pickTitle {
                color: #ffffff;
                font-size: 13px;
                font-weight: 600;
            }
            QLabel#pickSummary { font-size: 11px; }
            QLabel#pickHint { color: #aab2bd; font-size: 11px; }
            QLabel#pickMessage { color: #ffd28a; font-size: 12px; font-weight: 600; }
            QListWidget#pickList {
                background: rgba(20, 23, 28, 200);
                border: 1px solid #474e57;
                border-radius: 4px;
                color: #e8eaed;
                outline: none;
            }
            QListWidget#pickList::item { border-bottom: 1px solid #343a42; }
            QListWidget#pickList::item:hover,
            QListWidget#pickList::item:selected {
                background: #3b4450;
            }
            QFrame#TargetPickerBar QPushButton, QToolButton#pickListToggle {
                color: #e8eaed;
                background-color: #3b414a;
                border: 1px solid #66707c;
                border-radius: 4px;
                padding: 4px 10px;
            }
            QFrame#TargetPickerBar QPushButton:hover, QToolButton#pickListToggle:hover {
                background-color: #4a525d;
                border-color: #8793a1;
            }
            QFrame#TargetPickerBar QPushButton:disabled {
                color: #777f89;
                border-color: #474e57;
            }
        """)
        self._on_list_toggled(True)
        self.hide()

    def show_pick(self, pick):
        self.title.setText(pick.title)
        self.summary.setText("<br>".join(pick.summary))
        self.summary.setVisible(bool(pick.summary))
        self.hint.setObjectName("pickHint")
        self.hint.setText(
            "Click the target neuron in the view or below · right-click or Esc cancels"
        )
        self.hint.setStyleSheet("")
        self.note.setText(pick.note)
        self.note.setVisible(bool(pick.note))
        self.more_button.setText(pick.more_text)
        self.new_button.setEnabled(pick.allow_new)
        self.new_button.setToolTip(pick.new_tooltip)

        scroll = self.list.verticalScrollBar().value()
        self.list.clear()
        self._rows = {}
        for title, neurons in pick.sections:
            header = QListWidgetItem(title.upper())
            header.setFlags(Qt.ItemFlag.NoItemFlags)
            font = header.font()
            font.setPointSizeF(max(6.0, font.pointSizeF() - 1.5))
            font.setBold(True)
            header.setFont(font)
            header.setForeground(QColor("#8b939e"))
            self.list.addItem(header)
            if not neurons:
                empty = QListWidgetItem("    no other neuron nearby")
                empty.setFlags(Qt.ItemFlag.NoItemFlags)
                empty.setForeground(QColor("#8b939e"))
                self.list.addItem(empty)
            for neuron in neurons:
                label, lines = pick.rows[neuron]
                self._add_row(neuron, label, lines)
        self.list.verticalScrollBar().setValue(scroll)

        self._escape.setEnabled(True)
        self.show()
        self.raise_()
        self._position_timer.start()

    def _add_row(self, neuron, label, lines):
        row = QWidget()
        row.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents)
        row_layout = QVBoxLayout(row)
        row_layout.setContentsMargins(8, 3, 8, 3)
        row_layout.setSpacing(0)
        text = QLabel(f"<b>Into {label}</b>")
        text.setTextFormat(Qt.TextFormat.RichText)
        row_layout.addWidget(text)
        stats = QLabel("&nbsp;&nbsp;&nbsp;".join(lines))
        stats.setTextFormat(Qt.TextFormat.RichText)
        stats.setWordWrap(True)
        stats.setStyleSheet("font-size: 11px;")
        row_layout.addWidget(stats)

        item = QListWidgetItem()
        item.setData(Qt.ItemDataRole.UserRole, int(neuron))
        item.setSizeHint(row.sizeHint())
        self.list.addItem(item)
        self.list.setItemWidget(item, row)
        self._rows[int(neuron)] = item

    def clear_pick(self):
        self._escape.setEnabled(False)
        self.list.clear()
        self._rows = {}
        self.hide()

    def set_message(self, text):
        self.hint.setText(text)
        self.hint.setStyleSheet("color: #ffd28a; font-weight: 600;")

    def mark(self, neuron):
        """The row of the neuron hovered in the view."""
        item = self._rows.get(neuron)
        if item is None:
            self.list.clearSelection()
            return
        self.list.setCurrentItem(item)
        self.list.scrollToItem(item)

    def _neuron(self, item):
        value = None if item is None else item.data(Qt.ItemDataRole.UserRole)
        return int(value) if value is not None else None

    def _on_item_entered(self, item):
        self.hovered.emit(self._neuron(item))

    def _on_item_clicked(self, item):
        neuron = self._neuron(item)
        if neuron is not None:
            self.chosen.emit(neuron)

    def _on_list_toggled(self, shown):
        self.list.setVisible(shown)
        self.list_toggle.setText("Hide list" if shown else "Show list")
        self._position_timer.start()

    def eventFilter(self, watched, event):
        if watched is self.parentWidget() and event.type() == QEvent.Type.Resize:
            self._position_timer.start()
        elif watched is self.list.viewport() and event.type() == QEvent.Type.Leave:
            self.hovered.emit(None)
        return super().eventFilter(watched, event)

    def reposition(self):
        parent = self.parentWidget()
        if parent is None or not self.isVisible():
            return

        margin = 8
        width = max(1, min(self.MAX_WIDTH, parent.width() - 2 * margin))

        # The list as tall as its rows, at most a share of the view.
        rows = sum(
            self.list.sizeHintForRow(i) for i in range(self.list.count())
        ) + 2 * self.list.frameWidth()
        self.list.setFixedHeight(
            max(40, min(rows, int(parent.height() * self.LIST_HEIGHT_SHARE)))
        )

        layout = self.layout()
        layout.activate()
        height = self.heightForWidth(width) if self.hasHeightForWidth() else -1
        if height <= 0:
            height = self.sizeHint().height()
        height = min(height, max(1, parent.height() - 2 * margin))
        self.resize(width, height)
        self.move(margin, max(margin, parent.height() - height - margin))
        self.raise_()


class FootprintParametersController(ControlPanel.ControlPanel):
    """Gear panel of the footprints view, in three groups: which neurons,
    which sessions, how they are drawn. The session of "One session only"
    is the session slider next to the panel."""

    data_parameter_changed = Signal()

    session_only_changed = Signal(bool)
    reset_camera_requested = Signal()
    rotate_on_focus_changed = Signal(bool)
    show_coordinates_changed = Signal(bool)

    INDENT = 18  # px, options that depend on the one above

    def __init__(self, parent):
        super().__init__(parent)

        ## === neurons ===
        self._add_section("Neurons", first=True)
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

        self.checkbox_show_excluded = QCheckBox("Show excluded")
        self.checkbox_show_excluded.setChecked(False)
        self.checkbox_show_excluded.setToolTip(
            "Also draw excluded neurons and footprints."
        )
        self.checkbox_show_excluded.toggled.connect(
            lambda: self.display_parameter_changed.emit()
        )
        self.form.addRow(self.checkbox_show_excluded)

        ## === sessions ===
        self._add_section("Sessions")
        sessions_widget = QWidget()
        sessions_layout = QHBoxLayout(sessions_widget)
        sessions_layout.setContentsMargins(0, 0, 0, 0)
        sessions_layout.setSpacing(0)
        self.sessions_group = QButtonGroup(self)
        self.sessions_group.setExclusive(True)
        for text, position, tooltip in (
            ("All", "left", "Stack all (active) sessions."),
            (
                "With focused",
                "right",
                "Stack only sessions in which the focused neuron has a "
                "footprint (and the one-session-only session), evenly spaced.",
            ),
        ):
            button = QToolButton()
            button.setText(text)
            button.setCheckable(True)
            button.setToolTip(tooltip)
            button.setObjectName(f"scope{position.capitalize()}")
            button.setMinimumWidth(70)
            self.sessions_group.addButton(button)
            sessions_layout.addWidget(button)
        self.sessions_group.buttons()[0].setChecked(True)
        # "With focused" = collapse the sessions without the focused neuron.
        self.collapse_sessions_button = self.sessions_group.buttons()[1]
        self.collapse_sessions_button.toggled.connect(
            lambda: self.data_parameter_changed.emit()
        )
        self.form.addRow("Show", sessions_widget)

        self.checkbox_session_only = QCheckBox("One session only")
        self.checkbox_session_only.setToolTip(
            "Only footprints of the session chosen with the session slider "
            "next to the panel."
        )
        self.checkbox_session_only.setChecked(False)
        self.checkbox_session_only.toggled.connect(self._on_session_only_changed)
        self.form.addRow(self.checkbox_session_only)

        self.checkbox_filter_focused = QCheckBox("Also for the focused neuron")
        self.checkbox_filter_focused.setToolTip(
            "Off: the focused neuron keeps all its footprints."
        )
        self.checkbox_filter_focused.setChecked(False)
        self.checkbox_filter_focused.setEnabled(False)
        self.checkbox_filter_focused.toggled.connect(
            lambda: self.display_parameter_changed.emit()
        )
        self.form.addRow(self._indented(self.checkbox_filter_focused))

        self.checkbox_show_inferred = QCheckBox("Fill gaps with inferred (cyan)")
        self.checkbox_show_inferred.setEnabled(False)
        self.checkbox_show_inferred.setToolTip(
            "For neurons missing from the session, show "
            "the union footprint at its interpolated position.\n"
            "Cyan footprints are inferred, not detected components."
        )
        self.checkbox_show_inferred.toggled.connect(
            lambda: self.data_parameter_changed.emit()
        )
        self.form.addRow(self._indented(self.checkbox_show_inferred))

        ## === view ===
        self._add_section("View")
        initial_threshold = 0.1
        self.threshold_spin = QDoubleSpinBox()
        self.threshold_spin.setDecimals(3)
        self.threshold_spin.setRange(0.0, 1.0)
        self.threshold_spin.setSingleStep(0.01)
        self.threshold_spin.setValue(initial_threshold)
        self.threshold_spin.setToolTip(
            "Pixels below this fraction of a footprint's maximum are not drawn."
        )
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
        self.z_stretch_spin.setToolTip("Spacing of the stacked sessions.")
        self.z_stretch_spin.valueChanged.connect(
            lambda: self.data_parameter_changed.emit()
        )
        self.form.addRow("Z stretch", self.z_stretch_spin)

        for spin in (
            self.threshold_spin,
            self.z_stretch_spin,
        ):
            spin.setFixedWidth(72)

        self.checkbox_coordinates = QCheckBox("Coordinate grid")
        self.checkbox_coordinates.setChecked(True)
        self.checkbox_coordinates.setToolTip(
            "Floor grid with axes, the hovered point's coordinates and the "
            "direction marker."
        )
        self.checkbox_coordinates.toggled.connect(self.show_coordinates_changed)
        self.form.addRow(self.checkbox_coordinates)

        self.checkbox_hover_neuron = QCheckBox("Show neuron on hover")
        self.checkbox_hover_neuron.setChecked(True)
        self.checkbox_hover_neuron.setToolTip(
            "Hovering a footprint briefly shows the other footprints of its "
            "neuron in a faint hover colour (fading out within 2 s)."
        )
        self.form.addRow(self.checkbox_hover_neuron)

        ## rotate the view once around the newly focused neuron
        self.checkbox_rotate_on_focus = QCheckBox("Rotate on focus")
        self.checkbox_rotate_on_focus.setChecked(False)
        self.checkbox_rotate_on_focus.setToolTip(
            "On focusing a new neuron, turn the view once around the z axis "
            "through its union centroid. Dragging the view stops the turn."
        )
        self.rotation_duration_spin = QDoubleSpinBox()
        self.rotation_duration_spin.setDecimals(1)
        self.rotation_duration_spin.setRange(1.0, 30.0)
        self.rotation_duration_spin.setSingleStep(0.5)
        self.rotation_duration_spin.setValue(4.0)
        self.rotation_duration_spin.setSuffix(" s")
        self.rotation_duration_spin.setFixedWidth(72)
        self.rotation_duration_spin.setToolTip("Duration of one full turn.")
        self.rotation_duration_spin.setEnabled(False)
        self.checkbox_rotate_on_focus.toggled.connect(
            self.rotation_duration_spin.setEnabled
        )
        self.checkbox_rotate_on_focus.toggled.connect(self.rotate_on_focus_changed)

        rotate_row = QWidget()
        rotate_layout = QHBoxLayout(rotate_row)
        rotate_layout.setContentsMargins(0, 0, 0, 0)
        rotate_layout.addWidget(self.checkbox_rotate_on_focus)
        rotate_layout.addWidget(self.rotation_duration_spin)
        rotate_layout.addStretch(1)
        self.form.addRow(rotate_row)

    def _add_section(self, text, first=False):
        """Grey small-caps group header (as in the context menus)."""
        label = QLabel(text.upper())
        label.setObjectName("optionsSection")
        label.setStyleSheet(
            "QLabel#optionsSection { color: #8a929c; font-size: 8pt;"
            " font-weight: 600; letter-spacing: 1px;"
            f" padding-top: {0 if first else 6}px; }}"
        )
        self.form.addRow(label)

    def _indented(self, widget):
        row = QWidget()
        layout = QHBoxLayout(row)
        layout.setContentsMargins(self.INDENT, 0, 0, 0)
        layout.addWidget(widget)
        layout.addStretch(1)
        return row

    def _on_session_only_changed(self):

        active = self.checkbox_session_only.isChecked()

        self.checkbox_filter_focused.setEnabled(active)
        self.checkbox_show_inferred.setEnabled(active)
        self.session_only_changed.emit(active)

        super()._on_session_only_changed()


def match_quality_html(similarity, shift, p_same, cmap=MATCH_QUALITY_CMAP, units=None):
    """Coloured p / c / shift labels of the reassignment menus (the shift
    in px, shown in ``units``; the colour scale stays in px)."""
    html_probability = ""
    html_similarity = ""
    html_shift = ""
    if p_same is not None:
        html_probability = get_colored_label(
            f"p={p_same:.2f}", float(np.clip(1 - p_same, 0.0, 1.0)), cmap, ["b"]
        )

    if similarity is not None:
        html_similarity = "c=" + get_colored_label(
            f"{similarity:.2f}",
            float(np.clip((1 - similarity) / 2.0, 0.0, 1.0)),
            cmap,
        )

    if shift is not None:
        html_shift = "shift=" + get_colored_label(
            (
                f"{shift:.2f}px"
                if units is None
                else f"{units.to_display(shift, 'distance'):.2f}{units.label('distance')}"
            ),
            float(np.clip(shift / 10.0, 0.0, 1.0)),
            cmap,
        )

    return html_probability, html_similarity, html_shift


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
    empty = (
        np.empty((0, 3), dtype=np.float32),
        np.empty((0, 3), dtype=np.uint32),
        np.empty((0, 4), dtype=np.float32),
        np.empty((0, 3), dtype=np.float32),
    )

    footprint = sparse.csc_matrix(footprint, copy=True)
    footprint.sum_duplicates()
    footprint.eliminate_zeros()

    if footprint.nnz == 0:
        return empty

    maximum = footprint.data.max()

    if not np.isfinite(maximum) or maximum <= 0:
        return empty

    y, x = np.unravel_index(footprint.indices, dim)
    z = np.asarray(
        footprint.data / maximum,
        dtype=np.float32,
    )

    keep = np.isfinite(z) & (z > z_thr)

    if not np.any(keep):
        return empty

    x = x[keep]
    y = y[keep]
    z = z[keep]
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

    # Blocks with three of four pixels: one triangle across the block, so
    # outlines run diagonally instead of in pixel steps (same orientation
    # as the full blocks).
    present = [i00 >= 0, i10 >= 0, i01 >= 0, i11 >= 0]
    corners = (i00, i10, i01, i11)
    three = sum(m.astype(np.int8) for m in present) == 3
    if np.any(three):
        diagonal = []
        # (missing corner, triangle) with a=00, b=10, c=01, d=11
        for missing, (p, q, r) in (
            (3, (0, 1, 2)),  # no d: a b c
            (2, (0, 1, 3)),  # no c: a b d
            (1, (0, 3, 2)),  # no b: a d c
            (0, (1, 3, 2)),  # no a: b d c
        ):
            blocks = three & ~present[missing]
            if np.any(blocks):
                diagonal.append(
                    np.column_stack(
                        [corners[k][blocks].astype(np.uint32) + base_vertex_offset for k in (p, q, r)]
                    )
                )
        if diagonal:
            faces = np.concatenate([faces, *diagonal]).astype(np.uint32)

    colors = np.tile(
        np.asarray(rgba, dtype=np.float32),
        (vertices.shape[0], 1),
    )

    return vertices, faces, colors, pick_points
