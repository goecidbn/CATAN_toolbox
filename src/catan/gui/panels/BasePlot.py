from dataclasses import dataclass
from typing import Generic, List, Optional, TypeVar, Tuple, Literal
from collections.abc import Callable
import numpy as np

from vispy import scene
from vispy.scene.visuals import Rectangle, Markers, Line
from PySide6.QtCore import QObject, Signal
from PySide6.QtWidgets import QSizePolicy, QToolTip
from PySide6.QtGui import QCursor

import importlib
from catan.gui.panels import StatisticsData, styles
from catan.gui.structures.state import NeuronComponent, AppState
from catan.gui.structures.data import Data

from catan.gui.data.statistics import PickTable
from catan.gui.data.statistics.dimensions import (
    neuron_bound_dim,
    component_bound_dims,
)
from catan.gui.data.statistics.tabledata import (
    prepare_neuron_table_query,
    prepare_component_table_query,
)


@dataclass(slots=True)
class TooltipStatisticResult:
    table: PickTable
    session_dim: str | None
    row_lookup: dict
    title: str

    def value_for_component(self, component: NeuronComponent):

        if self.session_dim is None:
            key = int(component.neuron_id)

        else:
            if component.session_id is None:
                return np.nan

        row = self.row_lookup.get(component.id)

        if row is None:
            return np.nan

        return self.table.values[row]


SelectionType = Literal["selected", "focused", "highlighted", "hovered"]


class BaseCanvas(scene.SceneCanvas):

    start_drag = None

    main_visual: Callable
    main_visual_name: str

    overlays: List[SelectionType]

    def __init__(self, parent, controls, config=None):
        self.styles = styles.Styles()
        super().__init__(
            keys="interactive", bgcolor=self.styles.bg_color, parent=parent
        )

        self.native.setSizePolicy(
            QSizePolicy.Policy.Expanding,
            QSizePolicy.Policy.Expanding,
        )

        self.unfreeze()
        self.state: AppState = parent.state
        self.data: Data = parent.data
        self.display_mode = parent.display_mode

        self._tooltip_statistic_signature = None
        self._tooltip_statistic_results = []

        self.controls = controls

        self.plotting = {
            "data": {},
            "visuals": {},
            "overlays": {},
        }
        self.changes_on_click = "focused"  # or "highlighted"

        self.freeze()

    def plot_neurons(self, reset=False):
        pass

    def update_hover(
        self, component: Optional[NeuronComponent | List[NeuronComponent]]
    ):

        self.handle_tooltip(component)
        self.state.update_hovered_components(component)

    def handle_tooltip(
        self, component: Optional[NeuronComponent | List[NeuronComponent]] = None
    ):
        if component is None:
            QToolTip.hideText()
            return

        if isinstance(component, list):

            if len(component) == 0:
                QToolTip.hideText()
                return

            component = component[0]

        session = self.data.sessions[component.session_id]

        session_name = session.name

        fp_id = self.state.get_footprint_from_component(component)
        if fp_id is not None and fp_id >= 0:
            text = f"Neuron ID: {component.neuron_id}\n" f"{session_name}"
        else:
            text = f"Neuron ID: {component.neuron_id}\n" f"Not in {session_name}"

        statistic_lines = self._tooltip_statistic_lines(component)

        if statistic_lines:
            text += "\n" + "\n".join(statistic_lines)

        QToolTip.showText(
            QCursor.pos(),
            text,
            self.native,
        )

    def on_mouse_move(self, event):

        if (
            event.pos is None
            or self.data is None
            or self.state.current_session_id is None
        ):
            return

        if not self.plotting["visuals"]:
            # QToolTip.hideText()
            return

        component = self.find_closest_component(event.pos)
        self.update_hover(component)

    def on_mouse_press(self, event):
        self.start_drag = event.pos

    def is_drag(self, pos, max_drag_distance=5):
        drag_distance = np.sqrt(np.square(pos - self.start_drag).sum())
        return drag_distance > max_drag_distance

    def on_mouse_release(self, event):
        if not self.plotting["visuals"]:
            return

        if event.pos is None or event.button != 1 or self.is_drag(event.pos):
            return

        component = self.find_closest_component(event.pos)
        self.state.logger.debug(
            f"[BASEPLOT] Clicked on component: {component}, updating {self.changes_on_click} component"
        )

        if self.changes_on_click == "selected":
            self.state.update_selected_components(component, event.modifiers)
        elif self.changes_on_click == "focused":
            self.state.focused_component = component
        elif self.changes_on_click == "highlighted":
            self.state.update_highlighted_components(
                component, event.modifiers, max_components=3
            )
        else:
            raise ValueError(f"Invalid changes_on_click value: {self.changes_on_click}")

    def style_records(self, selection, style: str):
        """
        Translate an interaction selection into records that should
        receive an overlay.

        Direct component-based plots use the default implementation.
        Derived plots such as Statistics may override this mapping.
        """

        if selection is None:
            return []

        if not isinstance(selection, list):
            selection = [selection]

        records = []

        for component in selection:

            rec = self.plotting["data"].get(component.id)

            if rec is not None:
                records.append(rec)

        return records

    def update_style(self, component, style="default"):

        if style not in self.overlays:
            return

        visuals = self.plotting["overlays"].get(style, [])
        for vis in visuals:
            vis.visible = False

        if component is None:
            self.update()
            return

        records = self.style_records(component, style)

        for index, rec in enumerate(records):

            if index >= len(visuals):
                self.add_overlay(style)
                visuals = self.plotting["overlays"][style]

            plot_data = self.plot_data_from_rec(rec, style)

            self._set_overlay_data(visuals[index], plot_data)

            visuals[index].visible = True

        self.update()

    def _set_overlay_data(self, visual, plot_data):

        if isinstance(visual, Rectangle):
            for key, value in plot_data.items():
                setattr(visual, key, value)

            return

        visual.set_data(**plot_data)

    def plot_data_from_rec(self, rec, style: str) -> dict[str, np.ndarray]:
        return {}

    def add_overlay(self, style):

        if self.main_visual is None:
            return

        if style not in self.plotting["overlays"]:
            self.plotting["overlays"][style] = []

        plot_options = self.styles.get_plot_options(
            style, self.main_visual_name, values=0.7
        )

        if self.main_visual is Rectangle:
            ## Rectangle requires initial values of these
            plot_options["center"] = (0, 0)
            plot_options["width"] = 1
            plot_options["height"] = 1

        vis = self.main_visual(
            **plot_options,
            parent=self.plot_root,
        )
        vis.visible = False
        vis.set_gl_state(
            blend=True,
            depth_test=False,
            blend_func=("src_alpha", "one_minus_src_alpha"),
        )
        order = {
            "selected": 70,
            "focused": 80,
            "highlighted": 90,
            "hovered": 100,
        }
        if style in order:
            vis.order = order[style]

        self.plotting["overlays"][style].append(vis)

    def clear_overlays(self):
        for styles, visuals in self.plotting["overlays"].items():
            if visuals is None:
                continue
            for visual in visuals:
                if visual is not None:
                    visual.visible = False
        self.plotting["overlays"] = {}

    def find_closest_component(
        self, mouse_pos
    ) -> Optional[NeuronComponent | List[NeuronComponent]]:
        pass

    def _tooltip_statistic_entity_mode(self) -> str:

        if self.display_mode == "tracked_overview":
            return "neuron"

        return "footprint"

    def _get_tooltip_statistic_results(self) -> list[TooltipStatisticResult]:

        entity_mode = self._tooltip_statistic_entity_mode()

        config = self.data.statistic_display_config
        engine = self.data.statistic_engine

        raw_queries = config.queries(entity_mode, "single")

        # Rebuild if either:
        # - configured columns changed
        # - underlying data changed
        # - the statistics registry was rebuilt
        signature = (
            entity_mode,
            tuple(raw_queries),
            engine.data_version(),
            id(engine.registry),
        )

        if signature == self._tooltip_statistic_signature:
            return self._tooltip_statistic_results

        results = []

        for raw_query in raw_queries:

            try:
                if entity_mode == "neuron":
                    query = prepare_neuron_table_query(raw_query, engine.registry)
                else:
                    query = prepare_component_table_query(raw_query, engine.registry)

                table = engine.evaluate_table(query)
                if table is None:
                    continue

                # ------------------------------------------------
                # Determine row binding
                # ------------------------------------------------

                if entity_mode == "neuron":
                    neuron_dim = neuron_bound_dim(table.dims)
                    if neuron_dim is None:
                        continue

                    session_dim = None

                else:
                    bound = component_bound_dims(table.dims)
                    if bound is None:
                        continue

                    neuron_dim, session_dim = bound

                # ------------------------------------------------
                # Build semantic lookup
                # ------------------------------------------------

                neurons = np.asarray(table.refs[neuron_dim])

                sessions = (
                    np.asarray(table.refs[session_dim])
                    if session_dim is not None
                    else None
                )

                lookup = {}

                for row in range(table.n_rows):
                    neuron_id = int(neurons[row])

                    if sessions is None:
                        key = neuron_id
                    else:
                        key = (neuron_id, int(sessions[row]))

                    lookup.setdefault(key, row)

                stat_def = engine.registry[query.statistic_key]

                title = StatisticsData.format_query_expression(
                    query,
                    stat_def,
                )

                results.append(
                    TooltipStatisticResult(
                        table=table,
                        session_dim=session_dim,
                        row_lookup=lookup,
                        title=title,
                    )
                )

            except Exception:
                self.state.logger.exception(
                    "Failed to prepare tooltip statistic "
                    f"{raw_query.statistic_key!r}"
                )

        self._tooltip_statistic_signature = signature
        self._tooltip_statistic_results = results

        return results

    def _tooltip_statistic_lines(self, component: NeuronComponent) -> list[str]:

        lines = []

        for result in self._get_tooltip_statistic_results():

            value = result.value_for_component(component)

            if value is None or not np.isfinite(value):
                text = "--"
            else:
                text = f"{float(value):.4g}"

            lines.append(f"* {result.title}: {text}")

        return lines


class BaseDisplayController(QObject):
    requires_update = Signal()

    def __init__(self, display_section, config=None):
        super().__init__(display_section)
        self.section = display_section
        self.state: AppState = display_section.state
        self.data: Data = display_section.data

        self.config = config or {}
        self.controls = {}
        self.display_widget = None

        self.plot_type = None

        self.state.hovered_components_changed.connect(self._on_hover_changed)
        self.state.selected_components_changed.connect(self._on_selection_changed)
        self.state.focused_component_changed.connect(self._on_focus_changed)
        self.state.highlighted_components_changed.connect(self._on_highlight_changed)

        self.state.current_session_changed.connect(self._on_session_changed)
        self.state.session_color_changed.connect(self._on_session_style_changed)
        # self.state.session_toggled.connect(self._on_session_style_changed)

        self.state.plot_update_required.connect(self.initialize_display)
        self.state.data_changed.connect(self._on_data_changed)

        self.destroyed.connect(self.deactivate)

    def activate(self):
        self.state.logger.debug("Activating plot controller")
        # print(f"Activating plot controller for section {self.section.section_id}")
        # print(f"Display mode: {self.section.display_mode}")
        self.configure_display()
        self.build_controls()
        self.initialize_display()

    def build_controls(self):
        self.clean_controls()
        # pass

    def configure_display(self):
        raise NotImplementedError("Subclasses must implement configure_display()")

    def deactivate(self):
        self.state.logger.debug("Deactivating plot controller")
        self.disconnect_signals()
        self.clean_controls()

        if self.display_widget is not None:
            self.section.clear_display_widget()
            self.display_widget.parent = None
            self.display_widget.deleteLater()
            self.display_widget = None

    def clean_controls(self):

        for key in self.controls:
            self.controls[key].parent = None
            self.controls[key].deleteLater()

        self.controls.clear()

    def disconnect_signals(self):
        self.state.hovered_components_changed.disconnect(self._on_hover_changed)
        self.state.selected_components_changed.disconnect(self._on_selection_changed)
        self.state.focused_component_changed.disconnect(self._on_focus_changed)
        self.state.highlighted_components_changed.disconnect(self._on_highlight_changed)

        self.state.current_session_changed.disconnect(self._on_session_changed)
        self.state.session_color_changed.disconnect(self._on_session_style_changed)

    def get_config(self):
        return dict(self.config)

    def _on_data_changed(self, input: Tuple[str, int]):
        pass

    def _on_session_style_changed(self):
        pass

    def _on_hover_changed(self):
        self.update_styles()

    def _on_selection_changed(self):
        # self.state.logger.debug(
        #     f"[BASEPLOT - {self.section.display_mode}] Neurons changed. Current selection: {self.state.selected_components}"
        # )
        self.update_neuron_selection()

    def _on_focus_changed(self):
        self.state.logger.debug(
            f"[BASEPLOT - {self.section.display_mode}] Focus changed. Current focus: {self.state.focused_component}"
        )
        self.update_neuron_selection()

    def _on_highlight_changed(self):
        self.state.logger.debug(
            f"[BASEPLOT - {self.section.display_mode}] Highlight changed. Current highlighted: {self.state.highlighted_components}"
        )
        self.update_styles()

    def update_neuron_selection(self):
        pass

    def update_styles(self):
        pass

    def initialize_display(self):
        ## just call all, to ensure display is properly working also on hot reloading
        self._on_session_changed()

    def _initialize_overlays(self):
        pass

    def _on_session_changed(self):
        # print("updating highlights")
        self._on_hover_changed()
        self._on_selection_changed()
        self._on_focus_changed()
        self._on_highlight_changed()


CanvasT = TypeVar("CanvasT", bound=BaseCanvas)


class CanvasController(BaseDisplayController, Generic[CanvasT]):

    canvas: CanvasT

    def configure_display(self):
        self.canvas = self.section.display_cls(
            self.section, controls=self.controls, config=self.config
        )
        self.display_widget = self.canvas.native
        self.section.set_display_widget(self.canvas.native)

    def update_styles(self):
        self.canvas.update_style(self.state.hovered_components, "hovered")
        self.canvas.update_style(self.state.selected_components, "selected")
        self.canvas.update_style(self.state.focused_component, "focused")
        self.canvas.update_style(self.state.highlighted_components, "highlighted")


class TableController(BaseDisplayController):

    def configure_display(self):
        # print("configure display for table controller")
        self.table = self.section.display_cls(
            self.section, controls=self.controls, config=self.config
        )
        self.display_widget = self.table
        self.section.set_display_widget(self.table)

    def update_neuron_selection(self):
        self.table.update_display()

    def update_styles(self):
        self.table.update_display()


class ControlsController(BaseDisplayController):

    def configure_display(self):
        self.menu = self.section.display_cls(
            self.section, controls=self.controls, config=self.config
        )
        self.display_widget = self.menu
        self.section.set_display_widget(self.menu)

    def update_neuron_selection(self):
        self.update_display()

    def update_styles(self):
        self.update_display()

    def update_display(self):
        pass
