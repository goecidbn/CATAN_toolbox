from typing import Generic, List, Optional, TypeVar, Tuple
import numpy as np

from vispy import scene
from PySide6.QtCore import QObject, Signal
from PySide6.QtWidgets import QSizePolicy, QToolTip
from PySide6.QtGui import QCursor

# import importlib
from catan.gui.plots import styles
from catan.gui.structures.state import NeuronComponent, AppState
from catan.gui.structures.data import Data

# importlib.reload(styles)
# print("reloading BasePlot")


class BaseCanvas(scene.SceneCanvas):

    start_drag = None

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

        self.controls = controls

        self.plotting = {
            "data": {},
            "visuals": {},
            "overlays": {
                "selected": None,
                "hovered": None,
                "focused": None,
                "highlighted": None,
            },
        }
        self.changes_on_click = "focused"  # or "highlighted"

        self.freeze()

    def plot_neurons(self, single=True, reset=False):
        pass

    def update_hover(
        self, component: Optional[NeuronComponent | List[NeuronComponent]]
    ):

        self.handle_tooltip(component)
        self.update_style(component, "hovered")

    def handle_tooltip(
        self, component: Optional[NeuronComponent | List[NeuronComponent]] = None
    ):
        if component is None:
            QToolTip.hideText()
        else:
            if isinstance(component, list):
                if len(component) == 0:
                    QToolTip.hideText()
                    return
                component = component[0]
            fp_id = self.state.get_footprint_from_component(component)
            session_name = self.data.sessions[component.session_id].name
            if fp_id >= 0:
                A = self.data.sessions[component.session_id].A[:, fp_id]
                metrics = {
                    "SNR": self.data.sessions[component.session_id].quality["SNR_comp"][
                        fp_id
                    ],
                    "r-value": self.data.sessions[component.session_id].quality[
                        "r_values"
                    ][fp_id],
                    "CNN": self.data.sessions[component.session_id].quality[
                        "cnn_preds"
                    ][fp_id],
                    "Size": int((A > A.max() * 0.01).sum()),
                }
                text = "Neuron ID: {}\n{} \n".format(component.neuron_id, session_name)
                text += "\n".join(
                    f"* {k}: {v:.0f}" if isinstance(v, int) else f"* {k}: {v:.3f}"
                    for k, v in metrics.items()
                )
            else:
                text = f" Neuron ID: {component.neuron_id}\nNot in {session_name}"

            QToolTip.showText(
                QCursor.pos(),  # global mouse position
                text,
                self.native,  # Qt widget owning the tooltip
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
            self.state.highlighted_component = component
        else:
            raise ValueError(f"Invalid changes_on_click value: {self.changes_on_click}")

    def update_style(self, component, style):
        pass

    def find_closest_component(
        self, mouse_pos
    ) -> Optional[NeuronComponent | List[NeuronComponent]]:
        pass


class BaseDisplayController(QObject):
    requires_update = Signal()

    def __init__(self, display_section, config=None):
        super().__init__(display_section)
        self.section = display_section
        self.state = display_section.state
        self.data = display_section.data

        self.config = config or {}
        self.controls = {}
        self.display_widget = None

        self.plot_type = None
        self.single_mode = display_section.display_mode.startswith("single")

        self.state.selected_components_changed.connect(self._on_selection_changed)
        self.state.focused_component_changed.connect(self._on_focus_changed)
        self.state.highlighted_component_changed.connect(self._on_highlight_changed)

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
        pass

    def configure_display(self):
        raise NotImplementedError("Subclasses must implement configure_display()")

    def deactivate(self):
        self.state.logger.debug("Deactivating plot controller")
        self.disconnect_signals()
        for key in self.controls:
            self.controls[key].parent = None
            self.controls[key].deleteLater()

        self.controls.clear()

        if self.display_widget is not None:
            self.section.clear_display_widget()
            self.display_widget.parent = None
            self.display_widget.deleteLater()
            self.display_widget = None

    def disconnect_signals(self):
        self.state.selected_components_changed.disconnect(self._on_selection_changed)
        self.state.focused_component_changed.disconnect(self._on_focus_changed)
        self.state.highlighted_component_changed.disconnect(self._on_highlight_changed)

        self.state.current_session_changed.disconnect(self._on_session_changed)

    def get_config(self):
        return dict(self.config)

    def _on_data_changed(self, input: Tuple[str, int]):
        pass

    def _on_session_style_changed(self):
        pass

    def _on_selection_changed(self):
        # self.state.logger.debug(
        #     f"[BASEPLOT - {self.section.display_mode}] Neurons changed. Current selection: {self.state.selected_components}"
        # )
        self.update_neuron_selection()

    def _on_focus_changed(self):
        self.state.logger.debug(
            f"[BASEPLOT - {self.section.display_mode}] Focus changed. Current focus: {self.state.focused_component}"
        )
        if self.single_mode:
            self.update_neuron_selection()
        else:
            self.update_styles()

    def _on_highlight_changed(self):
        self.state.logger.debug(
            f"[BASEPLOT - {self.section.display_mode}] Highlight changed. Current highlighted: {self.state.highlighted_component}"
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
        self.canvas.update_style(self.state.selected_components, "selected")
        self.canvas.update_style(self.state.focused_component, "focused")
        self.canvas.update_style(self.state.highlighted_component, "highlighted")


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
