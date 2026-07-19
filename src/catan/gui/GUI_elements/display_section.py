from functools import partial

import importlib
from typing import Optional, Dict, TypeVar

from PySide6.QtCore import Signal
from PySide6.QtWidgets import (
    QLabel,
    QWidget,
    QGridLayout,
    QHBoxLayout,
    QSizePolicy,
    QPushButton,
    QVBoxLayout,
    QFrame,
    QToolButton,
    QMenu,
)

from catan.gui.plots.BasePlot import BaseCanvas
from catan.gui.structures.state import AppState
from catan.gui.structures.data import Data
from catan.gui.plots import Footprints, Overview, Statistics
from catan.gui.plots import (
    SelectionDisplay,
)

from catan.gui.plots import (
    Traces,
)

display_modes = {
    "overview": {
        "title": "Overview",
        "options": {
            "single_overview": "Session footprints",
            "tracked_overview": "Tracked footprints",
            "selection_display": "Selection display",
        },
    },
    "footprints": {
        "title": "Neuron footprints",
        "options": {
            "single_footprint": "Neuron footprints (single)",
            "multi_footprint": "Neuron footprints (multi)",
        },
    },
    "traces": {
        "title": "Neuron traces",
        "options": {
            "single_trace": "Neuron activity (single)",
            "multi_trace": "Neuron activity (multi)",
        },
    },
    "population": {
        "title": "Population statistics",
        "options": {
            "statistics": "Statistics",
            "raster": "Raster plot",
        },
    },
}

DISPLAY_CONTROLLERS = {
    "single_overview": Overview,
    "tracked_overview": Overview,
    "single_footprint": Footprints,
    "multi_footprint": Footprints,
    "single_trace": Traces,
    "multi_trace": Traces,
    "statistics": Statistics,
    "selection_display": SelectionDisplay,
}


class DisplaySection(QFrame):
    """
    Builds the basic layout and references of a plot element
     _________________________________
    |        |               |        |
    |        |               |        |
    |        |               |        |
    | y axis |    canvas     |        |
    |        |               | select |
    |        |               |  menu  |
    |________|_______________|        |
    |        |               |        |
    |   ??   |    x axis     |        |
    |________|_______________|________|

    """

    requestSplit = Signal(str, str)  # section_id, orientation
    requestClose = Signal(str)  # section_id

    def __init__(self, parent, section_id, display_mode="empty"):
        super().__init__(parent)
        self.section_id = section_id
        self.state: AppState = parent.state
        self.data: Data = parent.data

        # self.setFrameShape(QFrame.StyledPanel)

        self.display_mode = display_mode
        self.active_controller = None
        self.display_cls = None

        self.setProperty("card", True)
        self.setObjectName("DisplaySection")

        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(0)
        # print(done)
        self.header = QFrame(self)
        self.header.setProperty("cardHeader", True)

        header_layout = QHBoxLayout(self.header)
        header_layout.setContentsMargins(10, 6, 8, 6)
        header_layout.setSpacing(6)

        # self.title_label = QLabel("Tracked footprints")
        display_mode_selector = self.build_display_mode_selector()
        display_mode_selector.setProperty("sectionTitle", True)
        header_layout.addWidget(display_mode_selector)

        # header_layout.addWidget(self.title_label)
        header_layout.addStretch()
        split_view_buttons = self.build_split_options()
        header_layout.addLayout(split_view_buttons)

        outer.addWidget(self.header)

        self.body = QWidget(self)
        body_layout = QGridLayout(self.body)
        body_layout.setContentsMargins(8, 8, 8, 8)
        body_layout.setSpacing(8)

        outer.addWidget(self.body, 1)

        # layout = QVBoxLayout(self)
        # layout.setContentsMargins(8, 8, 8, 8)
        # layout.setSpacing(2)
        # layout.setContentsMargins(2, 2, 2, 2)

        # toolbar = QHBoxLayout()
        # toolbar.addStretch()

        # layout.addLayout(toolbar)

        self.section_area = self.build_section_area()
        # layout.addWidget(self.section_area, 1)
        body_layout.addWidget(self.section_area, 0, 0)

        self.set_display_mode(self.display_mode, force=True)  # set initial plot mode

        # Placeholder for your VisPy canvas / plot mode widget

    def build_section_area(self):

        section_area = QFrame()
        section_grid = QGridLayout(section_area)

        y_options = QWidget()
        self.y_options_layout = QVBoxLayout(y_options)
        section_grid.addWidget(y_options, 0, 0)

        x_options = QWidget()
        self.x_options_layout = QHBoxLayout(x_options)
        self.x_options_layout.addStretch()
        section_grid.addWidget(x_options, 1, 1)

        self.display_host = QFrame()
        self.display_host.setProperty("plotCard", True)
        self.display_host.setSizePolicy(
            QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding
        )

        self.display_host_layout = QVBoxLayout(self.display_host)
        self.display_host_layout.setContentsMargins(4, 4, 4, 4)
        # self.display_host_layout.setSpacing(0)

        section_grid.addWidget(self.display_host, 0, 1)

        empty_corner = QWidget()
        section_grid.addWidget(empty_corner, 1, 0)

        self.side_menu = self.build_side_menu()
        section_grid.addWidget(self.side_menu, 0, 2, 2, 1)

        self.opts = [
            self.y_options_layout,
            self.x_options_layout,
            self.side_menu_layout,
        ]

        section_grid.setColumnStretch(0, 0)
        section_grid.setColumnStretch(1, 1)
        section_grid.setColumnStretch(2, 0)
        section_grid.setRowStretch(0, 1)
        section_grid.setRowStretch(1, 0)

        return section_area

    def set_display_widget(self, widget: QWidget):
        self.clear_display_widget()
        self.display_host_layout.addWidget(widget)

    def clear_display_widget(self):
        while self.display_host_layout.count():
            item = self.display_host_layout.takeAt(0)
            w = item.widget()
            if w is not None:
                w.setParent(None)
                w.deleteLater()

    def build_display_mode_selector(self) -> QToolButton:

        ## define action that is triggered per option
        def add_options(menu, meta_mode, options):
            for mode, title in options.items():
                action = menu.addAction(title)
                action.triggered.connect(
                    partial(self.set_display_mode, mode, meta_mode)
                )

        # Create the "dropdown" button
        self.display_mode_selector = QToolButton()
        self.display_mode_selector.setText("Plot modes")
        self.display_mode_selector.setMinimumWidth(120)

        # open menu on click
        self.display_mode_selector.setPopupMode(
            QToolButton.ToolButtonPopupMode.InstantPopup
        )

        # Root menu
        menu = QMenu(self.display_mode_selector)

        for mode, opts in display_modes.items():
            submenu = QMenu(opts["title"], menu)
            add_options(submenu, mode, opts["options"])
            menu.addMenu(submenu)

        self.display_mode_selector.setMenu(menu)

        return self.display_mode_selector

    def set_display_mode(
        self,
        mode: str,
        meta_mode: Optional[str] = None,
        config: Optional[Dict] = None,
        force: bool = False,
    ):
        # print(f"set display mode to {mode} for section {self.section_id}")

        if mode not in DISPLAY_CONTROLLERS:
            # print(f"Unknown display mode: {mode}")
            return
        if mode == self.display_mode and not force:
            return

        # print("[controls] display mode changed to ", mode)

        if self.active_controller is not None:
            self.active_controller.deactivate()
            self.active_controller = None

        self._clear_option_panels()

        # print(f"[DisplaySection] reloading: {mode}")
        importlib.reload(DISPLAY_CONTROLLERS[mode])

        self.display_mode = mode
        if meta_mode is None:
            for m, opts in display_modes.items():
                if mode in opts["options"]:
                    meta_mode = m
                    break
        if meta_mode is None:
            print(f"Could not find meta mode for {mode}")
            return
        title = display_modes[meta_mode]["options"][mode]
        self.display_mode_selector.setText(title)
        controller_cls = DISPLAY_CONTROLLERS[mode].Controller
        if controller_cls is None:
            return

        self.active_controller = controller_cls(
            display_section=self,
            config=config or {},
        )

        self.display_cls = DISPLAY_CONTROLLERS[mode].Display

        self.active_controller.activate()

    def _clear_option_panels(self):
        for layout in self.opts:
            clear_layout(layout)

        self.opts = []

    def build_split_options(self) -> QHBoxLayout:

        buttons_layout = QHBoxLayout()

        ## add splitting options
        btn_split_h = QToolButton()
        btn_split_h.setText("+ →")
        btn_split_h.setMaximumWidth(40)
        btn_split_v = QToolButton()
        btn_split_v.setText("+ ↓")
        btn_split_v.setMaximumWidth(40)
        btn_close = QToolButton()
        btn_close.setText("×")
        btn_close.setMaximumWidth(20)

        btn_split_h.clicked.connect(
            lambda: self.requestSplit.emit(self.section_id, "horizontal")
        )
        btn_split_v.clicked.connect(
            lambda: self.requestSplit.emit(self.section_id, "vertical")
        )
        btn_close.clicked.connect(lambda: self.requestClose.emit(self.section_id))

        buttons_layout.addWidget(btn_split_h)
        buttons_layout.addWidget(btn_split_v)
        buttons_layout.addWidget(btn_close)

        self.btn_split_h = btn_split_h
        self.btn_split_v = btn_split_v
        self.btn_close = btn_close

        return buttons_layout

    def set_split_options_enabled(self, can_split: bool, can_close: bool):
        self.btn_split_h.setEnabled(can_split)
        self.btn_split_v.setEnabled(can_split)
        self.btn_close.setEnabled(can_close)

    def get_config(self) -> dict:
        return {
            "display_mode": self.display_mode,
            # add whatever else this section owns:
            # "trace_key": self.trace_key,
            # "show_processed": self.show_processed,
            # "distance_threshold": self.distance_threshold,
        }

    def apply_config(self, config: dict):
        self.display_mode = config.get("display_mode", "empty")

        # restore other values here
        # self.trace_key = config.get("trace_key", "C")
        # self.show_processed = config.get("show_processed", True)

        # self.update_plot()
        # print("updating plots")

    def build_side_menu(self) -> QWidget:
        side_menu = QWidget()
        self.side_menu_layout = QVBoxLayout(side_menu)
        return side_menu


from shiboken6 import isValid


def clear_layout(layout):
    if layout is None or not isValid(layout):
        return

    while layout.count():
        item = layout.takeAt(0)

        w = item.widget()
        if w is not None:
            w.setParent(None)
            w.deleteLater()
            continue

        child_layout = item.layout()
        if child_layout is not None:
            clear_layout(child_layout)
            continue
