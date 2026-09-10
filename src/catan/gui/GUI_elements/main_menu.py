from typing import Dict, Optional, Tuple, List
import importlib

from PySide6.QtWidgets import (
    QInputDialog,
    QMenu,
    QWidget,
    QFormLayout,
    QLineEdit,
    QPushButton,
    QHBoxLayout,
    QLabel,
    QFrame,
    QVBoxLayout,
    QCheckBox,
    QSizePolicy,
    QComboBox,
    QToolButton,
)
from PySide6.QtCore import QSettings, QThreadPool, Qt, Signal
from PySide6.QtGui import QAction
from shiboken6 import isValid

from pathlib import Path

from catan.gui.structures import data, state, config

from .resource_monitor import ResourceMonitor
from .fragments import (
    FieldConfigConstructor,
    TaskOverviewDisplay,
    make_icon_button,
    set_button_icon,
    choose_path,
)
from . import session_overview

selector_options = {
    "model": ["Load ..."],
    "assignments": ["Create new", "Load ..."],
}


class MainMenu(QFrame):
    """
    Side menu panel for data loading and parameter settings.
    Contains file path selectors, load/save buttons, and mode checkboxes.
    """

    load_config_changed = Signal(
        int
    )  # signal to indicate that the load configuration has changed

    def __init__(self, parent):
        super().__init__(parent)

        self.settings = QSettings()
        self.config: config.ConfigData = parent.config
        self.state: state.AppState = parent.state
        self.data: data.Data = parent.data

        self._restore_settings()

        self.threadpool = QThreadPool.globalInstance()
        self.current_worker = None

        self.setFrameShape(QFrame.Shape.StyledPanel)
        self.setMinimumWidth(350)
        self.setSizePolicy(
            QSizePolicy.Policy.Expanding,
            QSizePolicy.Policy.Expanding,
        )

        layout = QVBoxLayout(self)
        layout.setContentsMargins(4, 4, 4, 4)
        layout.setSpacing(8)

        # self.logging = QComboBox()
        # self.logging.addItems(["DEBUG", "WARNING", "ERROR"])
        # self.logging.setCurrentText(self.state.logging_level)
        # self.logging.currentTextChanged.connect(self.change_logging_level)
        # layout.addWidget(QLabel("Logging level:"))
        # layout.addWidget(self.logging)
        layout.addWidget(self.build_root_selector())

        session_list = session_overview.SessionOverview(self)
        layout.addWidget(session_list, stretch=1)

        layout.addWidget(self.build_app_mode_menu())

        layout.addStretch()

        self.task_overview = TaskOverviewDisplay(
            self.state.tasks,
        )
        layout.addWidget(self.task_overview)
        layout.addWidget(ResourceMonitor(parent=self))

        session_list.load_requested.connect(self.process_data_from_session)

        self.state.data_changed.connect(self._on_data_changed)
        self.state.busy_changed.connect(self.toggle_busy)

    def rebuild(self):
        # importlib.reload(session_overview)
        importlib.reload(data)
        importlib.reload(session_overview)

    # def change_logging_level(self):

    #     self.state.set_logging_level(self.logging.currentText())
    #     print("Logging level changed to:", self.state.logging_level)

    def toggle_busy(self, busy: bool):
        self.button_process.setEnabled(not busy)
        # self.button_cancel.setVisible(busy)

    def on_process_all(self):
        for session in self.data.sessions:
            self.process_data_from_session(session.id)

    def process_data_from_session(self, session_id: int):

        self.data.queue_load_data(session_id)

        if self.checkbox_update_model.isChecked():
            self.data.queue_update_model(session_id)

        if self.checkbox_assign_neurons.isChecked():
            self.data.queue_assign_neurons(session_id)

        # if not task.worker.is_cancelled():
        #     task.worker.cancel()
        #     print("Cancelled model update task after loading.")

    def _on_data_changed(self, input: tuple[str, int]):
        """
        updates GUI element availability based on the current state of the data
        """
        ## disable changing root path, when sessions are loaded,
        ## to avoid path inconsistencies
        sessions_loaded = len(self.data.sessions) > 0

        self.button_root_path.setEnabled(not sessions_loaded)
        self.edit_root_path.setEnabled(not sessions_loaded)

        ## model buttons
        local_model = (self.data.model is not None) and (not self.data.model.loaded)
        model_fit_possible = (
            local_model
            and sum([session.status["aligned"] for session in self.data.sessions]) > 1
        )
        self.loader["model"]["button_execute"].setEnabled(model_fit_possible)

        model_fitted = self.data.model is not None and self.data.model.fitted
        self.loader["model"]["button_save"].setEnabled(model_fitted)

        ## registration buttons
        self.loader["assignments"]["button_execute"].setEnabled(
            sessions_loaded and model_fitted
        )

        if self.data.assignments is None:
            return
        any_assigned = any(self.data.assignments.matched_status)
        self.loader["assignments"]["button_save"].setEnabled(any_assigned)
        self.update_buttons()

    def build_app_mode_menu(self) -> QWidget:
        """
        Here, rather build the whole menu inside a subspace of layout and only delete this
        (as right now, deleting the options shifts up th path list)
        """
        ## first, disband previous menu
        # while (child := self.paths_layout.takeAt(0)) is not None:
        #     if child.widget() is not None:
        #         child.widget().deleteLater()

        ## then, build new menu

        # File paths & fields
        paths_menu = QWidget()
        self.paths_layout = QVBoxLayout(paths_menu)

        formFrame = QFrame()
        formFrame.setFrameShape(QFrame.Shape.StyledPanel)

        form = QFormLayout(formFrame)
        self.form = form

        ## add connected path loading and editing option for root path

        form.addRow(QLabel("Data paths:"), QLabel(""))

        self.loader = {}
        ## model data loading options
        form.addRow(
            QLabel("Model"),
            self.build_load_options(
                "model",
                self.data.available_models,
                add_options=selector_options["model"],
            ),
        )

        ### assignment data loading options
        self.config_constructor = FieldConfigConstructor(self, self.data.assignments)
        form.addRow(
            QLabel("Assignments"),
            self.build_load_options(
                "assignments",
                self.data.available_assignments,
                add_options=selector_options["assignments"],
                add_widgets=[self.config_constructor.toggle_config_options],
            ),
        )
        form.addRow(self.config_constructor.config_options)

        self.paths_layout.addWidget(formFrame, alignment=Qt.AlignmentFlag.AlignTop)

        self.checkbox_update_model = QCheckBox("Register to model after loading")
        self.checkbox_update_model.setChecked(True)
        form.addRow(self.checkbox_update_model)

        self.checkbox_assign_neurons = QCheckBox("Track neurons after loading")
        self.checkbox_assign_neurons.setChecked(True)
        form.addRow(self.checkbox_assign_neurons)

        ### triggering processing
        ## default processing
        self.button_process = QToolButton(self)
        self.button_process.setText("Process data")
        self.button_process.setPopupMode(
            QToolButton.ToolButtonPopupMode.MenuButtonPopup
        )
        self.button_process.clicked.connect(self.on_process_all)

        ## alternative processing
        menu = QMenu(self.button_process)
        menu.addAction(QAction("Complete loading", menu))
        menu.addAction(QAction("Complete model registration", menu))
        menu.addAction(QAction("Complete registration", menu))
        self.button_process.setMenu(menu)

        self.paths_layout.addWidget(self.button_process)

        self.button_save = QPushButton("Save results")
        self.paths_layout.addWidget(self.button_save)

        self.update_buttons()

        return paths_menu

        # self.checkbox_auto_advance = QCheckBox("Auto-advance to next cluster")
        # self.checkbox_skip_processed_side = QCheckBox("Skip processed in navigation")
        # self.paths_layout.addWidget(self.checkbox_auto_advance)
        # self.paths_layout.addWidget(self.checkbox_skip_processed_side)

    def build_root_selector(self) -> QWidget:

        self.edit_root_path = QLineEdit(text=self.data.root)
        self.button_root_path = QPushButton("...")
        self.button_root_path.setFixedWidth(50)

        widget = QWidget()
        layout = QHBoxLayout(widget)
        layout.addWidget(QLabel("Root folder:"))
        layout.addWidget(self.edit_root_path)
        layout.addWidget(self.button_root_path)

        def on_root_path_changed():
            self.data.root = self.edit_root_path.text().strip()

        self.button_root_path.clicked.connect(
            lambda: (
                choose_path(
                    self,
                    pick_dir=True,
                    init_path=self.data.root,
                    edit_line=self.edit_root_path,
                    display_text="Select root folder",
                    only_existing=True,
                ),
                on_root_path_changed(),
            )
        )
        self.edit_root_path.editingFinished.connect(on_root_path_changed)
        return widget

    def build_load_options(
        self,
        key,
        options,
        add_options: Optional[list[str]] = None,
        add_widgets: List[QWidget] = [],
    ) -> QHBoxLayout:

        entry_layout = QHBoxLayout()

        self.loader[key] = {}
        self.loader[key]["options"] = options

        ## option selection
        self.loader[key]["selector"] = QComboBox()
        self.loader[key]["selector"].setFixedWidth(90)
        self.loader[key]["selector"].currentTextChanged.connect(
            lambda text, key=key: self._on_load_option_changed(key, text)
        )

        self.rebuild_selector(key, options=options, add_options=add_options)
        entry_layout.addWidget(self.loader[key]["selector"])

        ## load / execute button
        self.loader[key]["button_execute"] = make_icon_button(
            "folder-open", tooltip=f"Load {key} data", size=28, icon_size=22
        )
        self.loader[key]["button_execute"].setFixedWidth(25)
        entry_layout.addWidget(self.loader[key]["button_execute"])

        ## save button
        self.loader[key]["button_save"] = make_icon_button(
            "floppy-disk", tooltip=f"Save {key} data", size=28, icon_size=22
        )
        self.loader[key]["button_save"].setFixedWidth(35)
        self.loader[key]["button_save"].setEnabled(False)
        entry_layout.addWidget(self.loader[key]["button_save"])

        self.loader[key]["button_save"].clicked.connect(
            lambda method=key: self.save_data(key)
        )

        def on_button_click():
            pass

        if key == "model":
            set_button_icon(
                self.loader["model"]["button_execute"],
                "play",
                tooltip=f"Run model fitting",
            )

            def on_button_click():

                if self.data.model is not None and self.data.model.loaded:
                    self.state.issue(
                        "warning",
                        "Model registration not allowed",
                        f"Model '{self.data.current_model_name}' was loaded from file and cannot be updated. Please create a new model to fit to data.",
                    )
                    return
                for session in self.data.sessions:
                    self.data.queue_update_model(session.id)

        if key == "assignments":

            def on_button_click():
                if self.data.assignments is None:
                    return

                if (
                    self.data.assignments
                    and self.data.assignments.path
                    and not self.data.assignments.loaded
                ):
                    self.data.load_assignments()
                else:
                    # if self.data.assignments.loaded:
                    for session in self.data.sessions:
                        self.data.queue_assign_neurons(session.id)
                # else:

        self.loader[key]["button_execute"].clicked.connect(on_button_click)
        self.loader[key]["button_execute"].setEnabled(False)

        for widget in add_widgets:
            entry_layout.addWidget(widget)
        return entry_layout

    def save_data(self, key):

        save_path = choose_path(
            self,
            pick_dir=False,
            init_path=str(Path(self.data.root) / f"catan_{key}.hdf5"),
            display_text=f"Select folder to save {key} file to",
            only_existing=False,
        )
        if save_path is None:
            return

        # if key == "sessions":
        #     self.data.save_sessions(save_path)

        if key == "model":
            self.data.save_model(save_path)

        if key == "assignments":
            self.data.save_assignments(save_path)

    def _on_load_option_changed(self, key, opt: str):

        if not opt:
            return

        if key == "model":
            if opt in selector_options["model"]:
                load_path = None
                if opt == "Load ...":
                    load_path = choose_path(
                        self,
                        pick_dir=False,
                        init_path=self.data.root,
                        display_text="Select model file",
                        only_existing=True,
                    )
                    if not load_path:
                        self.loader[key]["selector"].setCurrentIndex(0)
                        return

                name, ok = QInputDialog.getText(
                    self, "Model name", "Enter a name for the model:"
                )
                if ok and isinstance(name, str):
                    self.data.add_model(name, load_path)

                    self.rebuild_selector(
                        key,
                        self.data.available_models,
                        add_options=selector_options[key],
                    )
                    index = self.data.available_models.index(name)
                else:
                    index = 0

                self.loader[key]["selector"].setCurrentIndex(index)
            else:
                self.data.change_model(opt)

        elif key == "assignments":
            if opt in selector_options["assignments"]:
                load_path = None
                if opt == "Load ...":
                    load_path = choose_path(
                        self,
                        pick_dir=False,
                        init_path=self.data.root,
                        display_text="Select assignment file",
                        only_existing=True,
                    )
                    if not load_path:
                        self.loader[key]["selector"].setCurrentIndex(0)
                        return

                name, ok = QInputDialog.getText(
                    self, "Assignment name", "Enter a name for the assignment:"
                )
                # print("load path obtainned: ", load_path)
                if ok and isinstance(name, str):
                    self.data.register_assignments(load_path, name)
                    # self.data.add_assignments(name, load_path)

                    self.rebuild_selector(
                        key,
                        self.data.available_assignments,
                        add_options=selector_options[key],
                    )
                    self.config_constructor.update_source(self.data.assignments)

                    if name in self.data.available_assignments:
                        index = self.data.available_assignments.index(name)
                    else:
                        ## if adding failed for whatever reason
                        index = 0
                else:
                    index = 0

                self.loader[key]["selector"].setCurrentIndex(index)
            else:
                self.data.change_assignments(opt)

            self.update_buttons()

    def update_buttons(self):

        ## update of assignments button
        loading = False
        if (
            self.data.assignments
            and self.data.assignments.path
            and not self.data.assignments.loaded
        ):
            loading = True
            set_button_icon(
                self.loader["assignments"]["button_execute"],
                "folder-open",
                tooltip=f"Load assignments",
            )
        elif self.data.assignments:
            set_button_icon(
                self.loader["assignments"]["button_execute"],
                "play",
                tooltip=f"Run neuron registration",
            )

        if self.data.assignments is None:
            enable_button = False
        else:
            enable_button = loading
            enable_button |= not all(
                [
                    not self.data.session_assigned(s.id) and s.status["spatial_loaded"]
                    for s in self.data.sessions
                ]
            )
        self.loader["assignments"]["button_execute"].setEnabled(enable_button)

    def rebuild_selector(
        self, key: str, options: list[str], add_options: Optional[list[str]] = None
    ):
        selector = self.loader[key]["selector"]
        assert isinstance(selector, QComboBox), "Selector must be a QComboBox"

        selector.blockSignals(True)
        selector.clear()
        if add_options:
            for add_opt in add_options:
                options.append(add_opt)

        selector.addItems([opt for opt in options])
        selector.blockSignals(False)

    ### ------------------------------------------###
    ###    Logic for saving/restoring settings    ###
    ### ------------------------------------------###
    def _restore_settings(self):
        # print("Restoring settings...")
        self.defaults = {
            f"{name}_{info['type']}": self.settings.value(
                f"paths/{name}_{info['type']}", "", type=str
            )
            for name, info in self.config.paths.items()
        }
        self.data.root = str(self.defaults["root_folder"])

    def _save_settings(self):
        """
        this is currently just in a quick patch state - should be fixed!
        """
        key = "root_folder"
        self.settings.setValue(f"paths/{key}", str(self.data.root))

        # for name, info in self.config.paths.items():
        #     key = f"{name}_{info['type']}"
        #     # try:
        #     ## only relative structure is stored
        #     # relative = paths[info["root"]]
        #     path_key = getattr(self, key, None)
        #     print(f"Saving setting for key: {key}, path: {path_key}")
        #     if path_key:
        #         if info["root"]:
        #             relative_path = Path(path_key).relative_to(
        #                 getattr(self, info["root"] + "_folder")
        #             )
        #         else:
        #             relative_path = path_key
        #         self.settings.setValue(f"paths/{key}", str(relative_path))
        #     # except:
        #     #     # if mode was not selected, variables wont be set, so just skip
        #     #     pass

        self.settings.sync()  # flush to disk


# def toggle_enable(widgets: List[QWidget], enabled: bool):
#     for w in widgets:
#         w.setEnabled(enabled)

#         if not enabled and isinstance(w, (QLineEdit, QLabel)):
#             w.setText("")
