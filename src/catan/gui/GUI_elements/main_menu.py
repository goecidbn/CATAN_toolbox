from typing import Dict, Optional, Tuple, List, Callable
import importlib

from PySide6.QtWidgets import (
    QDialog,
    QGridLayout,
    QInputDialog,
    QMenu,
    QWidget,
    QFormLayout,
    QLineEdit,
    QPushButton,
    QHBoxLayout,
    QMessageBox,
    QLabel,
    QFrame,
    QVBoxLayout,
    QCheckBox,
    QFileDialog,
    QSizePolicy,
    QComboBox,
    QToolButton,
    QWidgetAction,
)
from PySide6.QtCore import QSettings, QThreadPool, Qt
from PySide6.QtGui import QAction, QCursor
from .fragments.field_selector import FieldSelector
from shiboken6 import isValid

from pathlib import Path

from catan.gui.structures import data, state, config

from .resource_monitor import ResourceMonitor
from .fragments.FileReviewDialog import GlobReviewDialog
from .fragments.TaskQueueDisplay import TaskOverviewDisplay, TaskQueueDisplay
from .fragments.IconButton import make_icon_button, set_button_icon
from .fragments.toggle_option import ToggleOption

from . import session_overview

selector_options = {
    "model": ["Load ..."],
    "assignments": ["Create new", "Load ..."],
    # {
    #     "load": "Load...",
    # },
    # "assignments": {
    #     "new": "Create new",
    #     "load": "Load...",
    # }
}

class MainMenu(QFrame):
    """
    Side menu panel for data loading and parameter settings.
    Contains file path selectors, load/save buttons, and mode checkboxes.
    """

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
        self.setMinimumWidth(300)  # adjust to taste
        self.setSizePolicy(
            QSizePolicy.Policy.Expanding,
            QSizePolicy.Policy.Expanding,
        )

        layout = QVBoxLayout(self)
        layout.setContentsMargins(4, 4, 4, 4)
        layout.setSpacing(8)

        self.logging = QComboBox()
        self.logging.addItems(["DEBUG", "WARNING", "ERROR"])
        self.logging.setCurrentText(self.state.logging_level)
        self.logging.currentTextChanged.connect(self.change_logging_level)
        layout.addWidget(QLabel("Logging level:"))
        layout.addWidget(self.logging)

        self.paths_menu = QWidget()
        self.paths_layout = QVBoxLayout(self.paths_menu)
        self.build_app_mode_menu()
        layout.addWidget(self.paths_menu)

        layout.addStretch()  # push everything up, so empty space is at the bottom
        layout.addWidget(ResourceMonitor(parent=self))

        self.task_overview = TaskOverviewDisplay(
            self.state.tasks,
        )
        layout.addWidget(self.task_overview)

        self.state.data_changed.connect(self._on_data_changed)
        self.state.busy_changed.connect(self.toggle_busy)

    def rebuild(self):
        # importlib.reload(session_overview)
        importlib.reload(data)
        importlib.reload(session_overview)

    def change_logging_level(self):

        self.state.set_logging_level(self.logging.currentText())
        print("Logging level changed to:", self.state.logging_level)

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

        self.root["button"].setEnabled(not sessions_loaded)
        self.root["edit"].setEnabled(not sessions_loaded)

        ## session buttons
        self.loader["session"]["button_save"].setEnabled(sessions_loaded)

        ## model buttons
        local_model = (self.data.model is not None) and (not self.data.model.loaded)
        model_fit_possible = local_model and sum([session.status["aligned"] for session in self.data.sessions]) > 1
        self.loader["model"]["button_execute"].setEnabled(model_fit_possible)

        model_fitted = self.data.model is not None and self.data.model.fitted
        self.loader["model"]["button_save"].setEnabled(model_fitted)

        ## registration buttons
        self.loader["assignments"]["button_execute"].setEnabled(sessions_loaded and model_fitted)

        any_assigned = any([session.status["matched"] for session in self.data.sessions])
        self.loader["assignments"]["button_save"].setEnabled(any_assigned)

        ## field selector
        modified = self.data.load_configs.modified
        if self.data.load_configs.current is not None:
            self.load_config_selector.setEditable(modified)
            self.load_config_selector.setCurrentText("* " +self.data.load_configs.current.name + (" (modified)" if modified else ""))


    def build_app_mode_menu(self):
        """
        Here, rather build the whole menu inside a subspace of layout and only delete this
        (as right now, deleting the options shifts up th path list)
        """
        ## first, disband previous menu
        while (child := self.paths_layout.takeAt(0)) is not None:
            if child.widget() is not None:
                child.widget().deleteLater()

        ## then, build new menu

        # File paths & fields
        formFrame = QFrame()
        formFrame.setFrameShape(QFrame.Shape.StyledPanel)

        form = QFormLayout(formFrame)
        self.form = form

        ## add connected path loading and editing option for root path
        self.root = {
            "path": self.defaults[f"root_folder"],
            "edit": QLineEdit(text=self.defaults[f"root_folder"]),
            "button": QPushButton("..."),
        }
        self.root["button"].setFixedWidth(50)

        entry_layout = QHBoxLayout()
        entry_layout.addWidget(self.root["edit"])
        entry_layout.addWidget(self.root["button"])
        form.addRow(f"Root folder:", entry_layout)

        # lambda path: self.root["path"] = path
        def on_root_path_changed():
            self.root["path"] = self.root["edit"].text().strip()

        self.root["button"].clicked.connect(
            lambda: (
                self.choose_path(
                    pick_dir=True,
                    edit_line=self.root["edit"],
                    display_text="Select root folder",
                    only_existing=True,
                ),
                on_root_path_changed(),
            )
        )
        self.root["edit"].editingFinished.connect(on_root_path_changed)
        form.addRow(QLabel("Data paths:"), QLabel(""))

        self.loader = {}

        ## session data loading options
        

        ## menu for load configuration selection and saving
        load_config_menu_widget = QWidget()
        load_config_menu = QHBoxLayout()

        self.load_config_selector = QComboBox()
        self.load_config_selector.addItems(self.data.load_configs.names())

        self.field_selector = FieldSelector(self, self.data.load_configs.current)
        def on_load_config_changed(idx):
            name = self.data.load_configs.names()[idx]
            self.data.load_configs.select(name)
            self.field_selector.rebuild(self.data.load_configs.current)
        self.load_config_selector.currentIndexChanged.connect(
            lambda idx: on_load_config_changed(idx)
        )
        self.load_config_selector.setCurrentText(self.data.load_configs.current.name)


    
        self.load_config_save_button = make_icon_button("floppy-disk", tooltip="Save load configuration", size=28, icon_size=22)

        toggle_config_fields = ToggleOption(self.field_selector)
        load_config_menu.addWidget(QLabel("Load config preset:"))
        load_config_menu.addWidget(self.load_config_selector, alignment=Qt.AlignmentFlag.AlignTop)
        load_config_menu.addWidget(toggle_config_fields, alignment=Qt.AlignmentFlag.AlignTop)
        load_config_menu.addWidget(self.load_config_save_button, alignment=Qt.AlignmentFlag.AlignTop)

        def on_save_load_config():
            if self.data.load_configs.current is None:
                self.state.issue(
                    "warning",
                    "No load configuration selected",
                    "Please select a load configuration to save.",
                )
                return
            name, ok = QInputDialog.getText(
                self,
                "Save load configuration",
                "Enter a name for the load configuration:",
                text=self.data.load_configs.current.name,
            )
            if ok and name:
                self.data.load_configs.save_current_as(name)

            self.load_config_selector.clear()
            self.load_config_selector.addItems(self.data.load_configs.names())
            self.load_config_selector.setCurrentText(self.data.load_configs.current.name)


        self.load_config_save_button.clicked.connect(
            on_save_load_config
        )
        load_config_menu_widget.setLayout(load_config_menu)

        toggle_config_option = ToggleOption(load_config_menu_widget,icon_name="gear",tooltip="Show load configuration options",expanded=False)

        form.addRow(
            QLabel("Session Data"),
            self.build_load_options("session", ["from file", ".* (glob)"], add_widgets=[toggle_config_option]),
        )
        form.addRow(load_config_menu_widget)
        # self.paths_layout.addWidget(self.field_selector, alignment=Qt.AlignmentFlag.AlignTop)
        form.addRow(self.field_selector)


        opt_row = QHBoxLayout()
        self.loader["session"]["edit"] = QLineEdit("", placeholderText="regex pattern")
        self.loader["session"]["edit"].setText("Session0*/neuron*")

        opt_row.addWidget(self.loader["session"]["edit"])
        form.addRow(opt_row)
        self.loader["session"]["additional_options"] = opt_row
        self.form.setRowVisible(self.loader["session"]["additional_options"], False)

        self._on_load_option_changed("session", 0)

        



        form.addRow(
            QLabel("Model Data"),
            self.build_load_options("model", self.data.available_models, add_options=selector_options["model"]),
        )
        form.addRow(
            QLabel("Assignment Data"),
            self.build_load_options("assignments", self.data.available_assignments, add_options=selector_options["assignments"]),
        )

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
        self.button_process.setPopupMode(QToolButton.ToolButtonPopupMode.MenuButtonPopup)
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

        self.path_list = session_overview.SessionOverview(self)
        self.paths_layout.addWidget(self.path_list)

        self.path_list.rebuild()

        # self.checkbox_auto_advance = QCheckBox("Auto-advance to next cluster")
        # self.checkbox_skip_processed_side = QCheckBox("Skip processed in navigation")
        # self.paths_layout.addWidget(self.checkbox_auto_advance)
        # self.paths_layout.addWidget(self.checkbox_skip_processed_side)

        self.path_list.load_requested.connect(self.process_data_from_session)

    def build_load_options(self, key, options, add_options: Optional[list[str]] = None, add_widgets: List[QWidget]=[]) -> QHBoxLayout:

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
        self.loader[key]["button_execute"] = make_icon_button("folder-open", tooltip=f"Load {key} data", size=28, icon_size=22)
        self.loader[key]["button_execute"].setFixedWidth(25)
        entry_layout.addWidget(self.loader[key]["button_execute"])

        ## save button
        self.loader[key]["button_save"] = make_icon_button("floppy-disk", tooltip=f"Save {key} data", size=28, icon_size=22)
        self.loader[key]["button_save"].setFixedWidth(35)
        self.loader[key]["button_save"].setEnabled(False)
        entry_layout.addWidget(self.loader[key]["button_save"])

        self.loader[key]["button_save"].clicked.connect(lambda method=key : self.save_data(key))

        if key == "session":

            self.loader[key]["button_execute"].clicked.connect(self.on_register_session)
        if key == "model":
            set_button_icon(self.loader["model"]["button_execute"], "play", tooltip=f"Run model fitting")

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

            self.loader[key]["button_execute"].clicked.connect(on_button_click)
            self.loader[key]["button_execute"].setEnabled(False)
        if key == "assignments":
            set_button_icon(self.loader["assignments"]["button_execute"], "play", tooltip=f"Run neuron registration")

            def on_button_click():
                for session in self.data.sessions:
                    self.data.queue_assign_neurons(session.id)

            self.loader[key]["button_execute"].clicked.connect(on_button_click)
            self.loader[key]["button_execute"].setEnabled(False)

        for widget in add_widgets:
            entry_layout.addWidget(widget)
        return entry_layout
    
    def save_data(self, key):

        save_path = self.choose_path(
            pick_dir=False,
            init_path=str(Path(self.root["path"]) / f"catan_{key}.hdf5"),
            display_text=f"Select folder to save {key} file to",
            only_existing=False,
        )
        if save_path is None:
            return
        # save_path = Path(save_path) / "catan"
        if key == "session":
            self.data.save_sessions(save_path)

        if key == "model":
            self.data.save_model(save_path)

        if key == "assignments":
            self.data.save_assignments(save_path)

    def on_register_session(self):

        opt = self.loader["session"]["selector"].currentText()
        
        if opt.lower() == "from file":
            ## chooses automatically between loading from single detection session or from list of sessions (from hdf5 attribute)
            path = self.choose_path(
                pick_dir=False,
                init_path=self.root["path"],
                display_text="Select session file",
                only_existing=True
            )
            if path is None:
                return
            self.state.tasks.start(
                "loading",
                "Loading session data from file...",
                lambda ctx: self.data.register_session_data(path,fields_to_load={}, ctx=ctx)
            )
            
        elif opt == ".* (glob)":
            self.choose_sessions_from_glob()
        else:
            raise ValueError(f"Unknown option selected: {opt}")
        

    def _on_load_option_changed(self, key, opt: str):

        if not opt:
            return

        if key == "session":
            self.form.setRowVisible(
                self.loader[key]["additional_options"], opt == ".* (glob)"
            )

        elif key == "model":
            if opt in selector_options["model"]:
                load_path = None
                if opt=="Load ...":
                    load_path = self.choose_path(
                        pick_dir=False,
                        init_path=self.root["path"],
                        display_text="Select model file",
                        only_existing=True,
                    )
                    if not load_path:
                        self.loader[key]["selector"].setCurrentIndex(0)
                        return
                    
                name, ok = QInputDialog.getText(self, "Model name", "Enter a name for the model:")
                if ok and isinstance(name, str):
                    self.data.add_model(name,load_path)
                    
                    self.rebuild_selector(key, self.data.available_models, add_options=selector_options[key])
                    index = self.data.available_models.index(name)
                else:
                    index = 0

                self.loader[key]["selector"].setCurrentIndex(index)
            else:
                self.data.change_model(opt)

        elif key == "assignments":
            if opt in selector_options["assignments"]:
                load_path = None
                if opt=="Load ...":
                    load_path = self.choose_path(
                        pick_dir=False,
                        init_path=self.root["path"],
                        display_text="Select assignment file",
                        only_existing=True
                    )
                    if not load_path:
                        self.loader[key]["selector"].setCurrentIndex(0)
                        return
                
                name, ok = QInputDialog.getText(self, "Assignment name", "Enter a name for the assignment:")
                if ok and isinstance(name, str):
                    self.data.add_assignments(name,load_path)

                    self.rebuild_selector(key, self.data.available_assignments, add_options=selector_options[key])

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

    def rebuild_selector(self, key: str, options: list[str], 
    add_options: Optional[list[str]] = None):
        selector = self.loader[key]["selector"]
        assert isinstance(selector, QComboBox), "Selector must be a QComboBox"

        selector.blockSignals(True)
        selector.clear()
        if add_options:
            for add_opt in add_options:
                options.append(add_opt)

        selector.addItems([opt for opt in options])
        selector.blockSignals(False)


    def choose_sessions_from_glob(self):
        root = Path(self.root["path"])
        pattern = self.loader["session"]["edit"].text()
        paths = list(root.glob(pattern))

        # show warning / empty result dialog
        if not paths:
            return

        dialog = GlobReviewDialog(
            paths,
            parent=self,
        )

        if dialog.exec() != QDialog.DialogCode.Accepted:
            return
        paths = dialog.paths()
        if paths is None:
            return

        for path in paths:
            self.data.register_session_data(
                fname=path,
                fields_to_load={},
            )
        return

    def choose_path(
        self,
        pick_dir: bool = False,
        init_path: str = "",
        only_tail: bool = False,
        edit_line: Optional[QLineEdit] = None,
        display_text: str = "Select file",
        only_existing: bool = True
    ) -> Optional[str]:
        if pick_dir:
            path = QFileDialog.getExistingDirectory(
                self,
                display_text,
                init_path,  # initial directory ("" = current)
            )
        else:
            opts = (self,
                display_text,
                init_path,  # initial directory ("" = current)
                "HDF5 files (*.hdf5 *.h5);;MATLAB files (*.mat);;All files (*)"
            )
            if only_existing:
                path, _ = QFileDialog.getOpenFileName(
                *opts
                )
            else:
                path, _ = QFileDialog.getSaveFileName(
                    *opts
                )
        if not path:
            return

        if path and edit_line is not None:
            relative_path = (
                str(Path(path).relative_to(init_path)) if only_tail else path
            )
            edit_line.setText(relative_path)
        elif path:
            return path

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

    def _save_settings(self):
        """
        this is currently just in a quick patch state - should be fixed!
        """
        key = "root_folder"
        self.settings.setValue(f"paths/{key}", str(self.root["path"]))

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

