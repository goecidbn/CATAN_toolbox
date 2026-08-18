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
from .fragments.dialog_load_field import FieldSelectDialog
from shiboken6 import isValid

from pathlib import Path
from functools import partial

from catan.gui.structures import data, state, config

from .resource_monitor import ResourceMonitor
from .fragments.FileReviewDialog import GlobReviewDialog
from .fragments.TaskQueueDisplay import TaskOverviewDisplay, TaskQueueDisplay
from .fragments.FieldChips import FieldChip
from .utils.FlowLayout import FlowLayout

# from .session_overview import SessionOverview
from . import session_overview


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
        self.button_load.setEnabled(not busy)
        # self.button_cancel.setVisible(busy)

    def on_load_clicked(self):

        # run "busy method" to load data and update model
        self.state.busy = True
        for session in self.data.sessions:
            self.load_data_from_session(session.id)

    def load_data_from_session(self, session_id: int):

        session = self.data.sessions[session_id]

        self.state.tasks.start(
            "loading",
            f"Loading data for {session.name}",
            self.data.load_data,
            session_id=session_id,
            fields_to_load=self.config.fields,
            finished=lambda input=("session", session_id): self.state.data_changed.emit(
                input
            ),
        )

        if (
            not session.status["registered_to_model"]
            and self.checkbox_model_registration.isChecked()
        ):
            self.state.tasks.start(
                "model update",
                f"Update model for {session.name}",
                self.data.update_model_with_data,
                from_session_index=session_id,
                finished=self.fit_after_loading,
                ready=lambda session=session: session.status["aligned"],
            )

        if (
            not session.status["registered_to_model"]
            and self.checkbox_neuron_registration.isChecked()
        ):

            self.state.tasks.start(
                "calculating",
                f"Register neurons for {session.name}",
                self.data.register_neurons,
                from_session_index=session_id,
                clean_traces=False,
                ready=lambda session=session, session_id=session_id: (
                    (session_id == 0) or self.data.model_fitted
                )
                and session.status["aligned"],
            )

    def fit_after_loading(self, key: str = "model update"):

        if (
            self.state.tasks.current[key] is not None
            or len(self.state.tasks.queues[key]) > 0
        ):
            return

        print(" ------ fit to model! ------")

        self.state.tasks.start(
            "calculating",
            "Fit model to data",
            self.data.fit_to_model,
        )

        # if not task.worker.is_cancelled():
        #     task.worker.cancel()
        #     print("Cancelled model update task after loading.")

    def _on_data_changed(self, input: tuple[str, int]):
        ## only evaluate, when session is added / removed (?)
        data_type, data_value = input
        if (
            data_type != "sessions"
            or self.data is None
            or not isinstance(data_value, int)
        ):
            return

        ## disable changing root path, when sessions are loaded,
        ## to avoid path inconsistencies
        sessions_loaded = len(self.data.sessions) > 0

        self.root["button"].setEnabled(not sessions_loaded)
        self.root["edit"].setEnabled(not sessions_loaded)
        for key in self.config.fields:
            self.config.fields[key]["opts_ref"].setVisible(sessions_loaded)

        # session_id = data_value
        # session = self.data.sessions[session_id]
        # self.path_list._add_session_row(session_id, session)

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
                ),
                on_root_path_changed(),
            )
        )
        self.root["edit"].editingFinished.connect(on_root_path_changed)
        form.addRow(QLabel("Data paths:"), QLabel(""))
        self.loader = {}

        self.models = ["Local"]
        self.assignments = ["Local"]

        form.addRow(
            QLabel("Session Data"),
            self.build_load_options("session", ["Detection", ".* (glob)", "Tracked"]),
        )
        # self.loader["session"]["selector"].setCurrentIndex(1)

        opt_row = QHBoxLayout()
        self.loader["session"]["edit"] = QLineEdit("", placeholderText="regex pattern")
        self.loader["session"]["edit"].setText("Session0*/neuron*")

        opt_row.addWidget(self.loader["session"]["edit"])
        # opt_row.addWidget(self.loader["session"]["button"])
        form.addRow(opt_row)
        self.loader["session"]["additional_options"] = opt_row

        self._on_load_option_changed("session", 0)

        form.addRow(
            QLabel("Model Data"),
            self.build_load_options("model", self.models, add_option=True),
        )
        form.addRow(
            QLabel("Assignment Data"),
            self.build_load_options("assignment", self.assignments, add_option=True),
        )

        self.paths_layout.addWidget(formFrame, alignment=Qt.AlignmentFlag.AlignTop)

        load_options = self.build_load_defaults()
        self.paths_layout.addWidget(load_options, alignment=Qt.AlignmentFlag.AlignTop)

        self.checkbox_model_registration = QCheckBox("Register to model after loading")
        self.checkbox_model_registration.setChecked(True)
        form.addRow(self.checkbox_model_registration)

        self.checkbox_neuron_registration = QCheckBox("Track neurons after loading")
        self.checkbox_neuron_registration.setChecked(True)
        form.addRow(self.checkbox_neuron_registration)

        ### triggering processing
        ## default processing
        self.button_load = QToolButton(self)
        self.button_load.setText("Process data")
        self.button_load.setPopupMode(QToolButton.ToolButtonPopupMode.MenuButtonPopup)
        self.button_load.clicked.connect(self.on_load_clicked)

        ## alternative processing
        menu = QMenu(self.button_load)
        menu.addAction(QAction("Complete loading", menu))
        menu.addAction(QAction("Complete model registration", menu))
        menu.addAction(QAction("Complete registration", menu))
        self.button_load.setMenu(menu)

        self.paths_layout.addWidget(self.button_load)

        self.button_save = QPushButton("Save results")
        self.paths_layout.addWidget(self.button_save)

        self.path_list = session_overview.SessionOverview(self)
        self.paths_layout.addWidget(self.path_list)

        self.path_list.rebuild()

        # self.checkbox_auto_advance = QCheckBox("Auto-advance to next cluster")
        # self.checkbox_skip_processed_side = QCheckBox("Skip processed in navigation")
        # self.paths_layout.addWidget(self.checkbox_auto_advance)
        # self.paths_layout.addWidget(self.checkbox_skip_processed_side)

        self.path_list.load_requested.connect(self.load_data_from_session)

    def build_load_defaults(self) -> QWidget:

        ### Options for default processing
        load_options = QWidget()
        load_options_layout = QVBoxLayout(load_options)
        load_options_layout.setContentsMargins(4, 4, 4, 4)
        load_options_layout.setSpacing(3)

        load_options_layout.addWidget(QLabel("Load options on registration:"))

        def callback(group_key, label, method="edit"):

            if self.data.current_session is None:
                path = self.data.sessions[0].path
            else:
                path = self.data.current_session.path

            assert isinstance(
                path, str | Path
            ), "No valid session path found for field selection."

            self.config.change_config_fields(
                group_key,
                label,
                partial(FieldSelectDialog.get_field, path=path),
                method,
            )

            self.config.fields[group_key]["opts_ref"].rebuild(
                self.config.fields[group_key]
            )

        ## loading options
        self.field_options = {}
        for key, load_data in self.config.fields.items():
            load_data["opts_ref"] = OptionList(key, load_data, callback)

            opts_widget = checkbox_with_options(
                load_data,
                lambda checked, key=key: self.config.fields[key].update(
                    {"load": checked}
                ),
            )
            load_options_layout.addWidget(opts_widget)
            load_data["opts_ref"].setVisible(False)

        return load_options

    def build_load_options(self, key, options, add_option=False) -> QHBoxLayout:

        entry_layout = QHBoxLayout()

        self.loader[key] = {}
        self.loader[key]["options"] = options
        self.loader[key]["selector"] = QComboBox()
        self.loader[key]["selector"].setFixedWidth(155)
        if add_option:
            options.append("Load ...")
        self.loader[key]["selector"].addItems(options)
        entry_layout.addWidget(self.loader[key]["selector"])

        if key == "session":
            self.loader[key]["selector"].setFixedWidth(100)

            self.loader[key]["button"] = QPushButton("...")
            self.loader[key]["button"].setFixedWidth(50)
            entry_layout.addWidget(self.loader[key]["button"])

            self.loader[key]["button"].clicked.connect(lambda: None)
        if key == "model":
            self.loader[key]["selector"].setFixedWidth(100)

            self.loader[key]["button"] = QPushButton("Fit")
            self.loader[key]["button"].setFixedWidth(50)
            entry_layout.addWidget(self.loader[key]["button"])

            def on_fit_clicked():
                print("fit clicked!")
                for session in self.data.sessions:
                    if not session.status["registered_to_model"]:
                        self.data.update_model_with_data(from_session_index=session.id)
                self.data.fit_to_model()
                print("Model fitted to data.")

            self.loader[key]["button"].clicked.connect(on_fit_clicked)

        # else:
        #     entry_layout.addWidget(self.loader[key]["button"])

        # self._on_load_option_changed(key, 0)
        self.loader[key]["selector"].currentIndexChanged.connect(
            lambda index, key=key: self._on_load_option_changed(key, index)
        )
        return entry_layout

    def _on_load_option_changed(self, key, index: int):

        if key == "session":
            opt = self.loader[key]["options"][index]
            self.form.setRowVisible(
                self.loader[key]["additional_options"], opt == ".* (glob)"
            )
            # self.loader[key]["additional_options"].setVisible(False)
            self.loader[key]["button"].clicked.disconnect()
            if opt == "Detection":
                self.loader[key]["button"].clicked.connect(
                    lambda: self.choose_path(
                        pick_dir=False,
                        init_path=self.root["path"],
                        display_text="Select session file",
                        add_to_pending=True,
                    )
                )

            elif opt == ".* (glob)":
                self.loader["session"]["button"].clicked.connect(
                    self.choose_sessions_from_glob
                )

            elif opt == "Tracked":
                self.loader[key]["button"].clicked.connect(
                    lambda: self.choose_path(
                        pick_dir=False,
                        init_path=self.root["path"],
                        display_text="Select tracked session file",
                    )
                )

        elif key == "model":
            if self.loader[key]["options"][index].startswith("Load"):
                load_path = self.choose_path(
                    pick_dir=False,
                    init_path=self.root["path"],
                    # only_tail=True,
                    # edit_line=self.paths["model"]["edit"],
                    display_text="Select model file",
                )
                print("Selected model file:", load_path)
                print("Loading ...")
                self.loader[key]["selector"].setCurrentIndex(0)
            else:
                self.loader[key]["selector"].setCurrentIndex(index)

        elif key == "registration":
            if self.loader[key]["options"][index].startswith("Load"):
                load_path = self.choose_path(
                    pick_dir=False,
                    init_path=self.root["path"],
                    # only_tail=True,
                    # edit_line=self.paths["registration"]["edit"],
                    display_text="Select registration file",
                )
                print("Selected registration file:", load_path)
                print("Loading ...")
                self.loader[key]["selector"].setCurrentIndex(0)
            else:
                self.loader[key]["selector"].setCurrentIndex(index)
        # self.selector_load_from.setCurrentText(self.state.logging_level)

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
            self.data.register_session(
                fields_to_load={},
                from_file=str(path),
            )
        return

    def choose_path(
        self,
        pick_dir: bool = False,
        init_path: str = "",
        only_tail: bool = False,
        edit_line: Optional[QLineEdit] = None,
        display_text: str = "Select file",
        add_to_pending: bool = False,
    ):
        if pick_dir:
            path = QFileDialog.getExistingDirectory(
                self,
                display_text,
                init_path,  # initial directory ("" = current)
            )
        else:
            path, _ = QFileDialog.getOpenFileName(
                self,
                display_text,
                init_path,  # initial directory ("" = current)
                "HDF5 files (*.hdf5 *.h5);;MATLAB files (*.mat);;All files (*)",
            )

        if add_to_pending:
            session_id = self.data.register_session(
                fields_to_load={},
                from_file=str(path),
            )

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


def checkbox_with_options(load_data: dict, callback: Callable) -> QWidget:
    label = load_data["title"]
    opts_ref = load_data["opts_ref"]

    container = QWidget()
    layout = QVBoxLayout(container)
    layout.setContentsMargins(0, 0, 0, 0)
    layout.setSpacing(2)

    header = QWidget()
    header_layout = QHBoxLayout(header)
    header_layout.setContentsMargins(0, 0, 0, 0)
    header_layout.setSpacing(4)

    chk = QCheckBox(label)
    chk.setChecked(load_data["load"])
    chk.stateChanged.connect(
        lambda state: callback(state == Qt.CheckState.Checked.value)
    )

    toggle_button = QToolButton()
    toggle_button.setCheckable(True)
    toggle_button.setChecked(False)
    toggle_button.setAutoRaise(True)
    toggle_button.setFixedWidth(22)

    header_layout.addWidget(chk)
    header_layout.addStretch()
    header_layout.addWidget(toggle_button)

    layout.addWidget(header)

    # Indented child area
    options_container = QWidget()
    options_layout = QHBoxLayout(options_container)
    options_layout.setContentsMargins(18, 0, 0, 4)
    options_layout.addWidget(opts_ref)

    layout.addWidget(options_container)

    def toggle_options(expanded: bool):
        options_container.setVisible(expanded)
        toggle_button.setArrowType(
            Qt.ArrowType.DownArrow if expanded else Qt.ArrowType.RightArrow
        )

    toggle_button.toggled.connect(toggle_options)
    toggle_options(False)

    return container


class OptionList(QWidget):

    def __init__(
        self,
        key: str,
        load_data: dict,
        callback: Callable,
    ):
        # container = QWidget()
        # structure_path = QLineEdit("/estimates")
        # layout.addWidget(structure_path)
        super().__init__()

        self.key = key
        self.list_type = load_data["type"]

        self.callback = callback

        if load_data["type"] == "static":
            layout = QGridLayout(self)
            layout.setContentsMargins(0, 2, 0, 2)
            layout.setHorizontalSpacing(6)
            layout.setVerticalSpacing(3)
            self.list_layout = layout

        elif load_data["type"] == "dynamic":
            self.list_layout = FlowLayout(self)

        else:
            raise ValueError(f"Unknown load_data type: {load_data['type']}")

        self.rebuild(load_data)

    def rebuild(self, load_data):

        if self.list_type != load_data["type"]:
            raise ValueError(
                f"Cannot rebuild OptionList of type {self.list_type} with load_data of type {load_data['type']}"
            )
        if load_data["type"] == "static":
            self.rebuild_static_options(load_data)
        elif load_data["type"] == "dynamic":
            self.rebuild_dynamic_options(load_data)
        else:
            raise ValueError(f"Unknown load_data type: {load_data['type']}")

    def clear_chips(self):

        # Clear existing chips
        while (child := self.list_layout.takeAt(0)) is not None:
            if child.widget() is not None:
                child.widget().deleteLater()

    def rebuild_static_options(self, load_data):

        if not isinstance(self.list_layout, QGridLayout):
            raise ValueError("Cannot rebuild static options on non-grid layout.")

        self.clear_chips()

        self.edit = {}

        for row, (label, key) in enumerate(load_data["opts"].items()):
            self.edit[label] = QLineEdit(key)

            browse = QToolButton()
            browse.setText("…")
            browse.setToolTip(f"Find {label.lower()} field")
            browse.setFixedWidth(25)
            browse.clicked.connect(
                lambda _, key=self.key, label=label: self.callback(
                    key, label, method="edit"
                )
            )

            self.list_layout.addWidget(
                QLabel(label),
                row,
                0,
            )
            self.list_layout.addWidget(
                self.edit[label],
                row,
                1,
            )
            self.list_layout.addWidget(
                browse,
                row,
                2,
            )

            # self.spatial_edits[label.lower()] = edit
        self.list_layout.setColumnStretch(1, 1)

    def rebuild_dynamic_options(self, load_data):

        if not isinstance(self.list_layout, FlowLayout):
            raise ValueError("Cannot rebuild dynamic options on non-flow layout.")

        self.clear_chips()

        opts = (
            load_data["opts"].items()
            if isinstance(load_data["opts"], dict)
            else load_data["opts"]
        )
        for opt in opts:
            chip = FieldChip(opt)
            self.list_layout.addWidget(chip)

            chip.field_button.clicked.connect(
                lambda _, key=self.key, label=chip.label: self.callback(
                    key, label, method="rename"
                )
            )
            chip.remove_requested.connect(
                lambda chip_label, key=self.key: self.callback(
                    key, chip_label, method="remove"
                )
            )

        add_button = QToolButton()
        add_button.setText("+")
        add_button.setToolTip(f"Add {self.key} field")
        add_button.clicked.connect(lambda _, label=None: self.callback(self.key, label))
        self.list_layout.addWidget(add_button)
