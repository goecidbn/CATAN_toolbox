from typing import List, Optional
import os, tqdm, importlib
from PySide6.QtWidgets import (
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
)
from PySide6.QtCore import QSettings, QThreadPool, Qt
from shiboken6 import isValid

from pathlib import Path

from catan.gui.structures import data, state
from catan.gui.plots.colors import CyclicColorMap

from . import SessionOverview, ResourceMonitor
# from .dialog_load_field import FieldSelectDialog, list_file_fields

app_modes = {
    "Single session": "single",
    "Neuron tracking": "tracking",
    "Video": "video",
}

## defines paths which can be defined (and reloaded) by GUI
paths = {
    "root": {
        "mode": ["single", "tracking"],
        "type": "folder",
        "label": "Root folder",
        "root": None,
    },
    "results": {
        "mode": ["single"],
        "type": "file",
        "label": "Results file",
        "root": "root",
    },
    "model": {
        "mode": ["tracking"],
        "type": "file",
        "label": "Model file",
        "root": "root",
    },
    "registration": {
        "mode": ["tracking"],
        "type": "file",
        "label": "Registration file",
        "root": "root",
    },
}

print("reloading main menu")


class MainMenu(QFrame):
    """
    Side menu panel for data loading and parameter settings.
    Contains file path selectors, load/save buttons, and mode checkboxes.

    Exposes:
        - load_button: QPushButton
    """

    def __init__(self, parent):
        super().__init__(parent)

        self.settings = QSettings()
        self._restore_settings()

        self.state: state.AppState = parent.state
        # self.tracking: track_neurons = parent.tracking
        self.data: data.Data = parent.data

        self.threadpool = QThreadPool.globalInstance()
        self.current_worker = None

        self.session_colors = CyclicColorMap(n_colors=20, cmap_name="twilight")

        """
        TODO:
            move some parts to AppState:
                * app mode (single session / multi-session tracking / video)
                * auto-advance checkbox
                * skip processed checkbox
                * file paths
        """

        self.setFrameShape(QFrame.Shape.StyledPanel)
        self.setMinimumWidth(250)  # adjust to taste
        self.setSizePolicy(
            QSizePolicy.Policy.Expanding,
            QSizePolicy.Policy.Expanding,
        )

        # self.panel = QFrame()
        # self.panel.setFrameShape(QFrame.Shape.StyledPanel)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(4, 4, 4, 4)
        layout.setSpacing(8)

        self.logging = QComboBox()
        self.logging.addItems(["DEBUG", "WARNING", "ERROR"])
        self.logging.setCurrentText(self.state.logging_level)
        self.logging.currentTextChanged.connect(self.change_logging_level)
        layout.addWidget(QLabel("Logging level:"))
        layout.addWidget(self.logging)

        # self.checkbox_mode_multi = QCheckBox("Multi-session tracking mode")
        self.dropdown_app_mode = QComboBox()
        self.dropdown_app_mode.addItems([key for key in app_modes.keys()])
        self.dropdown_app_mode.setToolTip(
            "Select application mode: Single session stats, Multi-session neuron tracking, or Video mode"
        )
        layout.addWidget(QLabel("App mode options:"))
        layout.addWidget(self.dropdown_app_mode)

        self.paths_menu = QWidget()
        self.paths_layout = QVBoxLayout(self.paths_menu)
        self.build_app_mode_menu()
        layout.addWidget(self.paths_menu)

        layout.addStretch()  # push everything up, so empty space is at the bottom
        layout.addWidget(ResourceMonitor(parent=self))

        # layout.addLayout(self.build_menu_data_paths())

        # Meta parameters (add more as you need)

        self.dropdown_app_mode.currentIndexChanged.connect(self.change_app_mode)

        # self.state.app_mode_changed.connect(self.on_app_mode_changed)
        self.state.data_changed.connect(self.handler_data_changed)
        # self.state.current_session_changed.connect(self.on_current_session_changed)

    def rebuild(self):
        # importlib.reload(session_overview)
        importlib.reload(data)

    def change_logging_level(self):

        self.state.set_logging_level(self.logging.currentText())
        print("Logging level changed to:", self.state.logging_level)

    # def on_current_session_changed(self, session_id: int):
    #     print(f"Current session changed to {session_id}")

    ## cleaning previous trace to free memory
    ## shouldnt happen when comparing two analyses from one dataset (how?)
    # if self.data.current_session is not None:
    #     self.data.current_session.clean_traces()

    # self.data.current_session = self.data.sessions[session_id]
    # self.data.sessions[session_id].load_traces()
    # print("traces loaded, now present:", self.data.current_session.traces.keys())
    # print(
    #     "traces loaded, now present:", self.data.sessions[session_id].traces.keys()
    # )
    # print()

    def set_busy(self, busy: bool):
        pass
        # self.button_load.setEnabled(not busy)
        # self.plot_mode_selector.setEnabled(not busy)
        # self.button_cancel.setVisible(busy)

        # self.progress_bar.setVisible(busy)

        # optional: allow panning/hovering but block plot-changing controls
        # self.display_area.set_controls_enabled(not busy)

    def on_load_clicked(self):
        # print("Loading data...")
        # run "busy method" to load data and update model
        mode = app_modes[self.dropdown_app_mode.currentText()]
        if mode == "single":
            self.load_from_session(set_active=True)
        if mode == "tracking":
            self.load_from_tracking()

        # self.state.tasks.start(
        #     "rebuild_neurons",
        #     self.data.rebuild_neurons,
        #     finished=self.on_background_done,
        # )

    # def on_background_done(self, result):
    #     # print("Data loaded, now updating display...")
    #     # self.data.neurons = result
    #     self.set_busy(False)
    #     self.current_worker = None
    #     self.state.current_job = None
    #     self.state.plot_update_required.emit()
    #     self.state.current_session_id = 0

    #     # self.update_display()

    # def add_session(self, this_data: session_data):

    ## always add new session at the end of the list
    # session_id = len(self.data.sessions)

    # self.state.session_color = (session_id, self.session_colors.next())
    # self.state.session_offset = (session_id, 0)

    # if not hasattr(this_data, "path") or not this_data.path:
    #     name = f"Session {session_id + 1}"
    # else:
    #     path = Path(this_data.path).relative_to(self.root_folder)
    #     name = str(path.parent)

    # this_data.name = name
    # this_data.loaded = True

    # self.data.sessions.append(this_data)
    # self.state.session_added = session_id

    def load_from_session(self, set_active=True):
        """
        TODO:
        * change tracking stucture to hold assignments in base structure (and access from there, not hand over)
        """

        self.data.register_session(from_file=str(self.results_file), align=True)
        session_id = self.data.sessions[-1].id

        ## run tracking algorithm hereafter
        self.data.update_model_with_data(
            from_session_id=session_id,
        )
        self.state.session_color = (session_id, self.session_colors.next())

        if len(self.data.sessions) > 1:
            self.data.fit_to_model()
            # try:
            #     # print("Fitting model to data...")
            # except Exception as e:
            #     # print("Error fitting model:", e)
            #     QMessageBox.warning(
            #         self,
            #         "Model fitting error",
            #         f"An error occurred while fitting the model to the data:\n{e}",
            #     )
            #     self.state.session_added = session_id
            #     return

        self.data.register_neurons(from_session_id=session_id, clean_traces=False)
        self.state.assignments = self.data.assignments
        self.state.session_added = session_id
        # print(self.state.assignments)

        if set_active or self.state.current_session_id is None:
            self.state.current_session_id = session_id

    def load_from_tracking(self):

        self.data.load_model(self.model_file)
        self.data.load_registration(self.registration_file)
        self.state.assignments = self.data.assignments

        ## prepare checking for common path
        paths = [session.path for session in self.data.sessions if session.path is not None]
        assert len(paths) >= 1, "No sessions found in the loaded model."

        common_path = os.path.commonpath(paths)

        # self.data.sessions = []
        for session in tqdm.tqdm(self.data.sessions):
            if not common_path == self.root_folder:
                ## if the common path is not the root_folder, this might be from
                ## changing systems or file structure since analysing the data,
                ## so we adjust this
                if session.path is None:
                    raise ValueError(
                        f"Session {session.name} has no path, cannot adjust to new root folder."
                    )
                session.path = str(
                    Path(self.root_folder) / Path(session.path).relative_to(common_path)
                )
                # this_data.path = (
                #     new_path  # update path to current root folder structure
                # )
            # self.add_session(this_data)

        # self.state.current_session_id = 0

    def handler_data_changed(self, input: tuple[str, int]):
        data_type, data_value = input
        if (
            data_type != "session"
            or self.data is None
            or not isinstance(data_value, int)
        ):
            return

        session_id = data_value

        session = self.data.sessions[session_id]
        self.path_list._add_session_row(session_id, session)

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
        paths = QFrame()
        paths.setFrameShape(QFrame.Shape.StyledPanel)

        mode = self.dropdown_app_mode.currentText()
        if app_modes[mode] == "single":
            paths.setLayout(self._build_form_data_paths_single())
        elif app_modes[mode] == "tracking":
            paths.setLayout(self._build_form_data_paths_tracking())
        else:
            self.paths_layout.addWidget(QLabel("Video mode options coming soon."))

        self.paths_layout.addWidget(paths, alignment=Qt.AlignmentFlag.AlignTop)

        self.path_list = SessionOverview(self)
        # self.path_list = PathList(self)
        self.paths_layout.addWidget(self.path_list)

        self.path_list.rebuild()

        # Buttons
        self.button_load = QPushButton("Load data")
        self.button_save = QPushButton("Save results")
        self.paths_layout.addWidget(self.button_load)
        self.paths_layout.addWidget(self.button_save)

        self.checkbox_auto_advance = QCheckBox("Auto-advance to next cluster")
        self.checkbox_skip_processed_side = QCheckBox("Skip processed in navigation")
        self.paths_layout.addWidget(self.checkbox_auto_advance)
        self.paths_layout.addWidget(self.checkbox_skip_processed_side)

        self.button_load.clicked.connect(self.on_load_clicked)

    def change_app_mode(self):
        self.build_app_mode_menu()

    def _build_form_data_paths_tracking(self) -> QFormLayout:

        form = QFormLayout()

        for name, info in paths.items():
            if "tracking" not in info["mode"]:
                continue
            self.paths[name] = {}

            self.paths[name]["edit"] = QLineEdit(
                self.defaults[f"{name}_{info['type']}"]
            )
            self.paths[name]["button"] = QPushButton("...")
            self.paths[name]["button"].setFixedWidth(30)

            entry_layout = QHBoxLayout()
            entry_layout.addWidget(self.paths[name]["edit"])
            entry_layout.addWidget(self.paths[name]["button"])
            form.addRow(f"{info['label']}:", entry_layout)

        ## Button connections
        self.paths["root"]["button"].clicked.connect(
            lambda: self.choose_path(
                pick_dir=True,
                edit_line=self.paths["root"]["edit"],
                display_text="Select root folder",
            )
        )

        self.paths["model"]["button"].clicked.connect(
            lambda: self.choose_path(
                pick_dir=False,
                init_path=self.root_folder,
                only_tail=True,
                edit_line=self.paths["model"]["edit"],
                display_text="Select model file",
            )
        )
        self.paths["registration"]["button"].clicked.connect(
            lambda: self.choose_path(
                pick_dir=False,
                init_path=self.root_folder,
                only_tail=True,
                edit_line=self.paths["registration"]["edit"],
                display_text="Select registration file",
            )
        )

        return form

    def _build_form_data_paths_single(self) -> QFormLayout:
        form = QFormLayout()

        self.paths = {}

        for name, info in paths.items():
            if "single" not in info["mode"]:
                continue
            self.paths[name] = {}

            self.paths[name]["edit"] = QLineEdit(
                self.defaults[f"{name}_{info['type']}"]
            )
            self.paths[name]["button"] = QPushButton("...")
            self.paths[name]["button"].setFixedWidth(30)

            entry_layout = QHBoxLayout()
            entry_layout.addWidget(self.paths[name]["edit"])
            entry_layout.addWidget(self.paths[name]["button"])
            form.addRow(f"{info['label']}:", entry_layout)

        ## Button connections
        self.paths["root"]["button"].clicked.connect(
            lambda: self.choose_path(
                pick_dir=True,
                edit_line=self.paths["root"]["edit"],
                display_text="Select root folder",
            )
        )
        # self.paths["session"]["button"].clicked.connect(
        #     lambda: self.choose_path(
        #         pick_dir=True,
        #         init_path=self.root_folder,
        #         only_tail=True,
        #         edit_line=self.paths["session"]["edit"],
        #         display_text="Select session folder",
        #     )
        # )

        self.paths["results"]["button"].clicked.connect(
            lambda: self.choose_path(
                pick_dir=False,
                init_path=self.root_folder,
                only_tail=True,
                edit_line=self.paths["results"]["edit"],
                display_text="Select results file",
            )
        )
        # self.paths["footprints"]["button"].clicked.connect(
        #     lambda: self.choose_path(
        #         pick_dir=False,
        #         init_path=self.session_folder,
        #         only_tail=True,
        #         edit_line=self.paths["footprints"]["edit"],
        #         display_text="Select footprints file",
        #     )
        # )

        # self.paths["field_results"]["button"].clicked.connect(
        #     lambda: self.on_choose_field(self.results_file, self.paths["field_results"]["edit"])
        # )
        # self.paths["field_footprints"]["button"].clicked.connect(
        #     lambda: self.on_choose_field(
        #         self.footprints_path, self.paths["field_footprints"]["edit"]
        #     )
        # )
        # ## toggle enable etc
        # toggle_enable(
        #     [
        #         self.path_session_edit,
        #         self.path_session_button,
        #         self.path_results_edit,
        #         self.path_results_button,
        #         # self.field_background_edit,
        #         # self.field_background_button,
        #         # self.path_footprints_edit,
        #         # self.path_footprints_button,
        #         # self.field_footprints_edit,
        #         # self.field_footprints_button,
        #     ],
        #     False,
        # )

        ## logic to enable/disable based on existing paths
        # self.paths["root"]["edit"].textChanged.connect(
        #     lambda: toggle_enable(
        #         [
        #             self.paths["session"]["edit"],
        #             self.paths["session"]["button"],
        #         ],
        #         Path(self.root_folder).is_dir(),
        #     )
        # )

        # self.paths["session"]["edit"].textChanged.connect(
        #     lambda: toggle_enable(
        #         [
        #             self.paths["results"]["edit"],
        #             self.paths["results"]["button"],
        #         ],
        #         Path(self.session_folder).is_dir(),
        #     )
        # )

        # self.paths["results"]["edit"].textChanged.connect(
        #     lambda: toggle_enable(
        #         [
        #             self.paths["field_results"]["edit"],
        #             self.paths["field_results"]["button"],
        #         ],
        #         Path(self.results_file).is_file(),
        #     )
        # )

        # self.paths["footprints"]["edit"].textChanged.connect(
        #     lambda: toggle_enable(
        #         [
        #             self.paths["field_footprints"]["edit"],
        #             self.paths["field_footprints"]["button"],
        #         ],
        #         Path(self.footprints_path).is_file(),
        #     )
        # )

        return form

    def choose_path(
        self,
        pick_dir: bool = False,
        init_path: str = "",
        only_tail: bool = False,
        edit_line: QLineEdit | None = None,
        display_text: str = "Select file",
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

        if path and edit_line is not None:
            # print(f"Chosen path: {path}, only tail: {only_tail}")
            edit_line.setText(
                str(Path(path).relative_to(init_path)) if only_tail else path
            )  # fill QLineEdit
        elif path:
            return str(Path(path).relative_to(init_path)) if only_tail else path

    @property
    def root_folder(self) -> str:
        return self.paths["root"]["edit"].text().strip()

    # @property
    # def session_folder(self) -> str:
    #     if (
    #         not self.root_folder
    #         or "session" not in self.paths
    #         or not isValid(self.paths["session"]["edit"])
    #     ):
    #         return ""
    #     # def _get_session_folder(self) -> str:
    #     # root_folder = self.path_root_edit.text().strip()
    #     session_folder = self.paths["session"]["edit"].text().strip()
    #     if not session_folder:
    #         return ""
    #     return str(Path(self.root_folder) / session_folder)

    @property
    def results_file(self) -> str:
        if (
            not self.root_folder
            or "results" not in self.paths
            or not isValid(self.paths["results"]["edit"])
        ):
            return ""
        results_file = self.paths["results"]["edit"].text().strip()
        if not results_file:
            return ""
        return str(Path(self.root_folder) / results_file)

    @property
    def model_file(self) -> str:
        if (
            not self.root_folder
            or "model" not in self.paths
            or not isValid(self.paths["model"]["edit"])
        ):
            return ""
        model_file = self.paths["model"]["edit"].text().strip()
        if not model_file:
            return ""
        return str(Path(self.root_folder) / model_file)

    @property
    def registration_file(self) -> str:
        if (
            not self.root_folder
            or "registration" not in self.paths
            or not isValid(self.paths["registration"]["edit"])
        ):
            return ""
        registration_file = self.paths["registration"]["edit"].text().strip()
        if not registration_file:
            return ""
        return str(Path(self.root_folder) / registration_file)

    # @property
    # def footprints_field(self) -> str:
    #     return self.field_footprints_edit.text().strip()

    def on_choose_field(self, path: str, edit_line: QLineEdit):
        # path = edit_line.text().strip()
        if not path:
            QMessageBox.warning(self, "No file", "Please select a results file first.")
            return

        try:
            fields = list_file_fields(path)  # List[FieldInfo]
        except Exception as e:
            QMessageBox.critical(
                self, "Error", f"Could not read fields from file:\n{e}"
            )
            return

        if not fields:
            QMessageBox.information(
                self, "No fields", "No fields/datasets found in this file."
            )
            return

        selected_name = FieldSelectDialog.get_field(
            fields,
            title="Select field for A",
            parent=self,
        )
        if selected_name is not None:

            # selected = FieldSelectDialog.get_field(fields, title="Select data field", parent=self)
            # if selected is not None:
            edit_line.setText(selected_name)

    ### ------------------------------------------###
    ###    Logic for saving/restoring settings    ###
    ### ------------------------------------------###
    def _restore_settings(self):
        # print("Restoring settings...")
        self.defaults = {
            f"{name}_{info['type']}": self.settings.value(
                f"paths/{name}_{info['type']}", "", type=str
            )
            for name, info in paths.items()
        }

    def _save_settings(self):

        for name, info in paths.items():
            key = f"{name}_{info['type']}"
            # try:
            ## only relative structure is stored
            # relative = paths[info["root"]]
            path_key = getattr(self, key, None)
            if path_key:
                if info["root"]:
                    relative_path = Path(path_key).relative_to(
                        getattr(self, info["root"] + "_folder")
                    )
                else:
                    relative_path = path_key
                self.settings.setValue(f"paths/{key}", str(relative_path))
            # except:
            #     # if mode was not selected, variables wont be set, so just skip
            #     pass

        self.settings.sync()  # flush to disk


# def toggle_enable(widgets: List[QWidget], enabled: bool):
#     for w in widgets:
#         w.setEnabled(enabled)

#         if not enabled and isinstance(w, (QLineEdit, QLabel)):
#             w.setText("")
