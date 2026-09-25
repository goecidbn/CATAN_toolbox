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
    QSizePolicy,
    QComboBox,
    QToolButton,
    QSpinBox,
    QWidgetAction,
    QCheckBox,
)
from PySide6.QtCore import QSettings, QThreadPool, Qt, Signal
from PySide6.QtGui import QAction
from shiboken6 import isValid

from pathlib import Path

from catan.gui.structures import data, state

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

        # self.settings = QSettings()
        self.settings = parent.settings
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

        self.session_list = session_overview.SessionOverview(self)
        layout.addWidget(self.session_list, stretch=1)

        layout.addWidget(self.build_app_mode_menu())

        layout.addStretch()

        self.task_overview = TaskOverviewDisplay(
            self.state.tasks,
        )
        layout.addWidget(self.task_overview)
        layout.addWidget(ResourceMonitor(parent=self))

        self.session_list.load_requested.connect(self.process_data_from_session)

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
        row = self.session_list.load_row
        row._on_sessions_registered(
            [session.id for session in self.data.sessions],
            actions=row.registration_action_selector.actions(),
        )

    def process_data_from_session(self, session_id: int):
        row = self.session_list.load_row
        row._on_sessions_registered(
            [session_id],
            actions=row.registration_action_selector.actions(),
        )

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
        model_fit_possible = local_model and sessions_loaded
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

        self.checkbox_correct_rotation = QCheckBox("Correct session rotation")
        self.checkbox_correct_rotation.setChecked(self.data.correct_rotation)
        self.checkbox_correct_rotation.setToolTip(
            "Search for rotation as well as translation during new "
            "alignment calculations. Slower; existing alignments "
            "are unchanged."
        )

        def set_rotation_correction(enabled):
            self.data.correct_rotation = bool(enabled)
            self.settings.setValue("alignment/correct_rotation", bool(enabled))

        self.checkbox_correct_rotation.toggled.connect(set_rotation_correction)
        form.addRow(self.checkbox_correct_rotation)

        self.paths_layout.addWidget(formFrame, alignment=Qt.AlignmentFlag.AlignTop)

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
            self.settings.setValue(f"paths/root_folder", str(self.data.root))

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

    def _add_process_menu(self, button, key):
        menu = QMenu(button)
        button.setMenu(menu)

        rerun_all = menu.addAction("Rerun all")
        session_rows = []

        def dispatch(*, session_id=None, from_session_id=None):
            menu.close()

            kwargs = {"mode": "force"}
            if session_id is not None:
                kwargs["session_ids"] = [session_id]
            elif from_session_id is not None:
                kwargs["from_session_id"] = from_session_id

            if key == "model":
                self.data.queue_process_model(**kwargs)
            else:
                self.data.queue_process_assignments(**kwargs)

        def add_session_row(label, *, single):
            row = QWidget(menu)
            layout = QHBoxLayout(row)
            layout.setContentsMargins(8, 6, 8, 6)
            layout.addWidget(QLabel(label, row))

            session_index = QSpinBox(row)
            session_index.setMinimum(0)
            session_index.setToolTip("Session ID; numbering starts at 0.")
            layout.addWidget(session_index)

            run_button = QPushButton("Run", row)
            layout.addWidget(run_button)

            if key == "assignments":
                run_button.setToolTip(
                    "Also rebuilds already-registered downstream "
                    "sessions that depend on this session."
                )
            else:
                run_button.setToolTip(
                    "Updates counts associated with the selected "
                    "session(s), then refits the model when possible."
                )

            action = QWidgetAction(menu)
            action.setDefaultWidget(row)
            menu.addAction(action)

            def run():
                if single:
                    dispatch(session_id=session_index.value())
                else:
                    dispatch(from_session_id=session_index.value())

            run_button.clicked.connect(run)
            session_rows.append((action, row, session_index))

        add_session_row("Rerun from session", single=False)
        add_session_row("Rerun session", single=True)

        def refresh_menu():
            n_sessions = len(self.data.sessions)

            if key == "model":
                model = self.data.model
                available = model is not None and not model.loaded and n_sessions > 0
            else:
                assignments = self.data.assignments
                needs_loading = (
                    assignments is not None
                    and bool(assignments.path)
                    and not assignments.status["loaded"]
                )
                available = (
                    assignments is not None and not needs_loading and n_sessions > 0
                )

            rerun_all.setEnabled(available)

            for action, row, session_index in session_rows:
                session_index.setMaximum(max(0, n_sessions - 1))
                action.setEnabled(available)
                row.setEnabled(available)

        rerun_all.triggered.connect(lambda: dispatch())
        menu.aboutToShow.connect(refresh_menu)
        refresh_menu()

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
        if key in ("model", "assignments"):
            button = QToolButton(self)
            button.setFixedSize(46, 28)
            button.setPopupMode(QToolButton.ToolButtonPopupMode.MenuButtonPopup)
            set_button_icon(
                button,
                "folder-open",
                tooltip=f"Load {key} data",
            )
            self._add_process_menu(button, key)
        else:
            button = make_icon_button(
                "folder-open",
                tooltip=f"Load {key} data",
                size=28,
                icon_size=22,
            )
            button.setFixedWidth(25)

        self.loader[key]["button_execute"] = button
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
                tooltip="Process pending model counts and fit",
            )

            def on_button_click():
                self.data.queue_process_model(mode="pending")

        if key == "assignments":

            def on_button_click():
                if self.data.assignments is None:
                    return

                if (
                    self.data.assignments
                    and self.data.assignments.path
                    and not self.data.assignments.status["loaded"]
                ):
                    self.data.load_assignments()
                else:
                    self.data.queue_process_assignments(mode="pending")

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

        name = None
        ok = False
        if opt in selector_options[key]:
            load_path = None
            if opt == "Load ...":
                load_path = choose_path(
                    self,
                    pick_dir=False,
                    init_path=self.data.root,
                    display_text=f"Select {key} file",
                    only_existing=True,
                )
                if not load_path:
                    self.loader[key]["selector"].setCurrentIndex(0)
                    return

            name, ok = QInputDialog.getText(
                self,
                f"{key.capitalize()} name",
                f"Enter a name for the {key}:",
                text=Path(load_path).stem if load_path else "",
            )
        else:
            if key == "model":
                self.data.change_model(opt)
            elif key == "assignments":
                self.data.change_assignments(opt)

            return

        if ok:
            if key == "model":
                if isinstance(name, str):
                    self.data.add_model(name, load_path)

                available_names = self.data.available_models
            elif key == "assignments":

                if isinstance(name, str):
                    self.data.register_assignments(load_path, name)

                    self.config_constructor.update_source(self.data.assignments)
                available_names = self.data.available_assignments
            else:
                return

            self.rebuild_selector(
                key,
                available_names,
                add_options=selector_options[key],
            )

            if name in available_names:
                index = available_names.index(name)
            else:
                ## if adding failed for whatever reason
                index = 0

            self.loader[key]["selector"].setCurrentIndex(index)

        self.update_buttons()

    def update_buttons(self):

        ## update of assignments button
        loading = False
        if (
            self.data.assignments
            and self.data.assignments.path
            and not self.data.assignments.status["loaded"]
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
                tooltip="Process missing or outdated neuron registrations",
            )

        self.loader["assignments"]["button_execute"].setEnabled(
            self.data.assignments is not None and (loading or bool(self.data.sessions))
        )

        # if self.data.assignments is None:
        #     enable_button = False
        # else:
        #     enable_button = loading
        #     enable_button |= not all(
        #         [
        #             not self.data.session_assigned(s.id) and s.status["spatial_loaded"]
        #             for s in self.data.sessions
        #         ]
        #     )
        # self.loader["assignments"]["button_execute"].setEnabled(enable_button)

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
        self.data.root = self.settings.value(f"paths/root_folder", "", type=str)
