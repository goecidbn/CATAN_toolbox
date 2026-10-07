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
from PySide6.QtCore import QSettings, QThreadPool, Qt, Signal, QSignalBlocker, QSize
from PySide6.QtGui import QAction
from shiboken6 import isValid

from pathlib import Path

from catan.core.changes import DataChange
from catan.gui.structures import data, state
from catan.gui.background_tasks.TaskManager import TaskBatch
from catan.gui.resources.get_icon import get_fa_icon

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

    def process_data_from_session(self, session_id: int):
        row = self.session_list.load_row

        # The folder button must actually request loading, independently
        # of whether automatic loading is enabled globally.
        actions = set(row.registration_action_selector.actions)
        actions.add("load_data")

        row._on_sessions_registered(
            [session_id],
            actions=actions,
            batch=TaskBatch(),
        )

    def _on_data_changed(self, event: DataChange):
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

    def build_app_mode_menu(self):
        paths_menu = QWidget()
        self.paths_layout = QVBoxLayout(paths_menu)
        self.paths_layout.setContentsMargins(0, 0, 0, 0)

        frame = QFrame()
        frame.setFrameShape(QFrame.Shape.StyledPanel)

        form = QFormLayout(frame)
        form.setContentsMargins(8, 8, 8, 8)
        form.setHorizontalSpacing(6)
        form.setVerticalSpacing(5)
        form.setRowWrapPolicy(QFormLayout.RowWrapPolicy.DontWrapRows)
        form.setFieldGrowthPolicy(QFormLayout.FieldGrowthPolicy.AllNonFixedFieldsGrow)
        self.form = form

        def source_icon(icon_name, tooltip):
            label = QLabel()
            label.setFixedSize(24, 26)
            label.setAlignment(Qt.AlignmentFlag.AlignCenter)
            label.setPixmap(get_fa_icon(icon_name).pixmap(QSize(17, 17)))
            label.setToolTip(tooltip)
            return label

        self.loader = {}

        form.addRow(
            source_icon("chart-column", "Model"),
            self.build_load_options(
                "model",
                list(self.data.available_models),
                add_options=selector_options["model"],
            ),
        )

        self.config_constructor = FieldConfigConstructor(self, self.data.assignments)

        form.addRow(
            source_icon("layer-group", "Assignments"),
            self.build_load_options(
                "assignments",
                list(self.data.available_assignments),
                add_options=selector_options["assignments"],
                add_widgets=[self.config_constructor.toggle_config_options],
            ),
        )

        # Configuration content belongs to its popup, not this form.
        self.paths_layout.addWidget(frame)
        self.update_buttons()
        return paths_menu

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
        add_options=None,
        add_widgets=None,
    ):
        layout = QHBoxLayout()
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(4)

        self.loader[key] = {"options": list(options)}

        selector = QComboBox()
        selector.setMinimumWidth(40)
        selector.setMinimumContentsLength(1)
        selector.setSizeAdjustPolicy(
            QComboBox.SizeAdjustPolicy.AdjustToMinimumContentsLengthWithIcon
        )
        selector.setSizePolicy(
            QSizePolicy.Policy.Expanding,
            QSizePolicy.Policy.Fixed,
        )

        self.loader[key]["selector"] = selector
        self.rebuild_selector(
            key,
            options=list(options),
            add_options=add_options,
        )

        selector.currentTextChanged.connect(
            lambda text: self._on_load_option_changed(key, text)
        )
        selector.currentTextChanged.connect(selector.setToolTip)
        selector.setToolTip(selector.currentText())

        layout.addWidget(selector, 1)

        execute = QToolButton(self)
        execute.setAutoRaise(True)
        execute.setFixedSize(36, 26)
        execute.setPopupMode(QToolButton.ToolButtonPopupMode.MenuButtonPopup)
        set_button_icon(
            execute,
            "play" if key == "model" else "folder-open",
            tooltip=(
                "Process pending model counts and fit"
                if key == "model"
                else "Load assignments"
            ),
        )
        execute.setIconSize(QSize(16, 16))
        self._add_process_menu(execute, key)
        self.loader[key]["button_execute"] = execute
        layout.addWidget(execute)

        save = QToolButton(self)
        save.setAutoRaise(True)
        save.setFixedSize(26, 26)
        set_button_icon(save, "floppy-disk", tooltip=f"Save {key} data")
        save.setIconSize(QSize(16, 16))
        save.setEnabled(False)
        save.clicked.connect(lambda checked=False: self.save_data(key))
        self.loader[key]["button_save"] = save
        layout.addWidget(save)

        # Both rows reserve exactly the same configuration-button space.
        config_slot = QWidget()
        config_slot.setFixedSize(28, 26)
        config_layout = QHBoxLayout(config_slot)
        config_layout.setContentsMargins(0, 0, 0, 0)

        for widget in add_widgets or ():
            widget.setFixedSize(28, 26)
            widget.setIconSize(QSize(16, 16))
            config_layout.addWidget(widget)

        layout.addWidget(config_slot)

        def execute_clicked():
            if key == "model":
                self.data.queue_process_model(mode="pending")
                return

            assignments = self.data.assignments
            if assignments is None:
                return

            if assignments.path and not assignments.status["loaded"]:
                self._load_assignments_from_source()
            else:
                self.data.queue_process_assignments(mode="pending")

        execute.clicked.connect(execute_clicked)
        execute.setEnabled(False)

        return layout

    def save_data(self, key):
        if key not in {"model", "assignments"}:
            raise ValueError(f"Unsupported result type: {key}")

        save_path = choose_path(
            self,
            init_path=str(self.data.root),
            display_text=f"Save {key}",
            only_existing=False,
            default_suffix="hdf5",
            file_filters=[
                ("HDF5 file", ("*.hdf5", "*.h5")),
                ("MATLAB file", ("*.mat",)),
                ("NumPy file", ("*.npz",)),
            ],
            state=self.state,
            default_filename=f"catan_{key}.hdf5",
        )

        if save_path is not None:
            self.data.queue_save_result(key, save_path)

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
                    self._refresh_source_selector(key)
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

        if not ok:
            self._refresh_source_selector(key)
            return

        if isinstance(name, str) and load_path is not None:
            name = name.strip()

            # Keep showing the currently active source while loading.
            self._refresh_source_selector(key)

            if not name:
                return

            self.data.queue_import_source(
                key,
                load_path,
                name,
                on_loaded=lambda: self._on_source_registered(key),
            )
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

    def _show_assignments_configuration(self):
        constructor = self.config_constructor
        constructor.update_source(self.data.assignments)
        constructor.toggle_config_options.set_expanded(True)
        constructor.expanded_changed.emit()
        self.update_buttons()

    def _load_assignments_from_source(self):
        self.data.queue_load_assignments(
            on_config_required=self._show_assignments_configuration,
        )

    def _on_source_registered(self, key):
        self._refresh_source_selector(key)

        if key != "assignments":
            return

        source = self.data.assignments
        problems = getattr(source, "_load_config_problems", [])

        if problems:
            source.status["loading_possible"] = False
            self._show_assignments_configuration()

            self.state.issue(
                "info",
                "Choose or repair the assignments configuration",
                "\n\n".join(problems)
                + "\n\nSelect a matching preset or configure the IDs field. "
                "Then click the assignments open-folder button to load.",
                parent=self.window(),
            )
            return

        self._load_assignments_from_source()

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

        for controls in self.loader.values():
            controls["button_execute"].setIconSize(QSize(16, 16))
            controls["button_save"].setIconSize(QSize(16, 16))

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

    def _refresh_source_selector(self, key):
        if not isValid(self):
            return

        if key == "model":
            names = list(self.data.available_models)
            current = self.data.current_model_name
        else:
            names = list(self.data.available_assignments)
            current = self.data.current_assignments

        self.rebuild_selector(
            key,
            names,
            add_options=selector_options[key],
        )

        selector = self.loader[key]["selector"]

        with QSignalBlocker(selector):
            index = selector.findText(current) if current is not None else -1
            selector.setCurrentIndex(index)

        if key == "assignments":
            self.config_constructor.update_source(self.data.assignments)

        self.update_buttons()

    ### ------------------------------------------###
    ###    Logic for saving/restoring settings    ###
    ### ------------------------------------------###
    def _restore_settings(self):
        self.data.root = self.settings.value(f"paths/root_folder", "", type=str)
