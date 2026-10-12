from typing import Dict, Optional, Tuple, List
import importlib
import numpy as np

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
    QDialogButtonBox,
    QDialog,
    QMessageBox,
)

from PySide6.QtCore import QSettings, QThreadPool, Qt, Signal, QSignalBlocker, QSize
from PySide6.QtGui import QAction
from shiboken6 import isValid

from pathlib import Path

from catan.core.changes import DataChange
from catan.gui.structures import data, state
from catan.gui.background_tasks.TaskManager import TaskBatch
from catan.gui.resources.get_icon import get_fa_icon
from catan.gui.data.assignment_colors import assignment_label
from catan.core.changes import ChangeKind as C

from catan.tracking.structures import ReviewStatus
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
    "model": [
        "Load ...",
        "Remove current…",
    ],
    "assignments": [
        "Create new",
        "Create named copy…",
        "Load ...",
        "Remove current…",
    ],
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

        self._update_review_counts()

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

        if event.has(C.ASSIGNMENT_SET):
            self._refresh_source_selector("assignments")

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

        self.review_counts_label = QLabel()
        self.review_counts_label.setWordWrap(True)
        self.review_counts_label.setToolTip(
            "Review status of included neurons in the active assignments."
        )
        self.review_counts_label.setStyleSheet("font-size: 11px; padding: 2px 0;")

        form.addRow("", self.review_counts_label)
        self._update_review_counts()

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

        selector.currentIndexChanged.connect(
            lambda index: self._on_load_option_changed(
                key,
                selector.itemData(index),
            )
        )

        # selector.currentTextChanged.connect(
        #     lambda text: self._on_load_option_changed(key, text)
        # )
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

    def _update_review_counts(self):
        label = getattr(self, "review_counts_label", None)

        if label is None:
            return

        assignments = self.data.assignments

        if assignments is None:
            statuses = np.empty(0, dtype=np.uint8)
        else:
            statuses = np.asarray(assignments.review_status).reshape(-1)
            included = getattr(assignments.union, "included", None)

            if included is not None:
                included = np.asarray(included, dtype=bool)

                if included.shape == statuses.shape:
                    statuses = statuses[included]

        counts = {
            status: int(np.count_nonzero(statuses == status))
            for status in (
                ReviewStatus.REVIEWED,
                ReviewStatus.UNCERTAIN,
                ReviewStatus.PENDING,
            )
        }

        manipulation_counts = {
            "split": 0,
            "merge": 0,
            "reassign": 0,
        }

        category_for_type = {
            "split": "split",
            "merge": "merge",
            "neuron_merge": "merge",
            "reassign": "reassign",
        }

        if assignments is not None:
            for record in assignments.manipulations.values():
                category = category_for_type.get(record["type"])

                if category is not None:
                    manipulation_counts[category] += 1

        excluded_footprints = 0

        for session in self.data.sessions:
            included = getattr(session, "included", None)

            if included is not None:
                included = np.asarray(included, dtype=bool)
                excluded_footprints += int(np.count_nonzero(~included))

        label.setText(
            f"Reviewed: {counts[ReviewStatus.REVIEWED]} · "
            f"Uncertain: {counts[ReviewStatus.UNCERTAIN]} · "
            f"Pending: {counts[ReviewStatus.PENDING]}\n"
            f"Splits: {manipulation_counts['split']} · "
            f"Merges: {manipulation_counts['merge']} · "
            f"Reassignments: {manipulation_counts['reassign']} · "
            f"Excluded footprints: {excluded_footprints}"
        )

        label.setToolTip(
            "Review counts refer to included neurons in the active "
            "assignments.\n\n"
            "Manipulation counts refer to recorded operations in the "
            "active assignments, including loaded history. Each operation "
            "counts once. Merges include footprint and neuron merges.\n\n"
            "Excluded footprints counts all currently excluded footprints "
            "across the loaded sessions, including original footprints "
            "replaced by splitting or merging."
        )

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

        if opt == "Remove current…":
            name = (
                self.data.current_model_name
                if key == "model"
                else self.data.current_assignments
            )
            source = getattr(self.data, key)

            self._refresh_source_selector(key)

            if source is None or name is None:
                return

            explanation = (
                "Its contents will be cleared, leaving an empty Working entry."
                if name == "local"
                else "It will be removed from the current CATAN workspace."
            )

            answer = QMessageBox.question(
                self.window(),
                f"Remove {key}?",
                f"Remove {assignment_label(name)}?\n\n"
                f"{explanation}\n"
                "Files on disk will not be deleted.",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                QMessageBox.StandardButton.No,
            )

            if answer == QMessageBox.StandardButton.Yes:
                self.data.remove_loaded_source(
                    key,
                    name,
                    expected_source=source,
                    on_removed=lambda: self._refresh_source_selector(key),
                )

            return

        use_working = False
        replace_empty_model = False

        if key == "assignments" and opt == "Create named copy…":
            source_name = self.data.current_assignments
            self._refresh_source_selector(key)

            name, accepted = QInputDialog.getText(
                self,
                "Copy assignments",
                "Name for the independent assignments copy:",
                text=f"{source_name} copy",
            )

            if accepted:
                self.data.queue_copy_assignments(
                    name,
                    on_copied=lambda: self._refresh_source_selector("assignments"),
                )

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

            if key == "assignments" and load_path is not None:
                name, ok, use_working = self._ask_assignments_import_name(load_path)

            elif key == "model" and load_path is not None:
                name, ok, replace_empty_model = self._ask_model_import_name(load_path)

            else:
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
                self._refresh_source_selector(key)

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
                use_as_working=use_working,
                replace_empty_model=replace_empty_model,
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

    def _ask_model_import_name(self, path):
        dialog = QDialog(self)
        dialog.setWindowTitle("Load model")

        layout = QVBoxLayout(dialog)

        name_edit = QLineEdit(Path(path).stem)
        layout.addWidget(QLabel("Model name"))
        layout.addWidget(name_edit)

        target_name = self.data.current_model_name

        replace_current = QCheckBox("Replace current empty model")
        replace_current.setEnabled(self.data.can_replace_empty_model(target_name))
        replace_current.setToolTip(
            "Replace the current model only if it has no loaded model, "
            "fit, or accumulated counts."
        )
        replace_current.toggled.connect(name_edit.setDisabled)
        replace_current.setChecked(replace_current.isEnabled())
        layout.addWidget(replace_current)

        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel
        )
        buttons.accepted.connect(dialog.accept)
        buttons.rejected.connect(dialog.reject)
        layout.addWidget(buttons)

        accepted = dialog.exec() == QDialog.DialogCode.Accepted
        replace = replace_current.isChecked()

        name = target_name if replace else name_edit.text().strip()
        return name, accepted, replace

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

    def _ask_assignments_import_name(self, path):
        dialog = QDialog(self)
        dialog.setWindowTitle("Load assignments")

        layout = QVBoxLayout(dialog)

        name_edit = QLineEdit(Path(path).stem)
        layout.addWidget(QLabel("Assignments name"))
        layout.addWidget(name_edit)

        use_working = QCheckBox("Replace empty Working assignments")
        use_working.setEnabled(self.data.can_use_working_assignments())
        use_working.setToolTip(
            "Replace the empty Working assignments after successful "
            "loading. Uncheck to load a separate named dataset."
        )
        use_working.toggled.connect(name_edit.setDisabled)
        use_working.setChecked(use_working.isEnabled())

        layout.addWidget(use_working)

        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel
        )
        buttons.accepted.connect(dialog.accept)
        buttons.rejected.connect(dialog.reject)
        layout.addWidget(buttons)

        accepted = dialog.exec() == QDialog.DialogCode.Accepted

        replace_working = use_working.isChecked()
        name = name_edit.text().strip()

        if accepted and replace_working:
            # Loading first registers a temporary named dataset.
            # It becomes Working only after successful validation.
            base = Path(path).stem or "Imported assignments"
            name = base
            number = 2

            existing = set(self.data.available_assignments)

            while name in existing:
                name = f"{base} ({number})"
                number += 1

        return name, accepted, replace_working

    def _on_source_registered(self, key):
        self._refresh_source_selector(key)

        if key != "assignments":
            return

        source = self.data.assignments
        problems = getattr(source, "_load_config_problems", [])

        if problems:
            source.status["loading_possible"] = False

            self.state.issue(
                "info",
                "Choose or repair the assignments configuration",
                "\n\n".join(problems)
                + "\n\nSelect a matching preset or configure the IDs field. "
                "Then click the assignments open-folder button to load.",
                parent=self.window(),
            )

            # Open AFTER the message box closes.
            # Its nested event loop may have changed the active source.
            if isValid(self) and self.data.assignments is source:
                self._show_assignments_configuration()

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

        for option in options:
            selector.addItem(assignment_label(option), option)
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
            index = selector.findData(current) if current is not None else -1
            selector.setCurrentIndex(index)

        if key == "assignments":
            self._update_review_counts()
            self.config_constructor.update_source(self.data.assignments)

        self.update_buttons()

    ### ------------------------------------------###
    ###    Logic for saving/restoring settings    ###
    ### ------------------------------------------###
    def _restore_settings(self):
        self.data.root = self.settings.value(f"paths/root_folder", "", type=str)
