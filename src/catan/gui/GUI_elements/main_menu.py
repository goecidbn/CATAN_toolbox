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
    QGridLayout,
    QLabel,
    QFrame,
    QVBoxLayout,
    QSizePolicy,
    QComboBox,
    QToolButton,
    QSpinBox,
    QDoubleSpinBox,
    QWidgetAction,
    QCheckBox,
    QDialogButtonBox,
    QDialog,
    QMessageBox,
)

from PySide6.QtCore import QSettings, QThreadPool, Qt, Signal, QSignalBlocker, QSize
from PySide6.QtGui import QAction
from shiboken6 import isValid
from catan.gui.GUI_elements.utils.menu_creation import fit_message_box_buttons

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
from catan.gui.GUI_elements.utils.paths import display_path
from catan.gui.GUI_elements.utils.elided_label import ElidedLabel
import qtawesome as qta

# Source actions (row buttons, "+ from file"); the selectors list only the
# loaded models / assignment sets.
source_actions = {
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
selector_options = {"model": [], "assignments": []}


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
        self.session_list = session_overview.SessionOverview(self)
        load_row = self.session_list.load_row
        # "+ from file": a project opens the project; assignments / a model
        # are imported into their sections.
        load_row.projectFileSelected.connect(self.open_project)
        load_row.sourceFileSelected.connect(self.import_source_file)
        load_row.projectSaveRequested.connect(lambda save_as: self.save_project(save_as=save_as))
        load_row.assignmentsExportRequested.connect(lambda: self.save_data("assignments"))
        load_row.modelSaveRequested.connect(lambda: self.save_data("model"))

        # Three sections: loading (top, with the root folder), sessions
        # (status, alignment limits, list), then model and assignments.
        self.session_list.layout().removeWidget(load_row)
        load_row.setParent(self)
        load_row.set_root_selector(self.build_root_selector())
        layout.addWidget(load_row)

        sessions_frame = QFrame()
        sessions_frame.setFrameShape(QFrame.Shape.StyledPanel)
        sessions_layout = QVBoxLayout(sessions_frame)
        sessions_layout.setContentsMargins(6, 6, 6, 6)
        sessions_layout.setSpacing(4)
        for widget in load_row.take_session_widgets():
            widget.setParent(sessions_frame)
            sessions_layout.addWidget(widget)
        sessions_layout.addWidget(self.session_list, 1)
        layout.addWidget(sessions_frame, stretch=1)

        layout.addWidget(self.build_app_mode_menu())

        layout.addStretch()

        self.task_overview = TaskOverviewDisplay(
            self.state.tasks,
        )
        layout.addWidget(self.task_overview)
        layout.addWidget(ResourceMonitor(parent=self))

        self.session_list.load_requested.connect(self.process_data_from_session)

        self.state.data_changed.connect(self._on_data_changed)

        # "Go" buttons show a spinner while their work is scheduled/running.
        tasks = self.state.tasks
        for signal in (
            tasks.queue_changed,
            tasks.task_started,
            tasks.task_finished,
            tasks.task_cancelled,
            tasks.task_failed,
            tasks.scheduling_settled,
        ):
            signal.connect(self._refresh_execute_buttons)

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
        model_fitted = self.data.model is not None and self.data.model.fitted
        self.loader["model"]["button_save"].setEnabled(model_fitted)

        self._refresh_execute_buttons()

        if self.data.assignments is None:
            return
        any_assigned = any(self.data.assignments.matched_status)
        # A project can be saved once there are sessions.
        self.loader["assignments"]["button_save"].setEnabled(
            sessions_loaded or any_assigned
        )
        self.update_buttons()

        if event.has(C.ASSIGNMENT_SET):
            self._refresh_source_selector("assignments")

    def build_app_mode_menu(self):
        paths_menu = QWidget()
        self.paths_layout = QVBoxLayout(paths_menu)
        self.paths_layout.setContentsMargins(0, 0, 0, 0)

        def section(spacing=5):
            frame = QFrame()
            frame.setFrameShape(QFrame.Shape.StyledPanel)
            form = QFormLayout(frame)
            form.setContentsMargins(8, 8, 8, 8)
            form.setHorizontalSpacing(6)
            form.setVerticalSpacing(spacing)
            form.setRowWrapPolicy(QFormLayout.RowWrapPolicy.DontWrapRows)
            form.setFieldGrowthPolicy(QFormLayout.FieldGrowthPolicy.AllNonFixedFieldsGrow)
            return frame, form

        # Model and assignments: one frame each.
        model_frame, model_form = section()
        frame, form = section()
        self.form = form

        def source_icon(icon_name, tooltip):
            label = QLabel()
            label.setFixedSize(24, 26)
            label.setAlignment(Qt.AlignmentFlag.AlignCenter)
            label.setPixmap(get_fa_icon(icon_name).pixmap(QSize(17, 17)))
            label.setToolTip(tooltip)
            return label

        self.loader = {}

        model_form.addRow(
            source_icon("chart-column", "Model"),
            self.build_load_options(
                "model",
                list(self.data.available_models),
                add_options=selector_options["model"],
            ),
        )

        self.config_constructor = FieldConfigConstructor(self, self.data.assignments)
        # The preset is chosen on loading; the configuration shows itself
        # only for a loading problem (show_problems).
        self.config_constructor.auto_hide = True
        self.config_constructor.use_dialog("Assignments load configuration", self)

        form.addRow(
            source_icon("layer-group", "Assignments"),
            self.build_load_options(
                "assignments",
                list(self.data.available_assignments),
                add_options=selector_options["assignments"],
            ),
        )

        form.addRow("", self.build_registration_parameters())

        # Two fixed lines (elided, full text in the tooltip): growing
        # numbers must not add lines and shift the menu.
        self.review_counts_label = ElidedLabel(lines=2)
        self.review_counts_label.setToolTip(
            "Review status of included neurons in the active assignments."
        )
        self.review_counts_label.setStyleSheet("font-size: 11px; padding: 2px 0;")

        form.addRow("", self.review_counts_label)
        self._update_review_counts()

        # Configuration content belongs to its popup, not this form.
        self.paths_layout.setSpacing(8)
        self.paths_layout.addWidget(model_frame)
        self.paths_layout.addWidget(frame)
        self.update_buttons()
        return paths_menu

    def build_registration_parameters(self) -> QWidget:
        """Parameters of neuron registration (stored in the settings and
        applied to Tracking.params)."""
        widget = QWidget()
        # Two lines (match / drop thresholds, union update), captions and
        # spin boxes in their own columns so captions never overlap them.
        layout = QGridLayout(widget)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setHorizontalSpacing(6)
        layout.setVerticalSpacing(3)
        positions = {
            "match_threshold": (0, 0),
            "competitor_threshold": (0, 2),
            "union_update_strength": (1, 0),
        }

        specs = (
            (
                "match_threshold",
                "Match p",
                0.3,
                (0.05, 0.95),
                "Match p ≥: minimum match probability for assigning a "
                "footprint to an existing neuron.",
            ),
            (
                "competitor_threshold",
                "Drop p",
                0.3,
                (0.05, 0.95),
                "Drop competitor p ≥: an unmatched footprint whose match probability with any "
                "neuron reaches this value (it lost the match to another "
                "footprint) is dropped instead of becoming a new neuron, "
                "to avoid confusing the two downstream.",
            ),
            (
                "union_update_strength",
                "Union a",
                0.5,
                (0.05, 1.0),
                "Union update strength: weight of a matched footprint in the union footprint: "
                "U ← (1 − a)·U + a·A, a = strength × match probability. "
                "Lower: more inertia (less pulled by a single bad match); "
                "higher: follows changes faster. Also used whenever the union "
                "is rebuilt (loading, manipulations).",
            ),
        )

        self.registration_spins = {}

        for key, label, default, (low, high), tooltip in specs:
            value = self.settings.value(f"registration/{key}", default, type=float)
            self.data.params[key] = float(value)

            spin = QDoubleSpinBox()
            spin.setDecimals(2)
            spin.setRange(low, high)
            spin.setSingleStep(0.05)
            spin.setValue(float(value))
            spin.setToolTip(tooltip)
            spin.setFixedWidth(66)

            def changed(value, key=key):
                self.data.params[key] = float(value)
                self.settings.setValue(f"registration/{key}", float(value))

            spin.valueChanged.connect(changed)

            caption = QLabel(label)
            caption.setToolTip(tooltip)
            # Never truncated: the row keeps its full width.
            caption.setMinimumWidth(caption.sizeHint().width())
            row, column = positions[key]
            layout.addWidget(caption, row, column)
            layout.addWidget(spin, row, column + 1)
            self.registration_spins[key] = spin

        layout.setColumnStretch(4, 1)
        return widget

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

        if key == "model":
            # Separate model files (the project save includes the model).
            menu.addSeparator()
            save_model = menu.addAction("Save model…", lambda: self.save_data("model"))

            def refresh_model_actions():
                model = self.data.model
                save_model.setEnabled(
                    model is not None and (model.fitted or any(model.counts.values()))
                )

            menu.aboutToShow.connect(refresh_model_actions)

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
        def describe(text):
            selector.setToolTip(
                f"Active model: {text}. Used for evidence, fitting and registration."
                if key == "model"
                else f"Active assignments: {text}. Used for navigation, "
                "selection, the panels and registration."
            )

        selector.currentTextChanged.connect(describe)
        describe(selector.currentText())

        layout.addWidget(selector, 1)

        execute = QToolButton(self)
        execute.setAutoRaise(True)
        execute.setFixedSize(36, 26)
        execute.setPopupMode(QToolButton.ToolButtonPopupMode.MenuButtonPopup)
        set_button_icon(
            execute,
            "play",
            tooltip=(
                "Process pending model counts and fit"
                if key == "model"
                else "Process missing or outdated neuron registrations"
            ),
        )
        execute.setIconSize(QSize(16, 16))
        self._add_process_menu(execute, key)
        self.loader[key]["button_execute"] = execute
        layout.addWidget(execute)

        save = QToolButton(self)
        save.setAutoRaise(True)
        save.setFixedSize(26, 26)
        save.setIconSize(QSize(16, 16))
        save.setEnabled(False)
        self.loader[key]["button_save"] = save
        # Saved with the project (load section's save button / Ctrl+S); the
        # model also via its process button's arrow menu.
        save.setVisible(False)

        if key == "assignments":
            create = QToolButton(self)
            create.setAutoRaise(True)
            create.setFixedSize(36, 26)
            create.setIconSize(QSize(16, 16))
            create.setPopupMode(QToolButton.ToolButtonPopupMode.InstantPopup)
            set_button_icon(create, "square-plus", tooltip="Create assignments")
            create_menu = QMenu(create)
            create_menu.addAction(
                "New, empty…", lambda: self._on_load_option_changed(key, "Create new")
            )
            create_menu.addAction(
                "Copy of the current…",
                lambda: self._on_load_option_changed(key, "Create named copy…"),
            )
            create.setMenu(create_menu)
            self.loader[key]["button_create"] = create
            layout.insertWidget(1, create)

        remove = make_icon_button(
            "ban",
            color="red",
            tooltip=(
                "Remove the current model from the workspace (files stay)"
                if key == "model"
                else "Remove the current assignments from the workspace (files stay)"
            ),
            size=26,
        )
        remove.clicked.connect(lambda: self._on_load_option_changed(key, "Remove current…"))
        self.loader[key]["button_remove"] = remove
        layout.addWidget(remove)

        for widget in add_widgets or ():
            widget.setFixedSize(28, 26)
            widget.setIconSize(QSize(16, 16))
            layout.addWidget(widget)

        def execute_clicked():
            if key == "model":
                self.data.queue_process_model(mode="pending")
                return

            assignments = self.data.assignments
            if assignments is None:
                return

            if not (assignments.path and not assignments.status["loaded"]):
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
            "move_footprints": "reassign",
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

    def import_source_file(self, key, path):
        """Import assignments or a model from ``path`` (named as with the
        sections' Load …)."""
        if key == "assignments":
            name, ok, use_working = self._ask_assignments_import_name(path)
            replace = False
        else:
            name, ok, replace = self._ask_model_import_name(path)
            use_working = False
        name = (name or "").strip()
        if not ok or not name:
            return
        self.data.queue_import_source(
            key,
            path,
            name,
            use_as_working=use_working,
            replace_empty_model=replace,
            on_loaded=lambda: self._on_source_registered(key),
        )

    def save_project(self, *, save_as=False):
        """Save the project (Data.queue_save_project). The first save, and
        Save as, ask for the project name. Returns the path, or None."""
        path = None if save_as else self.data.project_path()
        if path is None:
            path = self._ask_project_path()
            if path is None:
                return None
        try:
            self.data.queue_save_project(path)
        except (ValueError, OSError) as exc:
            self.state.issue("warning", "Project not saved", str(exc), parent=self.window())
            return None
        return path

    def _ask_project_path(self):
        folder = self.data.project_folder()
        suffix = self.data.PROJECT_SUFFIX
        name, accepted = QInputDialog.getText(
            self,
            "Save project",
            f"Project name (saved as name{suffix} in\n{display_path(str(folder))}):",
            text=self.data.default_project_name(),
        )
        name = name.strip() if accepted else ""
        if name.endswith(suffix):
            name = name[: -len(suffix)]
        if not name:
            return None

        path = folder / f"{name}{suffix}"
        if path.exists() and str(path) != self.data.project_path():
            answer = QMessageBox.question(
                self.window(),
                "Replace project?",
                f"{display_path(str(path))} already exists. Replace it (and its "
                f"{name}.sessions.hdf5 / {name}.model.hdf5 where they are saved again)?",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                QMessageBox.StandardButton.No,
            )
            if answer != QMessageBox.StandardButton.Yes:
                return None
        return str(path)

    def open_project(self, path=None):
        """Open a project file (or a CATAN assignments file): its sessions
        and model (referenced next to it) are loaded first, then the
        assignments. Asks only if sessions or a model are loaded already.
        Without ``path``, a file dialog."""
        if path is None:
            folder = self.data.project_folder()
            path = choose_path(
                self,
                pick_dir=False,
                init_path=str(folder if folder.is_dir() else self.data.root),
                display_text="Open CATAN project",
                only_existing=True,
                file_filters=[("CATAN project", (f"*{self.data.PROJECT_SUFFIX}",))],
                state=self.state,
            )
        if not path:
            return

        suffix = self.data.PROJECT_SUFFIX
        base = Path(path).name
        base = base[: -len(suffix)] if base.endswith(suffix) else Path(path).stem
        use_working = self.data.can_use_working_assignments()
        name, number = base, 2
        while name in self.data.available_assignments:
            name, number = f"{base} ({number})", number + 1

        self._opening_project = True
        self.data.queue_import_source(
            "assignments",
            path,
            name,
            use_as_working=use_working,
            on_loaded=lambda: self._on_source_registered("assignments"),
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

        # The save may start later (after display workers stopped).
        if save_path is not None:
            self.data.queue_save_result(key, save_path)

        return save_path

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
        if opt in source_actions[key]:
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

    ASSIGNMENT_CONFIG_HINT = (
        "Select a matching preset or correct the IDs field, then click Load."
    )

    def _cancel_assignments_import(self, source):
        """Drop an imported assignments file that was not loaded (no
        orphaned file reference remains)."""
        name = next(
            (name for name, item in self.data._assignments.items() if item is source), None
        )
        if name is None or source.status["loaded"]:
            return
        self.data.remove_loaded_source(
            "assignments",
            name,
            expected_source=source,
            on_removed=lambda: self._refresh_source_selector("assignments"),
        )

    def _show_assignments_configuration(self, problems=None, hint=None):
        """Open the assignments load configuration; ``problems`` (why
        loading stopped) are shown in it, not in a separate message box."""
        constructor = self.config_constructor
        source = self.data.assignments
        constructor.update_source(source)
        if problems:
            constructor.show_problems(
                problems, self.ASSIGNMENT_CONFIG_HINT if hint is None else hint
            )
            # A file waiting to be loaded: confirm (Load) or drop (Cancel).
            if source is not None and source.path and not source.status["loaded"]:
                constructor.set_pending_load(
                    on_load=self._load_assignments_from_source,
                    on_cancel=lambda: self._cancel_assignments_import(source),
                )
        else:
            constructor.open()
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
            self._show_assignments_configuration(problems)
            return

        if not self._offer_assignments_provenance(source):
            return

        self._load_assignments_from_source()

    def _offer_assignments_provenance(self, source) -> bool:
        """Offer the sessions and model stored with saved assignments.

        Returns False if the user cancelled (assignments stay unloaded and
        can be loaded with the open-folder button). Otherwise the stored
        sessions/model are queued first when chosen; loading the
        assignments waits for that processing.
        """
        provenance = getattr(source, "provenance", None)

        for note in (provenance or {}).get("ignored", ()):
            self.state.logger.warning("Stored reference ignored: %s", note)

        if not provenance or not (
            provenance.get("recipe") is not None
            or provenance.get("model_path")
            or provenance.get("sessions_file")
        ):
            self._opening_project = False
            return True

        sessions = []
        if provenance.get("recipe") is not None:
            try:
                sessions = self.data.parse_loading_recipe(
                    provenance["recipe"], Path(source.path).parent
                )
            except ValueError as exc:
                self.state.issue(
                    "warning",
                    "Stored sessions not usable",
                    f"The assignments file lists its sessions, but they cannot "
                    f"be read: {exc}",
                    parent=self.window(),
                )

        model_path = provenance.get("model_path")
        current_model_path = getattr(self.data.model, "path", None)

        # Preferred: the CATAN session file (keeps alignment and loaded
        # data); the recipe re-registers from the original sources.
        sessions_file = provenance.get("sessions_file")
        if sessions_file and not Path(sessions_file).is_file():
            self.state.issue(
                "warning",
                "Stored session file not found",
                f"{display_path(sessions_file)}\n\nThe sessions can only be loaded from "
                "their original sources (loading recipe).",
                parent=self.window(),
            )
            sessions_file = None

        key = self.data.session_source_key
        if sessions_file and self.data.catan_sessions_file() is not None:
            sessions_differ = key(self.data.catan_sessions_file()) != key(sessions_file)
        elif sessions:
            sessions_differ = [
                key(session.path) for session in self.data.sessions
            ] != [key(session.path) for session in sessions]
        else:
            sessions_differ = bool(sessions_file)
        model_differs = bool(model_path) and (
            current_model_path is None
            or Path(current_model_path).resolve() != Path(model_path).resolve()
        )

        if not sessions_differ and not model_differs:
            return True

        model = self.data.model
        have_current = bool(self.data.sessions) or (
            model is not None and (model.fitted or model.loaded)
        )

        stored = []
        if sessions_differ and sessions_file:
            stored.append(
                (f"{len(sessions)} sessions" if sessions else "sessions")
                + f" (session file {Path(sessions_file).name})"
            )
        elif sessions_differ:
            stored.append(f"{len(sessions)} sessions (loading recipe)")
        if model_differs:
            stored.append(f"model {Path(model_path).name}")

        # Opening a project into an empty workspace: load without asking.
        opening = getattr(self, "_opening_project", False)
        self._opening_project = False
        if not (opening and not have_current):
            dialog = QMessageBox(self.window())
            dialog.setIcon(QMessageBox.Icon.Question)
            dialog.setWindowTitle("Sessions and model of these assignments")
            dialog.setText(
                "These assignments were saved together with "
                + " and ".join(stored)
                + "."
            )

            details = []
            if sessions_differ and sessions_file:
                details.append(f"Session file:\n  {display_path(sessions_file)}")
            elif sessions_differ:
                details.append(
                    "Sessions:\n"
                    + "\n".join(
                        f"  {display_path(session.path)}" for session in sessions[:8]
                    )
                    + (f"\n  … {len(sessions) - 8} more" if len(sessions) > 8 else "")
                )
            if model_differs:
                details.append(f"Model:\n  {display_path(model_path)}")

            if have_current:
                dialog.setInformativeText(
                    "Keep the currently loaded sessions and model, or replace them? "
                    "Replacing removes the current sessions, including their "
                    "columns in all loaded assignments."
                )
                load_button = dialog.addButton(
                    "Replace with stored", QMessageBox.ButtonRole.DestructiveRole
                )
                keep_button = dialog.addButton(
                    "Keep current", QMessageBox.ButtonRole.AcceptRole
                )
            else:
                dialog.setInformativeText("Load them before the assignments?")
                load_button = dialog.addButton(
                    "Load stored sessions and model", QMessageBox.ButtonRole.AcceptRole
                )
                keep_button = dialog.addButton(
                    "Assignments only", QMessageBox.ButtonRole.RejectRole
                )

            dialog.setDetailedText("\n\n".join(details))
            cancel_button = dialog.addButton(QMessageBox.StandardButton.Cancel)
            dialog.setDefaultButton(keep_button if have_current else load_button)
            fit_message_box_buttons(dialog)
            dialog.exec()

            clicked = dialog.clickedButton()

            if clicked is cancel_button or not isValid(self):
                return False

            if clicked is keep_button:
                return True

        if sessions_differ:
            for session_id in reversed(range(len(self.data.sessions))):
                self.data.remove_session(session_id)

            loader = self.session_list.load_row
            actions = set(loader.registration_action_selector.actions)

            if sessions_file:
                # Restores all sessions with alignment and loaded data.
                loader._queue_registration_path(
                    Path(sessions_file),
                    batch=TaskBatch(),
                    actions=actions,
                    force_first_session=True,
                )
            else:
                # The usual recipe review, then registration and loading.
                loader._review_recipe(
                    sessions,
                    batch=TaskBatch(),
                    actions=actions,
                )

        if model_differs:
            replace = self.data.can_replace_empty_model()

            if replace:
                # An empty current model is replaced under its own name.
                name = self.data.current_model_name
            else:
                base = Path(model_path).stem or "model"
                name, number = base, 2
                while name in self.data.available_models:
                    name, number = f"{base} ({number})", number + 1

            self.data.queue_import_source(
                "model",
                model_path,
                name,
                replace_empty_model=replace,
                on_loaded=lambda: self._refresh_source_selector("model"),
            )

        return self.data.assignments is source

    def update_buttons(self):

        self._refresh_execute_buttons()

        for controls in self.loader.values():
            controls["button_execute"].setIconSize(QSize(16, 16))
            controls["button_save"].setIconSize(QSize(16, 16))

    def _refresh_execute_buttons(self, *_):
        """Icons, tooltips and availability of the model / assignments "go"
        buttons. While their work is running, queued or deferred, they show
        a spinner and cannot be clicked again."""
        if not isValid(self) or not hasattr(self, "loader"):
            return

        sessions_loaded = bool(self.data.sessions)
        model = self.data.model

        # --- model
        button = self.loader["model"]["button_execute"]
        if self.data.model_work_pending():
            self._set_busy_icon(button, "Model update scheduled or running…")
        else:
            self._clear_busy_icon(button)
            set_button_icon(button, "play", tooltip="Process pending model counts and fit")
            button.setEnabled(model is not None and not model.loaded and sessions_loaded)

        # --- assignments
        button = self.loader["assignments"]["button_execute"]
        assignments = self.data.assignments
        loading = bool(
            assignments is not None
            and assignments.path
            and not assignments.status["loaded"]
        )

        if not loading and self.data.assignment_work_pending():
            self._set_busy_icon(button, "Neuron registration scheduled or running…")
            return

        self._clear_busy_icon(button)
        set_button_icon(
            button, "play", tooltip="Process missing or outdated neuron registrations"
        )

        # Registration needs match probabilities: a fitted (or imported) model.
        # A file waiting to be loaded is confirmed in its configuration.
        model_ready = model is not None and model.fitted
        button.setEnabled(
            assignments is not None and not loading and sessions_loaded and model_ready
        )
        if loading:
            button.setToolTip(
                "Load or cancel the selected assignments file in its "
                "configuration first."
            )
        elif not model_ready:
            button.setToolTip(
                "Process neuron registrations — fit or import a model first "
                "(registration needs its match probabilities)."
            )

    def _set_busy_icon(self, button, tooltip):
        spins = self.__dict__.setdefault("_busy_spins", {})
        if button not in spins:
            spin = qta.Spin(button, interval=40, step=12)
            spins[button] = spin
            button.setIcon(
                qta.icon(
                    "fa5s.spinner",
                    color="white",
                    color_disabled="white",
                    animation=spin,
                )
            )
        button.setToolTip(tooltip)
        button.setEnabled(False)

    def _clear_busy_icon(self, button):
        spin = self.__dict__.get("_busy_spins", {}).pop(button, None)
        if spin is not None:
            # The animation timer would otherwise keep repainting.
            for timer, *_ in spin.info.values():
                timer.stop()

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
