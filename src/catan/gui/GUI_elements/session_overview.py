from typing import Optional
from unicodedata import name

import pickle
import traceback
import os, re
from copy import deepcopy
from shiboken6 import isValid
import numpy as np
from pathlib import Path

from PySide6.QtCore import QSize, QTimer, Qt, Signal, QPoint
from PySide6.QtGui import QColor, QAction, QActionGroup, QShortcut, QKeySequence
from PySide6.QtWidgets import (
    QApplication,
    QAbstractItemView,
    QComboBox,
    QDialog,
    QWidget,
    QFrame,
    QLabel,
    QLineEdit,
    QCheckBox,
    QToolButton,
    QPushButton,
    QHBoxLayout,
    QVBoxLayout,
    QMenu,
    QListWidget,
    QListWidgetItem,
    QInputDialog,
    QColorDialog,
    QMessageBox,
    QWidgetAction,
    QSizePolicy,
    QFileDialog,
)


from catan.core.changes import (
    ChangeKind as C,
    DataChange,
    SESSION_STRUCTURE_CHANGES,
)
from catan.core.structures import sessiondata_type
from catan.core.structures.load_config import FieldSpec
from catan.core.io import resolve_source_path
from catan.core.io.isolated_read import read_operation

from catan.gui.structures import AppState, Data, SessionData
from catan.gui.background_tasks.TaskManager import TaskBatch
from catan.gui.background_tasks.runtime import current_task_context


from .fragments.FileReviewDialog import RecipeReviewDialog
from .fragments import (
    FieldConfigConstructor,
    GlobReviewDialog,
    make_icon_button,
    set_button_icon,
    choose_path,
)
from .fragments.dialog_load_field import (
    FieldSelectDialog,
    FieldSelection,
)
from .fragments.manual_alignment_dialog import (
    open_manual_alignment,
    select_alignment_draft,
)
from .fragments.file_inspection import FileInspection


class SessionRowWidget(QFrame):
    moveRequested = Signal(int, int)  # session_id, delta
    activeChanged = Signal(int, bool)  # session_id, active
    nameChanged = Signal(int, str)  # session_id, new_name

    setCurrentRequested = Signal(int)
    editOffsetRequested = Signal(int)
    changeColorRequested = Signal(int)

    loadRequested = Signal(int)  # session_id
    backgroundRequested = Signal(int)
    manualAlignmentRequested = Signal(int)

    traceToggled = Signal(int)
    qualityToggled = Signal(int)
    spatialToggled = Signal(int)

    modelRequested = Signal(int)
    assignmentRequested = Signal(int)
    removeRequested = Signal(int)

    expanded_changed = Signal()

    def __init__(
        self,
        session_id: int,
        item: QListWidgetItem,
        session: SessionData,
        current=False,
        parent=None,
    ):
        super().__init__(parent)

        self.index = session_id
        self.item = item
        self.session: SessionData = session

        self.state: AppState = parent._state
        self.data: Data = parent.data

        self.setSizePolicy(
            QSizePolicy.Policy.Expanding,
            QSizePolicy.Policy.Preferred,  # or minimum
        )
        self.setFixedWidth(300)

        self.setObjectName("SessionRowWidget")
        self.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
        self.customContextMenuRequested.connect(self._open_context_menu)

        order_layout = QVBoxLayout()
        order_layout.setContentsMargins(0, 0, 0, 0)
        order_layout.setSpacing(0)

        self.active_checkbox = QCheckBox()
        self.active_checkbox.stateChanged.connect(self._on_active_changed)

        self.name_edit = QLineEdit()
        self.name_edit.setObjectName("SessionNameEdit")
        self.name_edit.setMaximumWidth(80)
        self.name_edit.editingFinished.connect(self._on_name_finished)

        self.offset_label = QLabel()
        self.offset_label.setObjectName("SessionOffsetLabel")
        self.offset_label.setAlignment(Qt.AlignmentFlag.AlignCenter)

        self.status_button = QToolButton(self)
        self.status_button.setText("●")
        self.status_button.setFixedSize(24, 24)
        self.status_button.setCursor(Qt.CursorShape.PointingHandCursor)
        self.status_button.clicked.connect(self._show_status_details)

        self.load_fields_button = make_icon_button(
            "folder-open",
            color="white",
            tooltip="Process session data",
            fallback_theme_icon="system-run",
        )
        self.load_fields_button.setMinimumWidth(50)
        # QToolButton(self)
        # self.load_fields_button.setText("Process")
        self.load_fields_button.setPopupMode(
            QToolButton.ToolButtonPopupMode.MenuButtonPopup
        )

        self.load_fields_button.clicked.connect(
            lambda: self.loadRequested.emit(self.index)
        )

        ## define submenu for load_fields_button
        menu = QMenu(self.load_fields_button)

        container = QWidget(menu)
        layout_menu = QHBoxLayout(container)
        layout_menu.setContentsMargins(6, 6, 6, 6)
        layout_menu.setSpacing(4)

        self.trace_button = make_icon_button()
        self.quality_button = make_icon_button()
        self.spatial_button = make_icon_button()

        ## define further buttons
        self.register_model_button = make_icon_button(
            "plus",
            color="white",
            tooltip="Register neurons across sessions",
            fallback_theme_icon="system-run",
        )
        self.assignments_button = make_icon_button(
            "layer-group",
            color="white",
            tooltip="View assignments",
            fallback_theme_icon="system-run",
        )

        layout_menu.addWidget(self.trace_button)
        layout_menu.addWidget(self.quality_button)
        layout_menu.addWidget(self.spatial_button)
        layout_menu.addWidget(self.register_model_button)
        layout_menu.addWidget(self.assignments_button)

        widget_action = QWidgetAction(menu)
        widget_action.setDefaultWidget(container)

        menu.addAction(widget_action)

        self.load_fields_button.setMenu(menu)

        self.trace_button.clicked.connect(lambda: self.traceToggled.emit(self.index))
        self.quality_button.clicked.connect(
            lambda: self.qualityToggled.emit(self.index)
        )
        self.spatial_button.clicked.connect(
            lambda: self.spatialToggled.emit(self.index)
        )
        self.register_model_button.clicked.connect(
            lambda: self.modelRequested.emit(self.index)
        )
        self.assignments_button.clicked.connect(
            lambda: self.assignmentRequested.emit(self.index)
        )

        self.delete_button = make_icon_button(
            "ban",
            color="red",
            tooltip="Remove session data",
            fallback_theme_icon="edit-delete",
        )
        self.delete_button.clicked.connect(
            lambda: self.removeRequested.emit(self.index)
        )

        stacked_layout = QVBoxLayout(self)
        stacked_layout.setContentsMargins(0, 0, 0, 0)

        layout = QHBoxLayout()
        layout.setContentsMargins(6, 3, 6, 3)
        layout.setSpacing(3)

        layout.addLayout(order_layout)
        layout.addWidget(self.active_checkbox)
        layout.addWidget(self.name_edit)
        layout.addWidget(self.offset_label)
        layout.addWidget(self.status_button)
        layout.addStretch()
        layout.addWidget(self.load_fields_button)

        layout.addWidget(self.delete_button)

        self.config_constructor = FieldConfigConstructor(self, self.session)

        layout.addWidget(self.config_constructor.toggle_config_options)
        stacked_layout.addLayout(layout)

        self.refresh(current=current)

    def _on_data_changed(self, event: DataChange):
        if event.has(
            C.DATA_AVAILABILITY,
            C.SESSION_METADATA,
            C.SESSION_ACTIVITY,
            C.SESSION_TIMEBASE,
            C.FOOTPRINT_GEOMETRY,
            C.BACKGROUND_IMAGE,
            C.ASSIGNMENT_SET,
            C.ASSIGNMENT_MAPPING,
            C.INCLUSION,
            C.MODEL_COUNTS,
            C.MODEL_PARAMETERS,
            C.PROCESSING_STATUS,
        ):
            self._update_buttons()
            self._update_status()

        if event.has(C.DATA_AVAILABILITY):
            self.config_constructor.config_field_options.rebuild()

    def refresh(self, current=False):
        name = getattr(self.session, "name", f"Session{self.index:02d}")
        path = getattr(self.session, "path", "")
        active = getattr(self.session, "active", True)
        offset = getattr(self.session, "time_offset", 0)

        self.name_edit.blockSignals(True)
        self.name_edit.setText(str(name))
        self.name_edit.blockSignals(False)

        self.name_edit.setToolTip(str(path))

        self.active_checkbox.blockSignals(True)
        self.active_checkbox.setChecked(bool(active))
        self.active_checkbox.blockSignals(False)

        if offset:
            self.offset_label.setText(f"{offset:+d}")
            self.offset_label.setVisible(True)
            self.offset_label.setToolTip("Trace time offset")
        else:
            self.offset_label.setVisible(False)

        self._update_buttons()
        self._update_status()
        self._update_background(current=current)
        self.config_constructor.refresh()

    def _update_status(self):

        key, label, color = self._session_state()

        self.status_button.setStyleSheet(f"""
            QToolButton {{
                color: {color};
                background-color: rgba(
                    0, 0, 0, 80
                );
                border: 1px solid rgba(
                    255, 255, 255, 40
                );
                border-radius: 4px;
                font-size: 17px;
                font-weight: bold;
            }}

            QToolButton:hover {{
                background-color: rgba(
                    255, 255, 255, 30
                );
            }}
            """)

        tooltip = label

        if self.session.status["spatial_loaded"] and self.session.remap is not None:
            tooltip += "\n\n" + self._alignment_report_text(detailed=False)

        self.status_button.setToolTip(tooltip)

    def _show_status_details(self):

        key, label, _ = self._session_state()

        dialog = QMessageBox(self)

        dialog.setWindowTitle(f"Session: {self.session.name}")

        if key == "alignment_error":
            dialog.setIcon(QMessageBox.Icon.Warning)
        else:
            dialog.setIcon(QMessageBox.Icon.Information)

        dialog.setText(label)

        # ------------------------------------------
        # Alignment information
        # ------------------------------------------

        if self.session.status["spatial_loaded"] and self.session.remap is not None:

            dialog.setInformativeText(self._alignment_report_text(detailed=False))
            detailed = self._alignment_report_text(detailed=True)
            dialog.setDetailedText(detailed)

        # ------------------------------------------
        # Other lifecycle information
        # ------------------------------------------

        else:
            details = [
                f"Spatial loaded: " f"{self.session.status['spatial_loaded']}",
                f"Traces loaded: " f"{self.session.status['traces_loaded']}",
                f"Quality loaded: " f"{self.session.status['quality_loaded']}",
                f"Registered to model: "
                f"{self.session.status['registered_to_model']}",
                ("Neurons tracked: " f"{self.data.session_assigned(self.session.id)}"),
            ]

            dialog.setDetailedText("\n".join(details))

        dialog.exec()

    def _session_state(self):
        """
        Return:
            key, label, color
        """

        status = self.session.status

        session_id = self.session.id
        model = self.data.model

        outdated = []

        if self.data.alignment_is_stale(session_id):
            blocker = next(
                (sid for sid in range(session_id) if self.data.alignment_is_stale(sid)),
                None,
            )

            if blocker is not None:
                return (
                    "stale",
                    f"Alignment outdated — review S{blocker} first",
                    "#f2b84b",
                )

            return (
                "stale",
                "Alignment needs review — blocks later sessions",
                "#f2b84b",
            )

        if (
            model is not None
            and self.session.path is not None
            and model.counts_stale_for_path(self.session.path)
        ):
            outdated.append("model counts")

        if model is not None and model.fit_stale:
            outdated.append("current model fit")

        if (
            self.data.assignments is not None
            and self.data.session_assigned(session_id)
            and self.data.assignment_is_stale(session_id)
        ):
            outdated.append("neuron assignments")

        if outdated:
            return (
                "stale",
                "Outdated: " + ", ".join(outdated),
                "#f2b84b",
            )

        any_data_loaded = any(
            status.get(key, False)
            for key in ("spatial_loaded", "traces_loaded", "quality_loaded")
        )

        # ------------------------------------------
        # Registered only
        # ------------------------------------------
        if not any_data_loaded:
            return ("registered", "Registered – no data loaded", "#9aa0a6")

        # ------------------------------------------
        # Some auxiliary data, but no spatial data
        # ------------------------------------------
        if not status["spatial_loaded"]:
            return ("partial", "Data loaded – no spatial data", "#7aa2d6")

        # ------------------------------------------
        # Spatial data loaded, alignment failed
        # ------------------------------------------
        if not status["aligned"]:
            return ("alignment_error", "Alignment error", "#ef5350")

        # ------------------------------------------
        # Aligned, but not tracked
        # ------------------------------------------
        matched = self.data.assignments is not None and self.data.session_assigned(
            self.session.id
        )

        if not matched:
            return ("ready", "Aligned – neurons not tracked", "#f2b84b")

        # ------------------------------------------
        # Fully tracked
        # ------------------------------------------
        return ("tracked", "Aligned and tracked", "#66bb6a")

    def _update_buttons(self):

        spatial_loaded = self.session.status["spatial_loaded"]
        set_button_icon(
            self.spatial_button,
            "paw",
            color="white" if not spatial_loaded else "red",
            tooltip="Load footprint data",
            fallback_theme_icon="square",
        )

        trace_loaded = self.session.status["traces_loaded"]
        set_button_icon(
            self.trace_button,
            "chart-line",
            color="white" if not trace_loaded else "red",
            tooltip=("Load traces" if not trace_loaded else "Unload traces"),
            fallback_theme_icon="spinner",
        )

        quality_loaded = self.session.status["quality_loaded"]
        set_button_icon(
            self.quality_button,
            "chart-column",
            color="white" if not quality_loaded else "red",
            tooltip=(
                "Load quality parameters"
                if not quality_loaded
                else "Remove quality parameters"
            ),
            fallback_theme_icon="spinner",
        )

        registered = self.session.status["registered_to_model"]
        set_button_icon(
            self.register_model_button,
            "plus",
            color="white" if not registered else "red",
            tooltip=(
                "Register neurons to model"
                if not registered
                else "Unregister neurons from model"
            ),
            fallback_theme_icon="spinner",
        )

        if self.data.assignments is None:
            return
        matched = self.data.session_assigned(self.session.id)
        set_button_icon(
            self.assignments_button,
            "layer-group",
            color="white" if not matched else "red",
            tooltip=("Add to matching" if not matched else "Remove from matching"),
            fallback_theme_icon="spinner",
        )

    def _update_background(self, current=False):

        color = getattr(self.session, "color", QColor("#333333"))
        if current:
            color = getattr(self.session, "color", QColor("#4F7F5A"))

        if not isinstance(color, QColor):
            color = QColor(str(color))

        # Soft translucent background, so text remains readable.
        r, g, b, _ = color.getRgb()
        self.setStyleSheet(f"""
            QFrame#SessionRowWidget {{
                background-color: rgba({r}, {g}, {b}, 85);
                border: {"3px solid rgba(255, 255, 255, 85)" if current else "1px solid rgba(255, 255, 255, 85)"};
                border-radius: 4px;
                font-weight: {"bold" if current else "normal"};
            }}

            QLineEdit#SessionNameEdit {{
                background: rgba(255, 255, 255, 35);
                border: 1px solid rgba(255, 255, 255, 55);
                border-radius: 3px;
                padding-left: 3px;
            }}

            QLabel#SessionOffsetLabel {{
                padding: 1px 4px;
                border-radius: 3px;
                background: rgba(0, 0, 0, 55);
            }}
            """)

    def _alignment_report_text(self, *, detailed: bool = False) -> str:

        remap = self.session.remap

        if remap is None:
            return "No alignment has been calculated."

        report = remap.report

        lines = []

        if report.success:
            lines.append("Alignment successful")
        else:
            lines.append("Alignment failed")

            if report.reason:
                lines.append("Reason: " + str(report.reason).replace("_", " "))

        lines.append("")

        if report.shift is not None:

            shift = np.asarray(report.shift).ravel()

            if shift.size >= 2:
                lines.append(f"Shift: dy={shift[0]:.2f} px, dx={shift[1]:.2f} px")

            if np.isfinite(report.total_shift):
                lines.append(f"Total shift: {report.total_shift:.2f} px")

        if report.rotation is not None:
            lines.append(f"Rotation: {report.rotation:.2f}°")

        if report.correlation is not None:
            lines.append(f"Correlation: {report.correlation:.3f}")

        if report.correlation_zscore is not None:
            lines.append(f"Correlation z-score: {report.correlation_zscore:.2f}")

        lines.append(
            f"References passed: {report.n_successful_references}/{report.n_references}"
        )

        if not detailed:
            return "\n".join(lines)

        # ==================================================
        # Pairwise information
        # ==================================================

        if report.remap_data:

            lines.append("")
            lines.append("Pairwise alignments")
            lines.append("-------------------")

            for path, entry in report.remap_data.items():

                lines.append("")
                lines.append(str(path))

                shift = entry.get("shift")

                if shift is not None:
                    shift = np.asarray(shift).ravel()

                    if shift.size >= 2:
                        lines.append(
                            "  shift: " f"dy={shift[0]:.2f}px, dx={shift[1]:.2f}px"
                        )

                rotation = entry.get("rotation")

                if rotation is not None:
                    lines.append(f"  rotation: {rotation:.2f}°")

                corr = entry.get("c_max")

                if corr is not None:
                    lines.append(f"  correlation: {corr:.3f}")

                zscore = entry.get("c_zscored")

                if zscore is not None:
                    lines.append(f"  z-score: {zscore:.2f}")

                if entry.get("success", False):
                    lines.append("  status: accepted")
                else:
                    reason = entry.get("reason", "failed")

                    lines.append("  status: " + str(reason).replace("_", " "))

        return "\n".join(lines)

    def _on_active_changed(self, state):
        self.activeChanged.emit(
            self.index,
            state == Qt.CheckState.Checked.value,
        )

    def _on_name_finished(self):
        self.nameChanged.emit(self.index, self.name_edit.text().strip())

    def _open_context_menu(self, pos: QPoint):
        menu = QMenu(self)

        menu.addAction(
            "Set current session",
            lambda: self.setCurrentRequested.emit(self.index),
        )
        menu.addSeparator()

        menu.addAction(
            "Edit time offset…", lambda: self.editOffsetRequested.emit(self.index)
        )
        menu.addAction(
            "Change color…", lambda: self.changeColorRequested.emit(self.index)
        )

        spatial_loaded = self.session.status["spatial_loaded"]

        menu.addAction(
            (
                "Change background…"
                if spatial_loaded
                else "Select background and load spatial data…"
            ),
            lambda: self.backgroundRequested.emit(self.index),
        )

        if spatial_loaded:
            menu.addAction(
                "Adjust alignment…",
                lambda: self.manualAlignmentRequested.emit(self.index),
            )

        menu.addSeparator()
        remove_action = QAction("Remove session", menu)
        remove_action.triggered.connect(lambda: self.removeRequested.emit(self.index))
        menu.addAction(remove_action)

        menu.exec(self.mapToGlobal(pos))


REGISTRATION_ACTION_LABELS = {
    "load_data": "Load data",
    "load_all": "Load all data",
    "prompt_background": "Prompt for background",
    "register_model": "Register to model",
    "track_neurons": "Track neurons",
}


class RegistrationActionMenu(QMenu):
    """
    Checkable menu that stays open while options are toggled.
    """

    def mouseReleaseEvent(self, event):

        action = self.actionAt(event.position().toPoint())

        if action is not None and action.isEnabled() and action.isCheckable():
            action.setChecked(not action.isChecked())
            return

        super().mouseReleaseEvent(event)


class RegistrationActionSelector(QToolButton):

    actionsChanged = Signal(object)
    backgroundOrientationChanged = Signal(str)

    BACKGROUND_ORIENTATION_KEY = "session_registration/background_orientation"

    SETTINGS_KEY = "session_registration/actions"

    def __init__(self, settings, parent=None):
        super().__init__(parent)

        self.settings = settings
        self._syncing = False
        self._actions = {}

        self.setToolButtonStyle(Qt.ToolButtonStyle.ToolButtonTextBesideIcon)

        self.setPopupMode(QToolButton.ToolButtonPopupMode.InstantPopup)

        self.menu = RegistrationActionMenu(self)
        self.setMenu(self.menu)

        for key, label in REGISTRATION_ACTION_LABELS.items():

            action = QAction(label, self.menu)

            action.setCheckable(True)
            action.setData(key)

            action.toggled.connect(
                lambda checked, key=key: self._on_action_toggled(
                    key,
                    checked,
                )
            )

            self.menu.addAction(action)
            self._actions[key] = action

        self.menu.addSeparator()
        orientation_menu = self.menu.addMenu("Background orientation")

        self._orientation_group = QActionGroup(self)
        self._orientation_group.setExclusive(True)
        self._orientation_actions = {}

        saved = self.settings.value(self.BACKGROUND_ORIENTATION_KEY, "auto")
        if saved not in {"auto", "as_stored", "transpose"}:
            saved = "auto"

        for value, label in (
            ("auto", "Automatic"),
            ("as_stored", "As stored — skip detection"),
            ("transpose", "Transpose — skip detection"),
        ):
            action = orientation_menu.addAction(label)
            action.setCheckable(True)
            action.setData(value)
            self._orientation_group.addAction(action)
            self._orientation_actions[value] = action
            action.setChecked(value == saved)

        self._orientation_group.triggered.connect(
            self._on_background_orientation_changed
        )

        self._restore()
        self._update_text()

    @property
    def actions(self) -> set[str]:
        return {key for key, action in self._actions.items() if action.isChecked()}

    def set_actions(self, actions, *, save=True):

        actions = set(actions)

        self._syncing = True

        try:
            for key, action in self._actions.items():
                action.setChecked(key in actions)
        finally:
            self._syncing = False

        self._update_text()

        if save:
            self._save()

        self.actionsChanged.emit(self.actions)

    def _on_action_toggled(self, key, checked):

        if self._syncing:
            return

        # These are two alternative loading modes.
        if checked and key == "load_data":
            self._set_checked_silent("load_all", False)

        elif checked and key == "load_all":
            self._set_checked_silent("load_data", False)

        self._update_text()
        self._save()

        self.actionsChanged.emit(self.actions)

    @property
    def background_orientation(self):
        action = self._orientation_group.checkedAction()
        return action.data() if action is not None else "auto"

    def _on_background_orientation_changed(self, action):
        value = action.data()
        self.settings.setValue(self.BACKGROUND_ORIENTATION_KEY, value)
        self.backgroundOrientationChanged.emit(value)

    def _set_checked_silent(self, key, checked):

        self._syncing = True
        try:
            self._actions[key].setChecked(checked)
        finally:
            self._syncing = False

    def _update_text(self):

        actions = self.actions

        if not actions:
            text = "On registration: none"

        elif len(actions) == 1:
            key = next(iter(actions))
            text = "On registration: " + REGISTRATION_ACTION_LABELS[key]

        else:
            text = f"On registration: {len(actions)} actions"

        self.setText(text)

        selected = [
            REGISTRATION_ACTION_LABELS[key]
            for key in REGISTRATION_ACTION_LABELS
            if key in actions
        ]

        tooltip = "Actions performed automatically after " "registering a session."

        if selected:
            tooltip += "\n\n" + "\n".join(f"• {label}" for label in selected)

        tooltip += (
            "\n\nThe first individually registered "
            "session is loaded, registered to the model, "
            "and tracked automatically when possible."
        )

        self.setToolTip(tooltip)

    def _save(self):

        self.settings.setValue(self.SETTINGS_KEY, list(self.actions))

    def _restore(self):

        saved = self.settings.value(self.SETTINGS_KEY, [])

        # QSettings may deserialize an empty stored list as None.
        if saved is None:
            saved = []

        elif isinstance(saved, str):
            saved = [saved] if saved else []

        saved = [key for key in saved if key in self._actions]

        self.set_actions(saved, save=False)


class LoadSessionRowWidget(QFrame):
    loadRequested = Signal()
    loadConfigRequired = Signal(int)

    def __init__(self, parent: "SessionOverview"):
        super().__init__(parent)

        self.state = parent.state
        self.data = parent.data

        self.setObjectName("SessionRowWidget")

        root_layout = QVBoxLayout(self)
        root_layout.setContentsMargins(6, 3, 6, 5)
        root_layout.setSpacing(4)

        layout = QHBoxLayout()
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(3)

        root_layout.addLayout(layout)

        self._loading_check = None
        self._loading_status_batch = None
        self._loading_had_error = False
        self._loading_status_task_ids = set()

        self.loading_status = QWidget(self)
        status_layout = QVBoxLayout(self.loading_status)
        status_layout.setContentsMargins(0, 0, 0, 0)
        status_layout.setSpacing(3)

        self.loading_status_title = QLabel()
        self.loading_status_title.setTextFormat(Qt.TextFormat.PlainText)
        self.loading_status_title.setWordWrap(True)
        status_layout.addWidget(self.loading_status_title)

        self.loading_inspection = FileInspection(self.loading_status)
        self.loading_inspection.message_formatter = self._format_loading_message
        status_layout.addWidget(self.loading_inspection)

        self.loading_status_title.setText("Ready for loading further session data")
        self.loading_inspection.hide()
        self.loading_status.show()
        root_layout.addWidget(self.loading_status)

        self.loading_inspection.result.connect(self._on_loading_inspected)
        self.loading_inspection.failed.connect(self._on_loading_inspection_failed)
        self.loading_inspection.cancelled.connect(self._cancel_inline_loading)

        tasks = self.state.tasks
        tasks.task_started.connect(self._on_loading_task_started)
        tasks.task_message.connect(self._on_loading_task_message)
        tasks.task_failed.connect(self._on_loading_task_failed)
        tasks.scheduling_settled.connect(self._refresh_loading_status)

        # Do not leave a queue hold behind if this widget is destroyed.
        self._loading_lifetime = {"batch": None}
        lifetime = self._loading_lifetime
        self.destroyed.connect(
            lambda *_: (
                tasks.cancel_batch(lifetime["batch"])
                if lifetime["batch"] is not None
                else None
            )
        )

        load_button = make_icon_button(
            "plus", tooltip="Register new session data…", icon_size=30
        )
        load_button.clicked.connect(self.on_register_session)

        layout.addWidget(load_button, alignment=Qt.AlignmentFlag.AlignVCenter)

        self.selector_load_mode = QComboBox()
        self.selector_load_mode.addItems(["from file", ".* (glob)"])
        self.selector_load_mode.setMinimumWidth(100)
        self.selector_load_mode.setContentsMargins(3, 3, 3, 3)
        self.selector_load_mode.setSizeAdjustPolicy(
            QComboBox.SizeAdjustPolicy.AdjustToContents
        )
        self.selector_load_mode.setToolTip("Select how to load session data")
        layout.addWidget(
            self.selector_load_mode, alignment=Qt.AlignmentFlag.AlignVCenter
        )

        self.edit_load_glob = QLineEdit(
            str(
                parent.settings.value(
                    "session_registration/glob_pattern",
                    "Session0*/neuron*",
                )
                or ""
            ),
            placeholderText="Enter glob pattern",
        )

        self.edit_load_glob.textChanged.connect(
            lambda text: parent.settings.setValue(
                "session_registration/glob_pattern", text
            )
        )
        self.edit_load_glob.setTextMargins(6, 6, 6, 6)

        self._load_shortcuts = []

        for widget in (self.selector_load_mode, self.edit_load_glob):
            for key in (Qt.Key.Key_Return, Qt.Key.Key_Enter):
                shortcut = QShortcut(QKeySequence(key), widget)
                shortcut.setContext(Qt.ShortcutContext.WidgetShortcut)
                shortcut.setAutoRepeat(False)
                shortcut.activated.connect(load_button.click)
                self._load_shortcuts.append(shortcut)

        layout.addWidget(self.edit_load_glob, alignment=Qt.AlignmentFlag.AlignVCenter)
        layout.addStretch()
        self.edit_load_glob.setVisible(False)
        self.selector_load_mode.currentTextChanged.connect(
            lambda text: self.edit_load_glob.setVisible(text == ".* (glob)")
        )

        ## save button for sessions data
        self.button_save_sessions = make_icon_button(
            "floppy-disk", tooltip=f"Save sessions data", size=28, icon_size=22
        )
        self.button_save_sessions.setFixedWidth(48)
        self.button_save_sessions.setEnabled(False)
        layout.addWidget(
            self.button_save_sessions, alignment=Qt.AlignmentFlag.AlignRight
        )
        self.button_save_sessions.clicked.connect(lambda: self.save_data("sessions"))
        save_menu = QMenu(self.button_save_sessions)
        save_recipe_action = save_menu.addAction("Save loading recipe (.json)…")
        save_recipe_action.triggered.connect(self.save_loading_recipe)

        self.button_save_sessions.setMenu(save_menu)
        self.button_save_sessions.setPopupMode(
            QToolButton.ToolButtonPopupMode.MenuButtonPopup
        )

        self.registration_action_selector = RegistrationActionSelector(
            parent.settings, parent=self
        )
        self.data.background_orientation = (
            self.registration_action_selector.background_orientation
        )

        self.registration_action_selector.backgroundOrientationChanged.connect(
            lambda value: setattr(self.data, "background_orientation", value)
        )
        root_layout.addWidget(self.registration_action_selector)
        self._remembered_background_choice = None

        self.state.data_changed.connect(self._on_data_changed)

        # layout.addStretch()
        self._update_background()

    def _on_data_changed(self, input):
        # data_type, data_var = input
        # if data_type == "sessions":
        sessions_loaded = len(self.data.sessions) > 0
        self.button_save_sessions.setEnabled(sessions_loaded)

    def save_data(self, key):

        save_path = choose_path(
            self,
            pick_dir=False,
            init_path=str(self.data.root),
            display_text=f"Select folder to save {key} file to",
            only_existing=False,
            default_suffix="hdf5",
            file_filters=[
                ("HDF5 session file", ("*.hdf5", "*.h5")),
                ("MATLAB session file", ("*.mat",)),
                ("NumPy session file", ("*.npz",)),
            ],
            state=self.state,
            default_filename=f"catan_{key}.hdf5",
        )
        if save_path is None:
            return

        self.data.queue_save_sessions(save_path)

    def save_loading_recipe(self):
        try:
            document = self.data.loading_recipe_snapshot()
        except Exception as exc:
            self.state.issue(
                "warning",
                "Cannot save loading recipe",
                str(exc),
                parent=self.window(),
            )
            return

        path = choose_path(
            self,
            init_path=str(self.data.root),
            display_text="Save loading recipe",
            only_existing=False,
            default_suffix="json",
            file_filters=[
                ("CATAN loading recipe", ("*.json",)),
            ],
            state=self.state,
            default_filename="catan_loading_recipe.json",
        )

        if path is None:
            return

        self.state.tasks.start(
            "saving",
            "Saving loading recipe…",
            self.data.save_loading_recipe,
            path=path,
            document=document,
        )

    def _update_background(self):

        # Soft translucent background, so text remains readable.
        color = "#555555"
        self.setStyleSheet(f"""
            QFrame#SessionRowWidget {{
                background-color: {color};
                border: 2px solid rgba(255, 255, 255, 85);
                border-radius: 4px;
                font-weight: bold;
            }}
            """)

    def choose_sessions_from_glob(self):
        pattern = self.edit_load_glob.text()
        if not pattern.strip():
            self.state.issue(
                "info",
                "No file pattern",
                "Enter a file pattern before searching.",
                parent=self.window(),
            )
            return

        batch = TaskBatch()
        actions = set(self.registration_action_selector.actions)

        self.state.tasks.start(
            "loading",
            "Searching for session files…",
            self._discover_session_paths,
            str(self.data.root),
            pattern,
            batch=batch,
            on_result=lambda paths: self._review_glob_matches(
                paths,
                batch=batch,
                actions=actions,
            ),
        )

    @staticmethod
    def _discover_session_paths(root, pattern):
        return read_operation(
            "glob",
            root,
            pattern=pattern,
            ctx=current_task_context(),
            timeout=120.0,
            retry_hint=(
                "No sessions were registered by this search. Restore "
                "access to the root folder, or narrow the pattern, "
                "then run the glob search again."
            ),
        )

    def _review_glob_matches(self, paths, *, batch, actions):
        if batch.cancelled:
            return

        if not paths:
            self.state.issue(
                "info",
                "No matching session files",
                "No files matched the pattern. Check the root folder "
                "and pattern, then search again.",
                parent=self.window(),
            )
            return

        dialog = GlobReviewDialog(
            [Path(path) for path in paths],
            parent=self,
        )
        if dialog.exec() != QDialog.DialogCode.Accepted:
            self._cancel_registration_batch(batch)
            return

        if batch.cancelled:
            return

        for path in dialog.paths():
            if batch.cancelled:
                break

            self._queue_registration_path(
                path,
                batch=batch,
                actions=actions,
                force_first_session=False,
            )

    def _background_spec(self, session):

        config = session.source_config

        if config is None:
            return None

        spatial = config.groups.get("spatial")

        if spatial is None:
            return None

        return spatial.fields.get("background")

    def _ask_background_choice(self, *, configured_available: bool):
        dialog = QDialog(self.window())
        dialog.setWindowTitle("Background source")

        layout = QVBoxLayout(dialog)
        layout.setContentsMargins(20, 20, 20, 20)
        layout.setSpacing(12)

        description = QLabel(
            "Choose which background to use for this session."
            if configured_available
            else "The configured background is unavailable. "
            "Choose another source or construct it from footprints."
        )
        description.setWordWrap(True)
        layout.addWidget(description)

        remember = QCheckBox("Remember choice for following sessions")
        layout.addWidget(remember)

        result = {"mode": "cancel"}
        choice_buttons = []

        def choose(mode):
            result["mode"] = mode
            dialog.accept()

        choices = []
        if configured_available:
            choices.append(("Use configured background", "configured"))

        choices.extend(
            [
                ("Select background…", "select"),
                ("Construct from footprints", "footprints"),
            ]
        )

        for text, mode in choices:
            button = QPushButton(text)
            button.setAutoDefault(False)
            button.setMinimumHeight(button.fontMetrics().height() + 20)
            button.clicked.connect(lambda _checked=False, value=mode: choose(value))
            layout.addWidget(button)
            choice_buttons.append(button)

        cancel_row = QHBoxLayout()
        cancel_row.addStretch()

        cancel_button = QPushButton("Cancel")
        cancel_button.setAutoDefault(False)
        cancel_button.clicked.connect(dialog.reject)
        cancel_row.addWidget(cancel_button)
        layout.addLayout(cancel_row)

        # Account for both font scaling and the longest control label.
        dialog.setMinimumWidth(
            max(
                460,
                remember.sizeHint().width() + 40,
                max(button.sizeHint().width() for button in choice_buttons) + 40,
            )
        )
        dialog.adjustSize()

        try:
            accepted = dialog.exec() == QDialog.DialogCode.Accepted
            return (
                result["mode"] if accepted else "cancel",
                remember.isChecked() if accepted else False,
            )
        finally:
            dialog.deleteLater()

    def _apply_background_selection(self, session, selection: FieldSelection):

        config = session.source_config

        if config is None:
            return

        spatial = config.groups.get("spatial")

        if spatial is None:
            raise ValueError("Load configuration has no spatial group.")

        if "background" in spatial.fields:

            config.update_field(
                "spatial",
                "background",
                path=selection.path,
                source=selection.source,
                attribute=selection.attribute,
                source_path=selection.source_path,
            )

        else:

            config.add_field(
                "spatial",
                "background",
                path=selection.path,
                source=selection.source,
                attribute=selection.attribute,
                source_path=selection.source_path,
            )

    def _select_background_for_session(self, session) -> bool:

        selection = FieldSelectDialog.get_field(
            path=session.path,
            key="background",
            spec=self._background_spec(session),
            title="Select background",
            context=f"Session {session.name}",
            parent=self,
        )

        if selection is None:
            return False

        self._apply_background_selection(session, selection)

        return True

    def _prepare_background_for_registration(
        self,
        session,
        *,
        load_requested: bool,
        force_prompt: bool,
        configured_available: bool,
    ) -> str | None:
        """
        Returns
        -------
        "configured"
            Use the configured/selected background.

        "footprints"
            Construct the background from footprints.

        None
            Cancel remaining registration actions.
        """
        if session.status["spatial_loaded"]:
            return "configured"

        if session.source_config is None:
            return "configured"

        if not load_requested and not force_prompt:
            return "configured"

        needs_prompt = force_prompt or (load_requested and not configured_available)

        if not needs_prompt:
            return "configured"

        # ==========================================================
        # Remembered choice
        # ==========================================================

        remembered = self._remembered_background_choice

        if remembered == "footprints":
            return "footprints"

        if remembered == "configured":
            if configured_available:
                return "configured"

            # The next session does not actually have the
            # configured background. Ask again instead of
            # failing silently.
            self._remembered_background_choice = None

        elif remembered == "select":
            # remember that we want to SELECT another source,
            # not which source was selected previously.
            if self._select_background_for_session(session):
                return "configured"

            return None

        # ==========================================================
        # Ask what to do
        # ==========================================================

        mode, remember = self._ask_background_choice(
            configured_available=(configured_available)
        )

        if mode == "cancel":
            return None

        if mode == "configured":
            if remember:
                self._remembered_background_choice = "configured"

            return "configured"

        if mode == "footprints":
            if remember:
                self._remembered_background_choice = "footprints"

            return "footprints"

        if mode == "select":
            if not self._select_background_for_session(session):
                return None

            if remember:
                self._remembered_background_choice = "select"

            return "configured"

        raise ValueError(f"Unknown background mode: {mode!r}")

    def _on_sessions_registered(
        self,
        session_ids,
        *,
        actions,
        batch,
        force_first_session=False,
    ):
        if batch.cancelled or session_ids is None:
            return

        session_ids = list(session_ids)
        if not session_ids:
            return

        # Normally prevented by the queue hold. This also protects against
        # another explicit load request while an inspection is pending.
        if self._loading_check is not None:
            self.state.tasks.start(
                "loading",
                "Continue session registration",
                lambda ids=tuple(session_ids): ids,
                on_result=lambda ids: self._on_sessions_registered(
                    ids,
                    actions=actions,
                    batch=batch,
                    force_first_session=force_first_session,
                ),
                batch=batch,
            )
            return

        session_id = session_ids[0]
        session = self.data.sessions[session_id]
        effective_actions = set(actions)
        recipe_options = deepcopy(getattr(session, "_recipe_options", None))

        if recipe_options is not None:
            effective_actions.discard("load_all")
            effective_actions.discard("prompt_background")
            effective_actions.add("load_data")

        elif (
            force_first_session
            and len(session_ids) == 1
            and session_id == 0
            and not getattr(session, "_restored_from_catan", False)
        ):
            effective_actions.update({"load_data", "register_model", "track_neurons"})

        check = {
            "session": session,
            "remaining": session_ids[1:],
            "original_actions": set(actions),
            "actions": effective_actions,
            "recipe_options": recipe_options,
            "batch": batch,
            "token": None,
        }

        self._loading_check = check
        self._set_loading_status_batch(batch)

        check["token"] = self.state.tasks.pause_processing(
            "loading",
            batch=batch,
            on_cancel=lambda: self._discard_loading_check(check),
        )

        if self._loading_check is check:
            self._run_loading_step(self._begin_loading_check)

    def _run_loading_step(self, callback, *args):
        """Release the queue cleanly if a GUI continuation itself fails."""
        check = self._loading_check
        if check is None or check["batch"].cancelled:
            return

        try:
            callback(*args)
        except Exception as exc:
            details = traceback.format_exc()
            self.state.tasks.cancel_batch(check["batch"])
            self._remembered_background_choice = None

            self.state.issue(
                "error",
                "Session loading stopped",
                (
                    f"{exc}\n\n"
                    "Previously completed sessions have been retained. "
                    "Check this session's load configuration and source access, "
                    "then use its open-folder button to try again.\n\n"
                    f"{details}"
                ),
                parent=self.window(),
            )

    def _loading_config_signature(self, session):
        config = session.source_config
        fields = (
            config.get_fields_to_load(enabled_only=False)
            if config is not None
            else None
        )
        return pickle.dumps(
            (
                str(session.path),
                fields,
                None if config is None else config.dimensions,
            ),
            protocol=pickle.HIGHEST_PROTOCOL,
        )

    def _begin_loading_check(self):
        check = self._loading_check
        session = check["session"]
        actions = check["actions"]

        load_requested = bool({"load_data", "load_all"} & actions)
        check["load_requested"] = load_requested

        if not load_requested:
            self._begin_background_check()
            return

        if session.source_config is None:
            self._request_load_configuration(
                session,
                check["batch"],
                "No load configuration is selected.",
            )
            return

        fields = session.source_config.get_fields_to_load(
            enabled_only="load_all" not in actions
        )
        fields = self.data._missing_session_fields(session, fields)

        recipe = check["recipe_options"]
        skip_background = recipe is None or recipe["background_mode"] == "footprints"

        required = {}
        for group, specs in fields.items():
            selected = {
                name: spec
                for name, spec in specs.items()
                if spec.required
                and not (
                    skip_background and group == "spatial" and name == "background"
                )
            }
            if selected:
                required[group] = selected

        if not required:
            self._begin_background_check()
            return

        self._start_loading_inspection(
            "required",
            required,
            f"Checking required fields — {session.name}",
        )

    def _begin_background_check(self):
        check = self._loading_check
        session = check["session"]

        # Recipes already specify how the background should be obtained.
        if check["recipe_options"] is not None:
            self._complete_loading_check(False)
            return

        force_prompt = "prompt_background" in check["actions"]

        if (
            session.status["spatial_loaded"]
            or session.source_config is None
            or not (check["load_requested"] or force_prompt)
        ):
            self._complete_loading_check(False)
            return

        spec = self._background_spec(session)

        if session.path is None or spec is None:
            self._complete_loading_check(False)
            return

        self._start_loading_inspection(
            "background",
            {"spatial": {"background": spec}},
            f"Checking background source — {session.name}",
        )

    def _start_loading_inspection(self, stage, fields, title):
        check = self._loading_check
        check["stage"] = stage
        check["signature"] = self._loading_config_signature(check["session"])

        self.loading_status_title.setText(title)
        self.loading_status.show()
        self.loading_inspection.start(
            "compatibility",
            check["session"].path,
            fields_to_load=fields,
        )

    def _on_loading_inspected(self, report):
        self._run_loading_step(self._handle_loading_report, report)

    def _handle_loading_report(self, report):
        check = self._loading_check
        session = check["session"]

        # A user may have edited the configuration during the inspection.
        # Never apply a report produced for an older configuration.
        if check["signature"] != self._loading_config_signature(session):
            self._begin_loading_check()
            return

        if check["stage"] == "dimensions":
            if not report["ok"]:
                self._request_load_configuration(
                    session,
                    check["batch"],
                    report["message"],
                )
            else:
                self._queue_dimension_checked_session()
            return

        if check["stage"] == "background":
            available = bool(report.fields) and all(
                field.available for field in report.fields
            )
            self._complete_loading_check(available)
            return

        missing = [field for field in report.fields if not field.available]

        if missing:
            details = "\n".join(
                f"• {field.group}.{field.label}: {field.spec.path}"
                + (
                    f" (source: {field.spec.source_path})"
                    if field.spec.source_path
                    else ""
                )
                for field in missing
            )

            self._request_load_configuration(
                session,
                check["batch"],
                "The selected configuration refers to required fields "
                f"that are absent from the source:\n\n{details}",
            )
            return

        self._begin_background_check()

    def _on_loading_inspection_failed(self):
        if self._loading_check is None:
            return

        # Keep the queue paused. FileInspection already supplies Retry,
        # Details, and the source-specific recovery explanation.
        self.loading_inspection.cancel_button.setEnabled(True)
        self.loading_status_title.setText(
            "Loading paused — restore source access and Retry, " "or Cancel this batch."
        )

    def _complete_loading_check(self, configured_available):
        check = self._loading_check
        session = check["session"]
        batch = check["batch"]
        recipe = check["recipe_options"]

        if recipe is not None:
            background_mode = recipe["background_mode"]
        else:
            background_mode = self._prepare_background_for_registration(
                session,
                load_requested=check["load_requested"],
                force_prompt=("prompt_background" in check["actions"]),
                configured_available=configured_available,
            )

        # A genuine selection dialog may have run a nested event loop.
        if batch.cancelled or self._loading_check is not check:
            return

        if background_mode is None:
            self._cancel_registration_batch(batch)
            return

        check["background_mode"] = background_mode

        fields = (
            session.source_config.get_fields_to_load(
                enabled_only="load_all" not in check["actions"]
            )
            if session.source_config is not None
            else {}
        )
        fields = self.data._missing_session_fields(session, fields)

        if check["load_requested"] and "footprints" in fields.get("spatial", {}):
            if background_mode == "footprints":
                fields["spatial"].pop("background", None)

            check["stage"] = "dimensions"
            check["signature"] = self._loading_config_signature(session)

            self.loading_status_title.setText(
                f"Checking image dimensions — {session.name}"
            )

            self.loading_inspection.start(
                "dimensions",
                session.path,
                fields_to_load=fields,
                dimensions=session.source_config.dimensions,
                orientation=getattr(session, "_recipe_options", {}).get(
                    "background_orientation",
                    self.data.background_orientation,
                ),
                expected_dims=next(
                    (
                        tuple(other.dims)
                        for other in self.data.sessions
                        if other is not session and other.status["spatial_loaded"]
                    ),
                    None,
                ),
            )
            return

        self._queue_dimension_checked_session()

    def _queue_dimension_checked_session(self):
        check = self._loading_check
        session = check["session"]
        batch = check["batch"]
        background_mode = check["background_mode"]

        # Queue the remaining sessions first. queue_registration_actions()
        # then prepends the current session's data load ahead of them.
        # This preserves completed loads when a later session is cancelled.
        remaining = tuple(check["remaining"])
        original_actions = set(check["original_actions"])

        if remaining:
            self.state.tasks.start(
                "loading",
                "Continue session registration",
                lambda ids=remaining: ids,
                on_result=lambda ids: self._on_sessions_registered(
                    ids,
                    actions=original_actions,
                    batch=batch,
                    force_first_session=False,
                ),
                batch=batch,
                prepend=True,
            )

        self.data.queue_registration_actions(
            session.id,
            check["actions"],
            background_mode=background_mode,
            on_alignment_error=self._show_alignment_error,
            batch=batch,
        )

        token = check["token"]
        self._discard_loading_check(check)
        self.state.tasks.resume_processing(token)

    def _discard_loading_check(self, check):
        if self._loading_check is not check:
            return

        self._loading_check = None

        if isValid(self.loading_inspection):
            self.loading_inspection.invalidate()

    def _cancel_inline_loading(self):
        batch = self._loading_status_batch
        if batch is not None and not batch.cancelled:
            self._cancel_registration_batch(batch)

    def _show_alignment_error(self, session_id: int, report):

        session = self.data.sessions[session_id]

        text = (
            f"Automatic alignment failed for {session.name!r}.\n\n"
            "The session has been kept loaded, but no neurons were registered."
        )

        if report is not None:

            if report.reason is not None:
                text += "\n\nReason: " + report.reason.replace("_", " ")

            if report.shift is not None:
                text += f"\nShift: {report.total_shift:.1f} px"

            if report.rotation is not None:
                text += f"\nRotation: {report.rotation:.2f}°"

            if report.correlation is not None:
                text += f"\nCorrelation: {report.correlation:.3f}"

        self.state.alignment_failed.emit(session, text, session.remap)

    def on_register_session(self):
        opt = self.selector_load_mode.currentText()

        if opt == ".* (glob)":
            self.choose_sessions_from_glob()
            return

        if opt.lower() != "from file":
            raise ValueError(f"Unknown option selected: {opt}")

        path = choose_path(
            self,
            pick_dir=False,
            init_path=self.data.root,
            display_text="Select session file or loading recipe",
            only_existing=True,
        )
        if path is None:
            return

        self._queue_registration_path(
            Path(path),
            batch=TaskBatch(),
            actions=set(self.registration_action_selector.actions),
            force_first_session=not self.data.sessions,
        )

    def _cancel_registration_batch(self, batch):
        if batch.cancelled:
            return

        self._remembered_background_choice = None
        self.state.tasks.cancel_batch(batch)
        self._refresh_loading_status()

    def _queue_registration_path(
        self, path, *, batch, actions, force_first_session=False
    ):
        if batch.cancelled:
            return

        path = Path(path)

        if path.suffix.lower() == ".json":
            self.state.tasks.start(
                "loading",
                f"Reading loading recipe: {path.name}",
                self.data._read_loading_recipe,
                path,
                batch=batch,
                on_result=lambda sessions: self._review_recipe(
                    sessions,
                    batch=batch,
                    actions=actions,
                ),
            )
            return

        self.state.tasks.start(
            "loading",
            f"Registering session: {path.name}",
            self.data.register_session,
            from_file=path,
            batch=batch,
            on_result=lambda session_ids: self._on_sessions_registered(
                session_ids,
                actions=actions,
                force_first_session=force_first_session,
                batch=batch,
            ),
        )

    def _review_recipe(self, sessions, *, batch, actions):
        if batch.cancelled:
            return

        existing_by_path = {
            self.data.session_source_key(session.path): session
            for session in self.data.sessions
            if session.path is not None
        }

        rows = []
        for proposed in sessions:
            existing = existing_by_path.get(self.data.session_source_key(proposed.path))

            if existing is None:
                status = "Not registered"
                checked = True
            elif existing.source_config is None:
                status = "Registered; configuration required"
                checked = True
            else:
                fields = existing.source_config.get_fields_to_load()
                missing = self.data._missing_session_fields(existing, fields)

                if missing:
                    status = "Registered; data loading incomplete"
                    checked = True
                else:
                    status = "Configured data already loaded"
                    checked = False
                    if (
                        existing.status["spatial_loaded"]
                        and not existing.status["aligned"]
                    ):
                        status += "; alignment needs review"

            rows.append(
                {
                    "name": proposed.name or Path(proposed.path).parent.name,
                    "path": str(proposed.path),
                    "status": status,
                    "checked": checked,
                }
            )

        dialog = RecipeReviewDialog(rows, parent=self)
        if dialog.exec() != QDialog.DialogCode.Accepted:
            self._cancel_registration_batch(batch)
            return

        if batch.cancelled:
            return

        # Avoid duplicate source entries within this recipe selection.
        seen = set()
        for index in dialog.selected_indices():
            session = sessions[index]
            key = self.data.session_source_key(session.path)
            if key in seen:
                continue
            seen.add(key)

            self.state.tasks.start(
                "loading",
                f"Registering recipe session: {session.name or session.path}",
                self.data.register_session,
                prepared_sessions=[session],
                batch=batch,
                on_result=lambda session_ids: self._on_sessions_registered(
                    session_ids,
                    actions=actions,
                    batch=batch,
                ),
            )

    def _request_load_configuration(self, session, batch, explanation):
        # Stop downstream registration without discarding completed sessions.
        self.state.tasks.cancel_batch(batch)
        self._remembered_background_choice = None

        session.status["loading_possible"] = False
        self.loadConfigRequired.emit(session.id)

        self.state.issue(
            "info",
            "Choose or repair the load configuration",
            (
                f"Session: {session.name}\n\n"
                f"{explanation}\n\n"
                "Loading has stopped before reading the session data. "
                "Previously loaded sessions are retained.\n\n"
                "Choose a matching preset or correct the required field "
                "paths in the expanded configuration. Then click this "
                "session's open-folder button to load it.\n\n"
                "To resume the remaining batch, reopen the JSON recipe "
                "or glob selection and select the unfinished sessions."
            ),
            parent=self.window(),
        )

    def _set_loading_status_batch(self, batch):
        if self._loading_status_batch is not batch:
            self._loading_had_error = False
            self._loading_status_task_ids.clear()

        self._loading_status_batch = batch
        self._loading_lifetime["batch"] = batch
        self.loading_status.show()

        self.loading_inspection.show()

        for button in (
            self.loading_inspection.retry_button,
            self.loading_inspection.cancel_button,
            self.loading_inspection.details_button,
        ):
            button.show()

    def _on_loading_task_started(self, group, task_id):
        if group != "loading":
            return

        task = self.state.tasks.get_task(task_id)
        if task is None or task.batch is None:
            return

        self._set_loading_status_batch(task.batch)
        self._loading_status_task_ids.add(task_id)

        self.loading_status_title.setText(task.name)
        self.loading_inspection.label.setText("Loading…")
        self.loading_inspection.retry_button.setEnabled(False)
        self.loading_inspection.details_button.setEnabled(False)
        self.loading_inspection.cancel_button.setEnabled(True)

    def _on_loading_task_message(self, group, task_id, message):
        if group != "loading" or self._loading_check is not None:
            return

        task = self.state.tasks.get_task(task_id)
        if task is not None and task.batch is self._loading_status_batch:
            self.loading_inspection.set_message(message)

    def _on_loading_task_failed(self, group, task_id):
        if group == "loading" and task_id in self._loading_status_task_ids:
            self._loading_had_error = True

    def _refresh_loading_status(self):
        if self._loading_check is not None:
            return

        batch = self._loading_status_batch
        if batch is None:
            return

        pending = any(task.batch is batch for task in self.state.tasks.tasks.values())

        self.loading_inspection.retry_button.setEnabled(False)
        self.loading_inspection.cancel_button.setEnabled(
            pending and not batch.cancelled
        )

        if batch.cancelled:
            self.loading_status_title.setText("Loading stopped")
            self.loading_inspection.label.setText(
                "Previously completed sessions are retained. "
                "Select unfinished sessions again to continue."
            )
        elif pending:
            self.loading_status_title.setText("Loading and registration")
        elif self._loading_had_error:
            self.loading_status_title.setText("Loading finished with an error")
            self.loading_inspection.label.setText(
                "Review the reported issue. After correcting the source or "
                "configuration, use the session's open-folder button to retry."
            )
        else:
            self.loading_status_title.setText("Ready for loading further data")
            self.loading_inspection.label.clear()
            self.loading_inspection.hide()

        if not pending:
            self._loading_lifetime["batch"] = None

            for button in (
                self.loading_inspection.retry_button,
                self.loading_inspection.cancel_button,
                self.loading_inspection.details_button,
            ):
                button.hide()

    def _format_loading_message(self, message):
        root = self.data.root
        if root is None or not str(root).strip():
            return message

        # Purely lexical: do not resolve/stat a potentially remote path.
        root = os.path.abspath(os.path.normpath(os.fspath(root)))

        # Accept either separator on Windows.
        separator = r"[\\/]" if os.name == "nt" else "/"
        root_text = root.rstrip("\\/") if os.name == "nt" else root.rstrip("/")

        if os.name == "nt":
            root_pattern = re.escape(root_text.replace("\\", "/"))
            root_pattern = root_pattern.replace("/", r"[\\/]")
        else:
            root_pattern = re.escape(root_text)

        # Match the root only at the beginning of a path and require
        # a separator afterwards: /data must not match /database.
        pattern = r"(?<![\w./\\-])" + root_pattern + separator

        return re.sub(
            pattern,
            "",
            str(message),
            flags=re.IGNORECASE if os.name == "nt" else 0,
        )


class SessionList(QListWidget):
    drag_n_dropped = Signal(int, int)  # old_index, new_index
    refresh_requested = Signal()

    def __init__(self, parent: "SessionOverview"):
        super().__init__(parent)
        self._state = parent.state
        self.data = parent.data

        self.setDragDropMode(QAbstractItemView.DragDropMode.InternalMove)
        self.setDefaultDropAction(Qt.DropAction.MoveAction)

        self.setSelectionMode(QAbstractItemView.SelectionMode.SingleSelection)
        self.setStyleSheet("""
            QListWidget::item {
                background: transparent;
                border: none;
            }

            QListWidget::item:selected {
                background: transparent;
                border: none;
            }

            QListWidget::item:selected:active {
                background: transparent;
                border: none;
            }

            QListWidget::item:selected:!active {
                background: transparent;
                border: none;
            }
            """)

    def dropEvent(self, event):
        item = self.currentItem()
        old_index = self.row(item)
        super().dropEvent(event)

        new_index = self.row(item)

        print(f"Session moved from {old_index} to {new_index}")
        if old_index == new_index:
            return

        self.drag_n_dropped.emit(old_index, new_index)


class SessionOverview(QWidget):
    load_requested = Signal(int)  # session_id

    def __init__(self, parent):
        super().__init__(parent)

        self.data: Data = parent.data
        self.state: AppState = parent.state
        self.settings = parent.settings

        self.setSizePolicy(
            QSizePolicy.Policy.Expanding,
            QSizePolicy.Policy.Expanding,
        )

        self.list_widget = SessionList(parent=self)
        self.list_widget.setSpacing(3)
        self.list_widget.itemDoubleClicked.connect(self._on_item_double_clicked)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self.list_widget)
        self.load_row = LoadSessionRowWidget(self)
        layout.addWidget(self.load_row)

        self.load_row.loadConfigRequired.connect(self._show_load_configuration)

        self.list_widget.setDragDropMode(QAbstractItemView.DragDropMode.InternalMove)
        self.list_widget.setDefaultDropAction(Qt.DropAction.MoveAction)

        self._row_widgets: dict[int, SessionRowWidget] = {}

        self.state.data_changed.connect(self._on_data_changed)
        self.state.current_session_changed.connect(self._on_current_session_changed)
        self.list_widget.drag_n_dropped.connect(self.move_session)
        self.list_widget.refresh_requested.connect(self.refresh_rows)

        self._alignment_failures = []
        self._alignment_popup_active = False
        self._alignment_failure_timer = QTimer(self)
        self._alignment_failure_timer.setSingleShot(True)
        self._alignment_failure_timer.timeout.connect(self._open_next_alignment_failure)
        self.state.alignment_failed.connect(self._queue_alignment_failure)
        self.state.alignment_review_finished.connect(self._finish_alignment_review)
        self.state.alignment_review_changed.connect(self._schedule_alignment_review)

        self.rebuild()

    def rebuild(self):
        # print("rebuilding!")
        self.list_widget.clear()
        self._row_widgets.clear()

        for session in self.data.sessions:
            # print("Adding session row:", session.id, getattr(session, "name", None))
            self._add_session_row(session.id, session)

    def _add_session_row(self, session_id: int, session):
        item = QListWidgetItem()
        item.setData(Qt.ItemDataRole.UserRole, session_id)

        row = SessionRowWidget(
            session_id,
            item,
            session,
            current=session_id == self.state.current_session_id,
            parent=self.list_widget,
        )

        row.moveRequested.connect(self.move_session)
        row.activeChanged.connect(self.set_session_active)
        row.nameChanged.connect(self.rename_session)

        row.setCurrentRequested.connect(self.set_current_session)
        row.editOffsetRequested.connect(self.edit_time_offset)
        # row.changeColorRequested.connect(self.change_session_color)

        row.loadRequested.connect(lambda id=session_id: self.load_requested.emit(id))
        row.backgroundRequested.connect(self.change_session_background)
        row.manualAlignmentRequested.connect(self.adjust_session_alignment)

        row.traceToggled.connect(
            lambda id, which="traces": self.toggle_session_data(id, which)
        )
        row.qualityToggled.connect(
            lambda id, which="quality": self.toggle_session_data(id, which)
        )
        row.spatialToggled.connect(
            lambda id, which="spatial": self.toggle_session_data(id, which)
        )

        row.modelRequested.connect(self.toggle_model)
        row.assignmentRequested.connect(self.toggle_assignments)
        row.removeRequested.connect(self.remove_session)

        item.setSizeHint(row.sizeHint())

        # self.list_widget.addItem(item)
        self.list_widget.insertItem(row.index, item)
        self.list_widget.setItemWidget(item, row)

        self._row_widgets[session_id] = row

    def update_row_height(
        self,
        item: QListWidgetItem,
        row: QWidget,
    ):
        # A list rebuild may have deleted these since this was queued.
        if not isValid(self) or not isValid(item) or not isValid(row):
            return

        if not isValid(self.list_widget):
            return

        # Ignore rows that have been removed or replaced.
        if self.list_widget.itemWidget(item) is not row:
            return

        layout = row.layout()
        if layout is not None:
            layout.activate()

        row.adjustSize()

        # Layout changes can trigger further UI updates.
        if isValid(item) and isValid(row):
            item.setSizeHint(row.sizeHint())

    def _on_data_changed(self, event: DataChange):
        if event.has(*SESSION_STRUCTURE_CHANGES):
            self.rebuild()
            return

        if event.has(
            C.SESSION_METADATA,
            C.SESSION_ACTIVITY,
            C.SESSION_TIMEBASE,
            C.DATA_AVAILABILITY,
            C.FOOTPRINT_GEOMETRY,
            C.BACKGROUND_IMAGE,
            C.ASSIGNMENT_SET,
            C.ASSIGNMENT_MAPPING,
            C.INCLUSION,
            C.MODEL_COUNTS,
            C.MODEL_PARAMETERS,
            C.PROCESSING_STATUS,
        ):
            self.refresh_rows()

    def _on_current_session_changed(self):
        self.refresh_rows()

    def refresh_rows(self):
        """
        Use this when session properties changed but the order did not.
        """
        session_ids = [s.id for s in self.data.sessions]
        for row in list(self._row_widgets.values()):
            if not (row.session.id in session_ids):
                self.list_widget.takeItem(self.list_widget.row(row.item))
                del self._row_widgets[row.session.id]
                continue

            order_index = self.list_widget.row(row.item)
            row.index = order_index

            self._row_widgets[order_index] = row

            assert (
                order_index == row.session.id
            ), f"Row index {order_index} does not match session id {row.session.id}"

            row.refresh(current=row.session.id == self.state.current_session_id)

    def _on_item_double_clicked(self, item: QListWidgetItem):
        session_id = item.data(Qt.ItemDataRole.UserRole)
        self.set_current_session(session_id)

    def _show_load_configuration(self, session_id):
        row = self._row_widgets.get(session_id)

        if row is None:
            self.rebuild()
            row = self._row_widgets.get(session_id)

        if row is None:
            return

        for index in range(self.list_widget.count()):
            item = self.list_widget.item(index)
            if item.data(Qt.ItemDataRole.UserRole) == session_id:
                self.list_widget.setCurrentItem(item)
                self.list_widget.scrollToItem(item)
                break

        def open_config():
            if isValid(row):
                row.config_constructor.toggle_config_options.set_expanded(True)

        QTimer.singleShot(0, open_config)

    def set_current_session(self, session_id: int):
        self.state.current_session_id = session_id

    def set_session_active(self, session_id: int, active: bool):
        session = self.data.sessions[session_id]
        session.active = active

        self.data.notify_change(
            C.SESSION_ACTIVITY,
            session_id=session_id,
        )

    def rename_session(self, session_id: int, name: str):
        if not name:
            self.refresh_rows()
            return

        self.data.sessions[session_id].name = name
        self.data.notify_change(
            C.SESSION_METADATA,
            session_id=session_id,
        )

    def edit_time_offset(self, session_id: int):
        session = self.data.sessions[session_id]
        old_value = int(getattr(session, "time_offset", 0))

        frame_hint = ""
        if session.status["traces_loaded"]:
            frame_hint = f"Number of frames in file: {session.trace.shape[1]}"

        value, ok = QInputDialog.getInt(
            self,
            "Trace time offset",
            f"Offset for {getattr(session, 'name', session_id)}. {frame_hint}:",
            old_value,
            -10_000_000,
            10_000_000,
            1,
        )

        if not ok:
            return

        session.time_offset = value
        self.refresh_rows()

        self.data.notify_change(
            C.SESSION_TIMEBASE,
            session_id=session_id,
        )

    def adjust_session_alignment(self, session_id):
        open_manual_alignment(self.data, session_id, self)

    def change_session_background(self, session_id: int):

        session = self.data.sessions[session_id]

        if not session.status["spatial_loaded"]:
            config = session.source_config

            if config is None or "spatial" not in config.groups:
                self.state.issue(
                    "warning",
                    "Spatial configuration required",
                    "Choose a load configuration containing the spatial "
                    "footprint fields in this session's options, then "
                    "select the background again.",
                )
                return

            selection = FieldSelectDialog.get_field(
                path=session.path,
                key="background",
                spec=self.load_row._background_spec(session),
                title="Select background and load spatial data",
                context=f"Session: {session.name}",
                parent=self,
            )
            if selection is None:
                return

            self.load_row._apply_background_selection(session, selection)

            # Explicit recovery choice supersedes an earlier recipe choice
            # to construct the background from footprints.
            recipe_options = getattr(session, "_recipe_options", None)
            if recipe_options is not None:
                session._recipe_options = {
                    **recipe_options,
                    "background_mode": "configured",
                }

            session._load_background_mode = "configured"

            fields_to_load = config.get_fields_to_load(["spatial"])

            self.data.queue_load_data(
                session_id,
                fields_to_load=fields_to_load,
                finished=self.refresh_rows,
                batch=TaskBatch(),
            )
            return

        expected_version = self.state.data_version

        spec = None
        if session.source_config is not None:

            spatial = session.source_config.groups.get("spatial")
            if spatial is not None:
                spec = spatial.fields.get("background")

        selection = FieldSelectDialog.get_field(
            path=session.path,
            key="background",
            spec=spec,
            title="Select replacement background",
            context=(f"Session: {session.name}"),
            parent=self,
        )

        if selection is None:
            return

        source_path = resolve_source_path(session.path, selection.source_path)

        self.state.tasks.start(
            "loading",
            ("Testing background for " f"{session.name}"),
            lambda: (
                self.data.propose_background_remapping(
                    session_id,
                    source_path=source_path,
                    field_path=selection.path,
                    source=selection.source,
                    attribute=selection.attribute,
                )
            ),
            on_result=lambda result: self._show_background_proposal(
                session_id, selection, result, expected_version=expected_version
            ),
        )

    def _queue_alignment_failure(self, session, message, proposal):
        self._alignment_failures = [
            entry for entry in self._alignment_failures if entry[0] is not session
        ]
        self._alignment_failures.append((session, message, proposal))
        self._schedule_alignment_review()

    def _schedule_alignment_review(self, *_):
        if not self._alignment_popup_active:
            self._alignment_failure_timer.start(0)

    def _finish_alignment_review(self, session):
        review = self.state.alignment_review
        if review is None or review["session"] is not session:
            return

        self.state.alignment_review = None
        self.state.alignment_review_changed.emit()
        self._schedule_alignment_review()

    def _open_next_alignment_failure(self):
        if self._alignment_popup_active:
            return

        review = self.state.alignment_review
        if review is None and not self._alignment_failures:
            return

        tasks = self.state.tasks
        if (
            tasks.processing_busy()
            or tasks.processing_requested
            or QApplication.activeModalWidget() is not None
        ):
            self._alignment_failure_timer.start(150)
            return

        panels = sorted(
            (panel for panel in self.state.alignment_panels if not panel._disposed),
            key=lambda panel: not panel.isVisible(),
        )

        if review is None:
            session, message, proposal = self._alignment_failures[0]
            session_id = next(
                (
                    index
                    for index, item in enumerate(self.data.sessions)
                    if item is session
                ),
                None,
            )

            if (
                session_id is None
                or not session.status["spatial_loaded"]
                or session.background_template is None
            ):
                self._alignment_failures.pop(0)
                self._schedule_alignment_review()
                return

            parent = panels[0] if panels else self
            draft = select_alignment_draft(
                self.data,
                session_id,
                parent,
                proposal=proposal,
            )
            if draft is None:
                # Keep the request queued when discarding an existing draft
                # was declined. Do not repeatedly reopen that confirmation.
                return

            self._alignment_failures.pop(0)
            review = {
                "session": session,
                "message": message,
                "proposal": proposal,
            }
            self.state.alignment_review = review
            self.state.alignment_review_changed.emit()

            if panels:
                panel = panels[0]
                panel._sync_draft()
                panel.raise_()
                panel.setFocus()
                return

        elif panels:
            # A panel already owns this review. Wait for Use / Remove / Later.
            return

        session = review["session"]
        session_id = next(
            (index for index, item in enumerate(self.data.sessions) if item is session),
            None,
        )
        if session_id is None:
            self._finish_alignment_review(session)
            return

        self._alignment_popup_active = True
        try:
            # The shared draft already contains the reviewed proposal.
            # The editor displays the shared failure message.
            open_manual_alignment(self.data, session_id, self)
        finally:
            self._alignment_popup_active = False
            self._finish_alignment_review(session)
            self._schedule_alignment_review()

    def toggle_session_data(
        self, session_id: int, which: Optional[sessiondata_type] = None
    ):
        session = self.data.sessions[session_id]
        self.state.tasks.start(
            "loading",
            f"Loading {which} data for {session.name}",
            lambda: self.data.toggle_session_data(session_id, which),
            finished=self.refresh_rows,
        )

    def toggle_model(self, session_id: int):

        self.data.queue_update_model(
            session_id,
            to_present=not self.data.sessions[session_id].status["registered_to_model"],
            callback=self.refresh_rows,
        )

    def toggle_assignments(self, session_id: int):

        self.data.queue_assign_neurons(
            session_id,
            to_present=not self.data.session_assigned(session_id),
            callback=self.refresh_rows,
        )

    def remove_session(self, session_id: int):
        session = self.data.sessions[session_id]
        name = getattr(session, "name", f"Session {session_id}")

        result = QMessageBox.question(
            self,
            "Remove session",
            f"Remove {name} from the project?",
        )

        if result != QMessageBox.StandardButton.Yes:
            return

        # Important: use one central method for this if session_id appears
        # in assignments, plots, tracking arrays, caches, etc.
        self.data.remove_session(session_id)

    def move_session(self, session_index: int, new_session_index: int):
        # new_id = session_index + delta

        if new_session_index < 0 or new_session_index >= len(self.data.sessions):
            return

        print(f"Moving session {session_index} to {new_session_index}")
        self.data.move_session(session_index, new_session_index)

        # self.refresh_rows()
        # print(f"New session order: {[s.id for s in self.data.sessions]}")

        # currentItem = self.list_widget.takeItem(session_index)
        # self.list_widget.insertItem(new_session_index, currentItem)

        # self.list_widget.setCurrentRow(new_session_index)

    @staticmethod
    def _alignment_summary(report) -> str:

        if report is None:
            return "No alignment available"

        lines = ["successful" if report.success else "FAILED"]

        if report.reason:
            lines.append("reason: " + str(report.reason).replace("_", " "))

        if report.shift is not None:

            shift = np.asarray(report.shift).ravel()

            if shift.size >= 2:
                lines.append(f"shift: dy={shift[0]:.2f}px, " f"dx={shift[1]:.2f}px")

        if report.rotation is not None:
            lines.append("rotation: " f"{report.rotation:.2f}°")

        if report.correlation is not None:
            lines.append("correlation: " f"{report.correlation:.3f}")

        if report.correlation_zscore is not None:
            lines.append("z-score: " f"{report.correlation_zscore:.2f}")

        lines.append(f"method: {report.method}")

        return "\n".join(lines)

    def _show_background_proposal(
        self,
        session_id,
        selection,
        result,
        *,
        expected_version,
    ):
        if self.state.data_version != expected_version:
            self.state.issue(
                "warning",
                "Alignment preview outdated",
                "The data changed while calculating the preview. Try again.",
            )
            return

        candidate_template, candidate_remap = result

        session = self.data.sessions[session_id]

        current_report = None if session.remap is None else session.remap.report

        proposed_report = candidate_remap.report

        dialog = QMessageBox(self)

        dialog.setWindowTitle("Background alignment preview")

        dialog.setIcon(
            QMessageBox.Icon.Information
            if proposed_report.success
            else QMessageBox.Icon.Warning
        )

        dialog.setText(f"Alignment preview for " f"{session.name}")

        dialog.setInformativeText(
            "Current alignment\n"
            "-----------------\n" + self._alignment_summary(current_report) + "\n\n"
            "Proposed alignment\n"
            "------------------\n" + self._alignment_summary(proposed_report) + "\n\n"
            "No changes have been applied yet."
        )

        dialog.setStandardButtons(QMessageBox.StandardButton.Close)

        apply_button = dialog.addButton(QMessageBox.StandardButton.Apply)
        apply_button.setEnabled(proposed_report.success)

        dialog.exec()

        if dialog.clickedButton() is not apply_button:
            return

        source_path = resolve_source_path(
            session.path,
            selection.source_path,
        )

        background_spec = FieldSpec(
            path=selection.path,
            source=selection.source,
            attribute=selection.attribute,
            source_path=os.path.abspath(os.path.expanduser(os.fspath(source_path))),
            required=True,
        )

        self.data.queue_commit_session_realignment(
            session_id,
            background_template=candidate_template,
            remap=candidate_remap,
            background_spec=background_spec,
            expected_version=expected_version,
        )
