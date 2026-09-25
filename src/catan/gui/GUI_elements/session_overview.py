from typing import Optional
from unicodedata import name

from PySide6.QtCore import QSize, QTimer, Qt, Signal, QPoint
from PySide6.QtGui import QColor, QAction
from PySide6.QtWidgets import (
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
)

import numpy as np
from pathlib import Path

from catan.gui.structures import AppState, Data, SessionData
from catan.core.structures import sessiondata_type
from catan.core.structures.load_config import FieldSpec
from catan.core.io import resolve_source_path

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


class SessionRowWidget(QFrame):
    moveRequested = Signal(int, int)  # session_id, delta
    activeChanged = Signal(int, bool)  # session_id, active
    nameChanged = Signal(int, str)  # session_id, new_name

    setCurrentRequested = Signal(int)
    editOffsetRequested = Signal(int)
    changeColorRequested = Signal(int)

    loadRequested = Signal(int)  # session_id
    backgroundRequested = Signal(int)

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
        stacked_layout.addWidget(self.config_constructor.config_options)

        self.refresh(current=current)

    def _on_data_changed(self, input):

        data_type, data_var = input

        relevant_session = data_var in (self.session.id, -1)

        if data_type in ("session", "sessions") and relevant_session:
            self._update_buttons()
            self._update_status()

            self.config_constructor.config_field_options.rebuild()
            return

        if data_type in ("assignments", "model"):
            self._update_buttons()
            self._update_status()

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
        self.config_constructor._on_fields_changed()

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
            outdated.append("alignment")

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

        if self.session.status["spatial_loaded"]:
            menu.addAction(
                "Change background…",
                lambda: (self.backgroundRequested.emit(self.index)),
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
            "Session0*/neuron*", placeholderText="Enter glob pattern"
        )
        self.edit_load_glob.setTextMargins(6, 6, 6, 6)

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
        self.button_save_sessions.setFixedWidth(35)
        self.button_save_sessions.setEnabled(False)
        layout.addWidget(
            self.button_save_sessions, alignment=Qt.AlignmentFlag.AlignRight
        )
        self.button_save_sessions.clicked.connect(lambda: self.save_data("sessions"))

        self.registration_action_selector = RegistrationActionSelector(
            parent.settings, parent=self
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
            init_path=str(Path(self.data.root) / f"catan_{key}.hdf5"),
            display_text=f"Select folder to save {key} file to",
            only_existing=False,
        )
        if save_path is None:
            return

        # if key == "sessions":
        self.data.save_sessions(save_path)

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

    def on_register_session(self):

        opt = self.selector_load_mode.currentText()

        if opt.lower() == "from file":
            ## chooses automatically between loading from single detection session or from list of sessions (from hdf5 attribute)
            path = choose_path(
                self,
                pick_dir=False,
                init_path=self.data.root,
                display_text="Select session file",
                only_existing=True,
            )
            if path is None:
                return
            self.state.tasks.start(
                "loading",
                "Loading session data from file...",
                lambda: self.data.register_session(from_file=path),
            )

        elif opt == ".* (glob)":
            self.choose_sessions_from_glob()
        else:
            raise ValueError(f"Unknown option selected: {opt}")

    def choose_sessions_from_glob(self):
        root = Path(self.data.root)
        pattern = self.edit_load_glob.text()
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

        actions = set(self.registration_action_selector.actions)

        for path in paths:

            session_ids = self.data.register_session(from_file=path)

            self._on_sessions_registered(
                session_ids,
                actions=actions,
                force_first_session=False,
            )
        return

    def _background_spec(self, session):

        config = session.source_config

        if config is None:
            return None

        spatial = config.groups.get("spatial")

        if spatial is None:
            return None

        return spatial.fields.get("background")

    def _ask_background_choice(self, *, configured_available: bool):

        dialog = QMessageBox(self)

        dialog.setWindowTitle("Background source")

        if configured_available:
            dialog.setText("Choose which background to use for this session.")
        else:
            dialog.setText("No configured background could be loaded for this session.")

        configured_button = None

        if configured_available:
            configured_button = dialog.addButton(
                "Use configured background", QMessageBox.ButtonRole.AcceptRole
            )

        select_button = dialog.addButton(
            "Select background…", QMessageBox.ButtonRole.AcceptRole
        )

        footprints_button = dialog.addButton(
            "Construct from footprints", QMessageBox.ButtonRole.ActionRole
        )

        dialog.addButton(QMessageBox.StandardButton.Cancel)

        remember = QCheckBox("Remember choice for following sessions")

        dialog.setCheckBox(remember)
        dialog.exec()

        clicked = dialog.clickedButton()

        if configured_button is not None and clicked is configured_button:
            mode = "configured"
        elif clicked is select_button:
            mode = "select"
        elif clicked is footprints_button:
            mode = "footprints"
        else:
            mode = "cancel"

        return mode, remember.isChecked()

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

        configured_available = self.data.session_field_available(
            session.id, "spatial", "background"
        )

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
        self, session_ids, *, actions, force_first_session=False
    ):

        if session_ids is None:
            return

        session_ids = list(session_ids)
        single_new_session = len(session_ids) == 1

        for session_id in session_ids:

            effective_actions = set(actions)

            # First individually registered CATAN session:
            # always load data + register to model + track,
            # without changing the globally selected defaults.
            if force_first_session and single_new_session and session_id == 0:
                effective_actions.update(
                    {"load_data", "register_model", "track_neurons"}
                )

            session = self.data.sessions[session_id]
            load_requested = bool(effective_actions & {"load_data", "load_all"})

            background_mode = self._prepare_background_for_registration(
                session,
                load_requested=load_requested,
                force_prompt=("prompt_background" in effective_actions),
            )

            if background_mode is None:
                continue

            self.data.queue_registration_actions(
                session_id,
                effective_actions,
                background_mode=background_mode,
                on_alignment_error=self._show_alignment_error,
            )

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

        text += "\n\nYou can inspect and correct " "the alignment later."

        QMessageBox.information(self, "Alignment error", text)

    def on_register_session(self):

        opt = self.selector_load_mode.currentText()

        actions = set(self.registration_action_selector.actions)

        if opt.lower() == "from file":

            path = choose_path(
                self,
                pick_dir=False,
                init_path=self.data.root,
                display_text="Select session file",
                only_existing=True,
            )

            if path is None:
                return

            first_session = len(self.data.sessions) == 0

            self.state.tasks.start(
                "loading",
                "Registering session from file...",
                self.data.register_session,
                from_file=path,
                on_result=lambda session_ids: (
                    self._on_sessions_registered(
                        session_ids,
                        actions=actions,
                        force_first_session=first_session,
                    )
                ),
            )

        elif opt == ".* (glob)":
            self.choose_sessions_from_glob()

        else:
            raise ValueError(f"Unknown option selected: {opt}")


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

        self.list_widget.setDragDropMode(QAbstractItemView.DragDropMode.InternalMove)
        self.list_widget.setDefaultDropAction(Qt.DropAction.MoveAction)

        self._row_widgets: dict[int, SessionRowWidget] = {}

        self.state.data_changed.connect(self._on_data_changed)
        self.state.current_session_changed.connect(self._on_current_session_changed)
        self.list_widget.drag_n_dropped.connect(self.move_session)
        self.list_widget.refresh_requested.connect(self.refresh_rows)
        self.rebuild()

    def rebuild(self):
        # print("rebuilding!")
        self.list_widget.clear()
        self._row_widgets.clear()

        for session in self.data.sessions:
            # print("Adding session row:", session.id, getattr(session, "name", None))
            self._add_session_row(session.id, session)

        # self._add_load_row()

    # def _add_load_row(self):
    #     item = QListWidgetItem()
    #     flags = item.flags()
    #     flags &= ~Qt.ItemFlag.ItemIsDragEnabled
    #     flags &= ~Qt.ItemFlag.ItemIsDropEnabled
    #     item.setFlags(flags)

    #     row = LoadSessionRowWidget(self,item)

    #     item.setSizeHint(row.sizeHint())
    #     # self.list_widget.addItem(item)
    #     # self.list_widget.setItemWidget(item, row)
    #     # self._row_widgets[-1] = row

    #     # self.list_widget.set_fixed_last_item(item)

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

        row.config_constructor.expanded_changed.connect(
            lambda: QTimer.singleShot(0, lambda: self.update_row_height(item, row))
        )
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
        row.layout().activate()
        row.adjustSize()
        item.setSizeHint(row.sizeHint())

    def _on_data_changed(self, input):
        data_type, data_var = input
        # if data_type in ["sessions", "assignments"]:
        if data_type in ["session_added"]:  # ,"sessions","assignments"]:
            session_id = data_var
            self._add_session_row(session_id, self.data.sessions[session_id])
            # self.rebuild()
        elif data_type in ["session_removed", "session_moved"]:
            self.rebuild()
        else:
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

    def set_current_session(self, session_id: int):
        self.state.current_session_id = session_id

    def set_session_active(self, session_id: int, active: bool):
        session = self.data.sessions[session_id]
        session.active = active

        self.data.notify_change(("session", session_id))

    def rename_session(self, session_id: int, name: str):
        if not name:
            self.refresh_rows()
            return

        self.data.sessions[session_id].name = name

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

        self.data.notify_change(("sessions", session_id))

    def change_session_background(self, session_id: int):

        session = self.data.sessions[session_id]
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
            source_path=str(source_path.expanduser().resolve()),
            required=True,
        )

        self.data.queue_commit_session_realignment(
            session_id,
            background_template=candidate_template,
            remap=candidate_remap,
            background_spec=background_spec,
            expected_version=expected_version,
        )
