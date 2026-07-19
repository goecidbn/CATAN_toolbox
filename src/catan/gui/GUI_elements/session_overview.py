from PySide6.QtCore import Qt, Signal, QPoint
from PySide6.QtGui import QColor, QAction
from PySide6.QtWidgets import (
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
)

from catan.gui.structures.data import Data
from catan.gui.structures.state import AppState
from catan.gui.data.utils import move_index_along_axis

from catan.gui.GUI_elements.fragments.IconButton import (
    make_icon_button,
    set_button_icon,
)

# print("Reloading session_overview.py")


class SessionRowWidget(QFrame):
    moveRequested = Signal(int, int)  # session_id, delta
    activeChanged = Signal(int, bool)  # session_id, active
    nameChanged = Signal(int, str)  # session_id, new_name

    setCurrentRequested = Signal(int)
    editOffsetRequested = Signal(int)
    changeColorRequested = Signal(int)

    traceToggled = Signal(int)
    matchToggled = Signal(int)
    dataToggled = Signal(int)

    removeRequested = Signal(int)

    def __init__(self, session_id: int, session, parent=None):
        super().__init__(parent)

        self.session_id = session_id
        self.session = session

        self.setObjectName("SessionRowWidget")
        self.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
        self.customContextMenuRequested.connect(self._open_context_menu)

        self.up_button = QToolButton()
        self.up_button.setText("▲")
        self.up_button.setFixedSize(20, 16)
        self.up_button.clicked.connect(
            lambda: self.moveRequested.emit(self.session_id, -1)
        )

        self.down_button = QToolButton()
        self.down_button.setText("▼")
        self.down_button.setFixedSize(20, 16)
        self.down_button.clicked.connect(
            lambda: self.moveRequested.emit(self.session_id, +1)
        )

        order_layout = QVBoxLayout()
        order_layout.setContentsMargins(0, 0, 0, 0)
        order_layout.setSpacing(0)
        order_layout.addWidget(self.up_button)
        order_layout.addWidget(self.down_button)

        self.active_checkbox = QCheckBox()
        self.active_checkbox.stateChanged.connect(self._on_active_changed)

        self.name_edit = QLineEdit()
        self.name_edit.setObjectName("SessionNameEdit")
        self.name_edit.editingFinished.connect(self._on_name_finished)

        self.offset_label = QLabel()
        self.offset_label.setObjectName("SessionOffsetLabel")
        self.offset_label.setAlignment(Qt.AlignmentFlag.AlignCenter)

        self.data_button = make_icon_button(
            ("fa6s.upload", "fa5s.upload"),
            tooltip="Load session data",
            fallback_theme_icon="document-open",
        )

        self.match_button = make_icon_button(
            ("fa6s.layer-group", "fa5s.layer-group"),
            tooltip="Add to matching",
            fallback_theme_icon="applications-games",
        )

        self.trace_button = make_icon_button(
            (
                # "fa6s.wave-square",
                # "fa5s.wave-square",
                "fa6s.chart-line",
                "fa5s.chart-line",
            ),
            tooltip="Load traces",
            fallback_theme_icon="media-playback-start",
        )

        # self.trace_button = QPushButton()
        self.trace_button.clicked.connect(
            lambda: self.traceToggled.emit(self.session_id)
        )

        # self.match_button = QPushButton()
        self.match_button.clicked.connect(
            lambda: self.matchToggled.emit(self.session_id)
        )

        # self.data_button = QPushButton()
        self.data_button.clicked.connect(lambda: self.dataToggled.emit(self.session_id))

        layout = QHBoxLayout(self)
        layout.setContentsMargins(6, 3, 6, 3)
        layout.setSpacing(5)

        layout.addLayout(order_layout)
        layout.addWidget(self.active_checkbox)
        layout.addWidget(self.name_edit)
        layout.addWidget(self.offset_label)
        layout.addStretch()
        layout.addWidget(self.trace_button)
        layout.addWidget(self.match_button)
        layout.addWidget(self.data_button)

        self.refresh()

    def refresh(self):
        name = getattr(self.session, "name", f"Session{self.session_id:02d}")
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
        self._update_background()

    def _update_buttons(self):
        has_traces = getattr(self.session, "traces_loaded", False)
        is_matched = getattr(self.session, "matched", False)
        is_loaded = getattr(self.session, "loaded", True)
        is_scheduled = getattr(self.session, "scheduled_for_loading", False)

        # self.trace_button.setText("Unload traces" if has_traces else "Load traces")
        # self.match_button.setText("Unmatch" if is_matched else "Match")

        # print("Updating buttons for session", self.session_id, ":", has_traces)
        if has_traces:
            set_button_icon(
                self.trace_button,
                ("fa6s.trash-can", "fa5s.trash-alt"),
                tooltip="Unload traces",
                fallback_theme_icon="edit-delete",
            )
        else:
            set_button_icon(
                self.trace_button,
                (
                    # "fa6s.wave-square", "fa5s.wave-square",
                    "fa6s.chart-line",
                    "fa5s.chart-line",
                ),
                tooltip="Load traces",
                fallback_theme_icon="media-playback-start",
            )

        if is_matched:
            set_button_icon(
                self.match_button,
                ("fa6s.note-sticky", "fa5s.note-sticky"),
                tooltip="Remove from matching",
                fallback_theme_icon="edit-delete",
            )
        else:
            set_button_icon(
                self.match_button,
                ("fa6s.layer-group", "fa5s.layer-group"),
                tooltip="Add to matching",
                fallback_theme_icon="applications-games",
            )

        if is_loaded:
            set_button_icon(
                self.data_button,
                ("fa6s.trash-can", "fa5s.trash-alt"),
                tooltip="Remove session data",
                fallback_theme_icon="edit-delete",
            )
        else:
            set_button_icon(
                self.data_button,
                ("fa6s.upload", "fa5s.upload"),
                tooltip="Load session data",
                fallback_theme_icon="document-open",
            )
        # if is_loaded:
        #     self.data_button.setText("Remove")
        # elif is_scheduled:
        #     self.data_button.setText("Load")
        # else:
        #     self.data_button.setText("Load")

    def _update_background(self):
        color = getattr(self.session, "color", QColor("#555555"))

        if not isinstance(color, QColor):
            color = QColor(str(color))

        # Soft translucent background, so text remains readable.
        r, g, b, _ = color.getRgb()
        self.setStyleSheet(f"""
            QFrame#SessionRowWidget {{
                background-color: rgba({r}, {g}, {b}, 55);
                border-radius: 4px;
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

    def _on_active_changed(self, state):
        self.activeChanged.emit(
            self.session_id,
            state == Qt.CheckState.Checked.value,
        )

    def _on_name_finished(self):
        self.nameChanged.emit(self.session_id, self.name_edit.text().strip())

    def _open_context_menu(self, pos: QPoint):
        menu = QMenu(self)

        menu.addAction(
            "Set current session",
            lambda: self.setCurrentRequested.emit(self.session_id),
        )
        menu.addSeparator()

        menu.addAction(
            "Edit time offset…", lambda: self.editOffsetRequested.emit(self.session_id)
        )
        menu.addAction(
            "Change color…", lambda: self.changeColorRequested.emit(self.session_id)
        )

        menu.addSeparator()
        menu.addAction(
            self.trace_button.text(), lambda: self.traceToggled.emit(self.session_id)
        )
        menu.addAction(
            self.match_button.text(), lambda: self.matchToggled.emit(self.session_id)
        )
        menu.addAction(
            self.data_button.text(), lambda: self.dataToggled.emit(self.session_id)
        )

        menu.addSeparator()
        remove_action = QAction("Remove session", menu)
        remove_action.triggered.connect(
            lambda: self.removeRequested.emit(self.session_id)
        )
        menu.addAction(remove_action)

        menu.exec(self.mapToGlobal(pos))


from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QWidget,
    QListWidget,
    QListWidgetItem,
    QVBoxLayout,
    QInputDialog,
    QColorDialog,
    QMessageBox,
)


class SessionListPanel(QWidget):
    def __init__(self, parent):
        super().__init__(parent)

        self.data: Data = parent.data
        self.state: AppState = parent.state

        self.list_widget = QListWidget()
        self.list_widget.setSpacing(3)
        self.list_widget.itemDoubleClicked.connect(self._on_item_double_clicked)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self.list_widget)

        self._row_widgets: dict[int, SessionRowWidget] = {}

        self.rebuild()

    def rebuild(self):
        self.list_widget.clear()
        self._row_widgets.clear()

        for session_id, session in enumerate(self.data.sessions):
            self._add_session_row(session_id, session)

    def _add_session_row(self, session_id: int, session):
        item = QListWidgetItem()
        item.setData(Qt.ItemDataRole.UserRole, session_id)

        row = SessionRowWidget(session_id, session, parent=self.list_widget)

        row.moveRequested.connect(self.move_session)
        row.activeChanged.connect(self.set_session_active)
        row.nameChanged.connect(self.rename_session)

        row.setCurrentRequested.connect(self.set_current_session)
        row.editOffsetRequested.connect(self.edit_time_offset)
        # row.changeColorRequested.connect(self.change_session_color)

        row.traceToggled.connect(self.toggle_traces)
        row.matchToggled.connect(self.toggle_match)
        row.dataToggled.connect(self.toggle_session_data)
        row.removeRequested.connect(self.remove_session)

        item.setSizeHint(row.sizeHint())

        self.list_widget.addItem(item)
        self.list_widget.setItemWidget(item, row)

        self._row_widgets[session_id] = row

    def refresh_rows(self):
        """
        Use this when session properties changed but the order did not.
        """
        for session_id, row in self._row_widgets.items():
            row.session_id = session_id
            row.session = self.data.sessions[session_id]
            row.refresh()

    def rebuild_after_structure_change(self):
        """
        Use this after adding/removing/reordering sessions.
        """
        self.rebuild()

    def _on_item_double_clicked(self, item: QListWidgetItem):
        session_id = item.data(Qt.ItemDataRole.UserRole)
        self.set_current_session(session_id)

    def set_current_session(self, session_id: int):
        self.state.current_session_id = session_id

    def set_session_active(self, session_id: int, active: bool):
        session = self.data.sessions[session_id]
        session.active = active

        # Recommended: state signal that all displays/statistics should respect.
        if hasattr(self.state, "data_changed"):
            self.state.data_changed.emit(("session", session_id))

    def rename_session(self, session_id: int, name: str):
        if not name:
            self.refresh_rows()
            return

        self.data.sessions[session_id].name = name

        # if hasattr(self.state, "data_changed"):
        #     self.state.data_changed.emit()

    def edit_time_offset(self, session_id: int):
        session = self.data.sessions[session_id]
        old_value = int(getattr(session, "time_offset", 0))

        value, ok = QInputDialog.getInt(
            self,
            "Trace time offset",
            f"Offset for {getattr(session, 'name', session_id)}:",
            old_value,
            -10_000_000,
            10_000_000,
            1,
        )

        if not ok:
            return

        session.time_offset = value
        self.refresh_rows()

        if hasattr(self.state, "data_changed"):
            self.state.data_changed.emit(("session", session_id))

    # def change_session_color(self, session_id: int):
    #     session = self.data.sessions[session_id]
    #     old_color = getattr(session, "color", None)

    #     if old_color is None:
    #         old_color = QColor("#888888")
    #     elif not isinstance(old_color, QColor):
    #         old_color = QColor(str(old_color))

    #     color = QColorDialog.getColor(
    #         old_color,
    #         self,
    #         "Choose session color",
    #     )

    #     if not color.isValid():
    #         return

    #     session.color = color
    #     self.refresh_rows()

    #     if hasattr(self.state, "data_changed"):
    #         self.state.data_changed.emit()

    def toggle_traces(self, session_id: int):
        # print("Toggling traces for session", session_id)

        self.data.change_trace_presence(session_id)
        self.refresh_rows()

    def toggle_match(self, session_id: int):

        if getattr(self.data.sessions[session_id], "matched", False):
            self.data.unregister_neurons(session_id)
        else:
            self.data.register_neurons(from_session_id=session_id)

        self.refresh_rows()

        # if hasattr(self.state, "data_changed"):
        self.state.assignments = self.data.assignments
        print(
            " not optimal - both should be updated automatically, or only one should exist!"
        )
        self.state.data_changed.emit(("assignment", -1))

    def toggle_session_data(self, session_id: int):
        session = self.data.sessions[session_id]

        if getattr(session, "loaded", True):
            self.remove_session(session_id)
            return

        # Future implementation.
        session.load_data()

        self.refresh_rows()

        if hasattr(self.state, "data_changed"):
            self.state.data_changed.emit(("session", session_id))

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

        self.reindex_sessions_after_order_change()
        self.rebuild_after_structure_change()

        if hasattr(self.state, "data_changed"):
            self.state.data_changed.emit(("session", session_id))

    def move_session(self, session_id: int, delta: int):
        new_id = session_id + delta

        if new_id < 0 or new_id >= len(self.data.sessions):
            return

        self.data.move_session(session_id, new_id)
        self.state.assignments = move_index_along_axis(
            self.state.assignments, axis=1, old=session_id, new=new_id
        )

        self.reindex_sessions_after_order_change()
        self.rebuild_after_structure_change()

        if hasattr(self.state, "data_changed"):
            self.state.data_changed.emit(("session", -1))

    def reindex_sessions_after_order_change(self):
        for session_id, session in enumerate(self.data.sessions):

            if session is None:
                continue

            if self.state.current_session_id == session.id:
                self.state.current_session_id = session_id
            session.id = session_id
