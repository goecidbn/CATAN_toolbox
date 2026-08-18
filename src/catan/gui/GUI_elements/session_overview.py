from PySide6.QtCore import Qt, Signal, QPoint
from PySide6.QtGui import QColor, QAction
from PySide6.QtWidgets import (
    QAbstractItemView,
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
)

from catan.gui.structures import AppState, Data

from .fragments.IconButton import (
    make_icon_button,
    set_button_icon,
)


class SessionRowWidget(QFrame):
    moveRequested = Signal(int, int)  # session_id, delta
    activeChanged = Signal(int, bool)  # session_id, active
    nameChanged = Signal(int, str)  # session_id, new_name

    setCurrentRequested = Signal(int)
    editOffsetRequested = Signal(int)
    changeColorRequested = Signal(int)

    loadRequested = Signal(int)  # session_id
    traceToggled = Signal(int)
    qualityToggled = Signal(int)
    spatialToggled = Signal(int)

    removeRequested = Signal(int)

    def __init__(self, session_id: int, session, current=False, parent=None):
        super().__init__(parent)

        self.index = session_id
        self.session = session

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
        self.name_edit.editingFinished.connect(self._on_name_finished)

        self.offset_label = QLabel()
        self.offset_label.setObjectName("SessionOffsetLabel")
        self.offset_label.setAlignment(Qt.AlignmentFlag.AlignCenter)

        self.load_fields_button = QToolButton(self)
        self.load_fields_button.setText("Load")
        self.load_fields_button.setPopupMode(
            QToolButton.ToolButtonPopupMode.MenuButtonPopup
        )

        self.load_fields_button.clicked.connect(
            lambda: self.loadRequested.emit(self.index)
        )

        menu = QMenu(self.load_fields_button)

        container = QWidget(menu)
        layout_menu = QHBoxLayout(container)
        layout_menu.setContentsMargins(6, 6, 6, 6)
        layout_menu.setSpacing(4)

        self.trace_button = make_icon_button()
        self.quality_button = make_icon_button()
        self.spatial_button = make_icon_button()

        layout_menu.addWidget(self.trace_button)
        layout_menu.addWidget(self.quality_button)
        layout_menu.addWidget(self.spatial_button)

        widget_action = QWidgetAction(menu)
        widget_action.setDefaultWidget(container)

        menu.addAction(widget_action)

        self.load_fields_button.setMenu(menu)

        self.delete_button = make_icon_button(
            ("fa6s.ban", "fa5s.ban"),
            color="red",
            tooltip="Remove session data",
            fallback_theme_icon="edit-delete",
        )

        # self.trace_button = QPushButton()
        self.trace_button.clicked.connect(lambda: self.traceToggled.emit(self.index))
        self.quality_button.clicked.connect(
            lambda: self.qualityToggled.emit(self.index)
        )

        # self.spatial_button = QPushButton()
        self.spatial_button.clicked.connect(
            lambda: self.spatialToggled.emit(self.index)
        )

        # self.data_button = QPushButton()
        self.delete_button.clicked.connect(
            lambda: self.removeRequested.emit(self.index)
        )

        layout = QHBoxLayout(self)
        layout.setContentsMargins(6, 3, 6, 3)
        layout.setSpacing(5)

        layout.addLayout(order_layout)
        layout.addWidget(self.active_checkbox)
        layout.addWidget(self.name_edit)
        layout.addWidget(self.offset_label)
        layout.addStretch()
        layout.addWidget(self.load_fields_button)
        # layout.addWidget(self.trace_button)
        # layout.addWidget(self.quality_button)
        # layout.addWidget(self.spatial_button)
        layout.addWidget(self.delete_button)

        self.refresh(current=current)

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
        self._update_background(current=current)

    def _update_buttons(self):

        spatial_loaded = self.session.status["spatial_loaded"]
        if not spatial_loaded:
            set_button_icon(
                self.spatial_button,
                ("fa6s.layer", "fa5s.layer"),
                color="white",
                tooltip="Load footprint data",
                fallback_theme_icon="applications-games",
            )
        else:
            matched = self.session.status["matched"]
            set_button_icon(
                self.spatial_button,
                ("fa6s.layer-group", "fa5s.layer-group"),
                color="white" if not matched else "red",
                tooltip=("Add to matching" if not matched else "Remove from matching"),
                fallback_theme_icon=(
                    "applications-games" if not matched else "edit-delete"
                ),
            )

        trace_loaded = self.session.status["traces_loaded"]
        set_button_icon(
            self.trace_button,
            ("fa6s.chart-line", "fa5s.chart-line"),
            color="white" if not trace_loaded else "red",
            tooltip=("Load traces" if not trace_loaded else "Unload traces"),
            fallback_theme_icon=(
                "media-playback-start" if not trace_loaded else "edit-delete"
            ),
        )

        quality_loaded = self.session.status["quality_loaded"]
        set_button_icon(
            self.quality_button,
            ("fa6s.chart-column", "fa5s.chart-column"),
            color="white" if not quality_loaded else "red",
            tooltip=(
                "Load quality parameters"
                if not quality_loaded
                else "Remove quality parameters"
            ),
            fallback_theme_icon=(
                "document-open" if not quality_loaded else "edit-delete"
            ),
        )

    def _update_background(self, current=False):
        color = getattr(self.session, "color", QColor("#888888"))
        if current:
            color = getattr(self.session, "color", QColor("#333333"))

        if not isinstance(color, QColor):
            color = QColor(str(color))

        # Soft translucent background, so text remains readable.
        r, g, b, _ = color.getRgb()
        self.setStyleSheet(f"""
            QFrame#SessionRowWidget {{21
                background-color: rgba({r}, {g}, {b}, 55);
                border: {"2px solid rgba(255, 255, 255, 55)" if current else "1px solid rgba(255, 255, 255, 35)"};
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

        menu.addSeparator()
        menu.addAction(
            self.trace_button.text(), lambda: self.traceToggled.emit(self.index)
        )
        menu.addAction(
            self.quality_button.text(), lambda: self.qualityToggled.emit(self.index)
        )
        menu.addAction(
            self.spatial_button.text(), lambda: self.spatialToggled.emit(self.index)
        )

        menu.addSeparator()
        remove_action = QAction("Remove session", menu)
        remove_action.triggered.connect(lambda: self.removeRequested.emit(self.index))
        menu.addAction(remove_action)

        menu.exec(self.mapToGlobal(pos))


class SessionList(QListWidget):
    drag_n_dropped = Signal(int, int)  # old_index, new_index

    def __init__(self, parent: "SessionOverview"):
        super().__init__(parent)
        # self.parent = parent

        self.setDragDropMode(QAbstractItemView.DragDropMode.InternalMove)
        self.setDefaultDropAction(Qt.DropAction.MoveAction)

        self.setSelectionMode(QAbstractItemView.SelectionMode.SingleSelection)

    def dropEvent(self, event):
        item = self.currentItem()
        old_index = self.row(item)
        super().dropEvent(event)
        new_index = self.row(item)

        if old_index == new_index:
            return

        self.drag_n_dropped.emit(old_index, new_index)


class SessionOverview(QWidget):
    load_requested = Signal(int)  # session_id

    def __init__(self, parent):
        super().__init__(parent)

        self.data: Data = parent.data
        self.state: AppState = parent.state

        self.list_widget = SessionList(parent=self)
        self.list_widget.setSpacing(3)
        self.list_widget.itemDoubleClicked.connect(self._on_item_double_clicked)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self.list_widget)

        self.list_widget.setDragDropMode(QAbstractItemView.DragDropMode.InternalMove)
        self.list_widget.setDefaultDropAction(Qt.DropAction.MoveAction)

        self._row_widgets: dict[int, SessionRowWidget] = {}

        self.state.data_changed.connect(self._on_data_changed)
        self.state.current_session_changed.connect(self._on_current_session_changed)
        self.list_widget.drag_n_dropped.connect(self.move_session)
        self.rebuild()

    def rebuild(self):
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
        row.traceToggled.connect(self.toggle_traces)
        row.qualityToggled.connect(self.toggle_quality)
        row.spatialToggled.connect(self.toggle_spatial)
        row.removeRequested.connect(self.remove_session)

        item.setSizeHint(row.sizeHint())

        self.list_widget.addItem(item)
        self.list_widget.setItemWidget(item, row)

        self._row_widgets[session_id] = row

    def _on_data_changed(self, input):
        data_type, data_var = input
        if data_type in ["sessions", "assignments"]:
            self.rebuild()
        else:
            self.refresh_rows()

    def _on_current_session_changed(self):
        self.refresh_rows()

    def refresh_rows(self):
        """
        Use this when session properties changed but the order did not.
        """
        for index, row in self._row_widgets.items():
            row.index = index
            # session_id = row.session.id
            # session_id = self.data.sessions[index].id
            row.session = self.data.sessions[index]

            row.refresh(current=index == self.state.current_session_id)

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

        self.state.data_changed.emit(("sessions", session_id))

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

    # def load_fields(self, session_id: int):
    #     self.data.load_data(session_id, ["spatial", "traces", "quality"])
    #     self.refresh_rows()

    def toggle_traces(self, session_id: int):
        session = self.data.sessions[session_id]
        print(f"Toggling trace data for {session.name} (ID {session_id})")
        self.state.tasks.start(
            "loading",
            f"Toggling trace data for {session.name}",
            lambda ctx: self.data.change_trace_presence(session_id, ctx=ctx),
            finished=self.refresh_rows,
        )

    def toggle_quality(self, session_id: int):
        session = self.data.sessions[session_id]
        self.state.tasks.start(
            "loading",
            f"Toggling quality data for {session.name}",
            lambda ctx: self.data.change_quality_presence(session_id, ctx=ctx),
            finished=self.refresh_rows,
        )

    def toggle_spatial(self, session_id: int):
        session = self.data.sessions[session_id]
        if not session.status["spatial_loaded"]:
            self.state.tasks.start(
                "loading",
                f"Loading data for {session.name}",
                lambda ctx: self.data.change_spatial_presence(
                    session_id, True, ctx=ctx
                ),
                finished=self.refresh_rows,
            )
            return

        if self.data.sessions[session_id].status["matched"]:
            self.data.unregister_neurons(session_id)
        else:
            self.data.register_neurons(from_session_index=session_id)

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
        # print(f"New session order: {[s.id for s in self.data.sessions]}")

        # currentItem = self.list_widget.takeItem(session_index)
        # self.list_widget.insertItem(new_session_index, currentItem)

        # self.list_widget.setCurrentRow(new_session_index)
