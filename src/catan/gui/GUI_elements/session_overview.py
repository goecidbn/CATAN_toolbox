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

from pathlib import Path

from catan.gui.structures import AppState, Data, SessionData
from catan.core.structures import sessiondata_type

from .fragments import (
    FieldConfigConstructor,
    GlobReviewDialog,
    make_icon_button,
    set_button_icon,
    choose_path,
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
        if data_type == "sessions" and data_var == self.session.id:
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
        self._update_background(current=current)
        self.config_constructor._on_fields_changed()

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
        remove_action = QAction("Remove session", menu)
        remove_action.triggered.connect(lambda: self.removeRequested.emit(self.index))
        menu.addAction(remove_action)

        menu.exec(self.mapToGlobal(pos))


class LoadSessionRowWidget(QFrame):
    loadRequested = Signal()

    def __init__(self, parent: "SessionOverview"):
        super().__init__(parent)

        self.state = parent.state
        self.data = parent.data

        self.setObjectName("SessionRowWidget")

        # order_layout = QVBoxLayout(self)
        # order_layout.setContentsMargins(5, 5, 5, 5)
        # order_layout.setSpacing(5)

        layout = QHBoxLayout(self)
        layout.setContentsMargins(6, 3, 6, 3)
        layout.setSpacing(3)
        # order_layout.addLayout(layout)

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
                lambda ctx: self.data.register_session(from_file=path, ctx=ctx),
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

        for path in paths:
            self.data.register_session(from_file=path)
        return


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
        layout.addWidget(LoadSessionRowWidget(self))

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

        self.state.data_changed.emit(("sessions", session_id))

    def toggle_session_data(
        self, session_id: int, which: Optional[sessiondata_type] = None
    ):
        session = self.data.sessions[session_id]
        self.state.tasks.start(
            "loading",
            f"Loading {which} data for {session.name}",
            lambda ctx: self.data.toggle_session_data(session_id, which, ctx=ctx),
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
