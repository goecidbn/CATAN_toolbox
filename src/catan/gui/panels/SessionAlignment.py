from PySide6.QtCore import QObject, QTimer
from PySide6.QtWidgets import (
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QComboBox,
    QPushButton,
    QLabel,
)

from catan.core.changes import (
    ChangeKind as C,
    DataChange,
    SESSION_STRUCTURE_CHANGES,
)
from catan.gui.GUI_elements.fragments.manual_alignment_dialog import (
    ManualAlignmentEditor,
    select_alignment_draft,
    open_manual_alignment,
)


class Display(QWidget):
    def __init__(self, display_section):
        super().__init__(display_section)
        self.data = display_section.data
        self.state = display_section.state

        self.editor = None
        self._disposed = False

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        self.toolbar = QWidget(display_section)
        self.toolbar_layout = QVBoxLayout(self.toolbar)
        self.toolbar_layout.setContentsMargins(0, 0, 0, 0)
        self.toolbar_layout.setSpacing(4)

        display_section.x_options_layout.insertWidget(0, self.toolbar, 1)

        self.placeholder = QLabel(
            "Select a session with loaded spatial data and a background.", self
        )
        layout.addWidget(self.placeholder)
        self.editor_layout = QVBoxLayout()
        self.editor_layout.setContentsMargins(0, 0, 0, 0)
        self.editor_layout.setSpacing(0)
        layout.addLayout(self.editor_layout, 1)

        self._follow_timer = QTimer(self)
        self._follow_timer.setSingleShot(True)
        self._follow_timer.timeout.connect(self._follow_current)

        self.state.data_changed.connect(self._refresh_sessions)
        self.state.current_session_changed.connect(self._schedule_current)
        self.state.alignment_draft_changed.connect(self._sync_draft)
        self.state.tasks.queue_changed.connect(self._schedule_current)

        self._follow_current()
        self.state.alignment_panels.add(self)
        self.state.alignment_review_changed.emit()

        self._refresh_sessions()
        self._sync_draft()

    def _refresh_sessions(self, event: DataChange | None = None):
        if event is not None and not (
            event.has(
                *SESSION_STRUCTURE_CHANGES,
                C.SESSION_METADATA,
                C.FOOTPRINT_GEOMETRY,
                C.BACKGROUND_IMAGE,
            )
            or event.has_availability("spatial")
        ):
            return

        self._schedule_current()

    def _schedule_current(self, *_):
        if not self._disposed:
            self._follow_timer.start(0)

    def _follow_current(self):
        if self._disposed:
            return

        sid = self.state.current_session_id
        valid = (
            sid is not None
            and 0 <= sid < len(self.data.sessions)
            and self.data.sessions[sid].status["spatial_loaded"]
            and self.data.sessions[sid].background_template is not None
        )

        if not valid:
            self.placeholder.setText(
                "Select a session with loaded spatial data and a background."
            )
            self.placeholder.show()
            self.toolbar.hide()
            if self.editor is not None:
                self.editor.hide()
            return

        old = self.state.alignment_draft
        same_session = (
            old is not None
            and old.session is self.data.sessions[sid]
            and old.session_id == sid
        )

        # Keep edited drafts visible if unrelated data invalidated their
        # version. Reset explicitly discards them and takes a fresh snapshot.
        if same_session and (old.is_current(self.data) or old.dirty):
            self._sync_draft()
            return

        tasks = self.state.tasks
        if tasks.processing_busy() or tasks.processing_requested:
            if self.editor is None or self.editor.draft.session_id != sid:
                self.placeholder.setText("Waiting for processing to finish…")
                self.placeholder.show()
                self.toolbar.hide()
                if self.editor is not None:
                    self.editor.hide()
            return

        draft = select_alignment_draft(self.data, sid, self)
        if draft is None:
            # A rejected discard also restores CATAN's current selection.
            if old is not None:
                previous_id = next(
                    (
                        index
                        for index, session in enumerate(self.data.sessions)
                        if session is old.session
                    ),
                    None,
                )
                if previous_id is not None:
                    self.state.current_session_id = previous_id
            return

        self._sync_draft()

    def _open_window(self):
        open_manual_alignment(self.data, self.state.current_session_id, self)

    def _sync_draft(self):
        if self._disposed:
            return

        draft = self.state.alignment_draft
        if draft is None or draft.session_id != self.state.current_session_id:
            self._schedule_current()
            return

        if self.editor is None:
            self.editor = ManualAlignmentEditor(self.data, draft, self)
            self.toolbar_layout.addWidget(self.editor.view_controls)
            self.editor.apply_requested.connect(self._apply)
            self.editor.popout_requested.connect(self._open_window)
            self.editor_layout.addWidget(self.editor)
        else:
            self.editor.set_draft(draft)

        self.placeholder.hide()
        self.editor.show()
        self.toolbar.show()

    def _remove_editor(self):
        if self.editor is not None:
            self.editor.dispose()
            self.editor_layout.removeWidget(self.editor)
            self.editor.deleteLater()
            self.editor = None

    def _apply(self):
        if self.editor is None or not self.editor.can_apply():
            return

        draft = self.editor.draft
        self.data.queue_commit_session_realignment(
            draft.session_id,
            background_template=draft.template,
            remap=draft.make_remap(),
            background_spec=None,
            expected_version=draft.expected_version,
        )

    def dispose(self):
        if self._disposed:
            return

        self._disposed = True
        self.state.alignment_panels.discard(self)
        self.state.alignment_review_changed.emit()

        self._follow_timer.stop()
        self.state.current_session_changed.disconnect(self._schedule_current)
        self.state.tasks.queue_changed.disconnect(self._schedule_current)

        self.state.data_changed.disconnect(self._refresh_sessions)
        self.state.alignment_draft_changed.disconnect(self._sync_draft)
        self._remove_editor()
        self.toolbar.hide()
        self.toolbar.deleteLater()


class Controller(QObject):
    def __init__(self, display_section, config=None):
        super().__init__(display_section)
        self.section = display_section
        self.display = None

    def activate(self):
        self.display = Display(self.section)
        self.section.set_display_widget(self.display)

    def deactivate(self):
        if self.display is not None:
            self.display.dispose()
            self.section.clear_display_widget()
            self.display = None

        self.deleteLater()
