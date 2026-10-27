import numpy as np
from PySide6.QtCore import QObject, QTimer, Qt
from PySide6.QtWidgets import (
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QComboBox,
    QPushButton,
    QLabel,
    QToolButton,
    QMenu,
    QMessageBox,
    QCheckBox,
)

from catan.core.changes import (
    ChangeKind as C,
    DataChange,
    SESSION_STRUCTURE_CHANGES,
)
from catan.gui.GUI_elements.utils.menu_creation import fit_message_box_buttons
from catan.gui.GUI_elements.fragments.manual_alignment_dialog import (
    ManualAlignmentEditor,
    select_alignment_draft,
    select_overview_background_draft,
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

        # What to align: the current session, or the overview background
        # loaded from another source (Overview options).
        self.target = QComboBox(display_section)
        self.target.addItem("Align: current session", "session")
        self.target.addItem("Align: overview background", "overview")
        self.target.setToolTip(
            "The overview background is a display-only image loaded in the "
            "Overview options (Background: from file)."
        )
        # In the editor's options while an editor is shown (_place_target).
        self._x_options_layout = display_section.x_options_layout
        self._x_options_layout.insertWidget(0, self.target)
        self.target.currentIndexChanged.connect(lambda _: self._schedule_current())
        self._update_target_options()

        # Outdated (stale) alignments after an earlier session changed;
        # the arrow menu realigns all sessions.
        self.update_stale = QToolButton(display_section)
        self.update_stale.setPopupMode(QToolButton.ToolButtonPopupMode.MenuButtonPopup)
        self.update_stale.setToolButtonStyle(Qt.ToolButtonStyle.ToolButtonTextOnly)
        self.update_stale.clicked.connect(self._update_stale_alignments)
        menu = QMenu(self.update_stale)
        self.realign_all_action = menu.addAction(
            "Realign all sessions…", self._realign_all
        )
        self.update_stale.setMenu(menu)
        display_section.x_options_layout.insertWidget(1, self.update_stale)

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
        self.state.data_changed.connect(self._refresh_stale_button)
        self.state.current_session_changed.connect(self._schedule_current)
        self.state.alignment_draft_changed.connect(self._sync_draft)
        self.state.tasks.queue_changed.connect(self._schedule_current)
        self.state.tasks.queue_changed.connect(self._refresh_stale_button)
        self._refresh_stale_button()

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
                C.SESSION_ACTIVITY,
                C.FOOTPRINT_GEOMETRY,
                C.BACKGROUND_IMAGE,
                C.OVERVIEW_BACKGROUND,
            )
            or event.has_availability("spatial")
        ):
            return

        self._update_target_options()
        self._schedule_current()

    def _stale_sessions(self):
        return [
            session_id
            for session_id in range(len(self.data.sessions))
            # Inactive sessions are skipped by alignment updates.
            if self.data.alignment_is_stale(session_id)
            and self.data.session_is_active(session_id)
        ]

    def _refresh_stale_button(self, *_):
        if self._disposed:
            return

        stale = self._stale_sessions()
        tasks = self.state.tasks
        idle = not (tasks.processing_busy() or tasks.processing_requested)

        self.update_stale.setVisible(len(self.data.sessions) > 1)
        self.update_stale.setEnabled(idle)
        if self.editor is not None:
            self.editor.set_refine_all_available(
                idle and not stale and len(self.data.sessions) > 1,
                self.editor.refine_all_action.toolTip().split("\n\n")[0]
                + ("\n\nUpdate the outdated alignments first." if stale else ""),
            )
        self.realign_all_action.setEnabled(idle)

        if stale:
            self.update_stale.setText(f"Update outdated alignments ({len(stale)})")
            self.update_stale.setToolTip(
                "Realign in session order (manual alignments keep their "
                "transform; stops at a session needing review):\n"
                + "\n".join(
                    f"S{session_id} {self.data.sessions[session_id].name}"
                    for session_id in stale
                )
                + "\n\nArrow: realign all sessions."
            )
        else:
            self.update_stale.setText("Alignments up to date")
            self.update_stale.setToolTip(
                "No outdated alignments. Arrow: realign all sessions."
            )

    def _refine_from_footprints(self):
        tasks = self.state.tasks
        if tasks.processing_busy() or tasks.processing_requested:
            return

        expected_version = self.state.data_version
        if self.editor is not None:
            self.editor.set_refine_all_available(False)

        def estimate():
            try:
                return self.data.estimate_footprint_corrections(), None
            except ValueError as exc:
                return None, str(exc)

        tasks.start(
            "calculating",
            "Estimate footprint refinement",
            estimate,
            on_result=lambda result: self._show_footprint_refinement(
                result, expected_version
            ),
        )

    def _show_footprint_refinement(self, result, expected_version):
        if self._disposed:
            return
        self._refresh_stale_button()

        results, error = result
        if error is not None:
            self.state.issue("info", "Footprint refinement", error, parent=self)
            return
        if self.state.data_version != expected_version:
            self.state.issue(
                "info",
                "Footprint refinement",
                "The data changed during the estimate. Run it again.",
                parent=self,
            )
            return

        lines = []
        accepted = 0
        for sid, entry in sorted(results.items()):
            if entry is None:
                continue
            dx, dy, angle = entry["parameters"]
            name = self.data.sessions[sid].name
            if entry["accepted"]:
                accepted += 1
                lines.append(
                    f"S{sid} {name}: dx {dx:+.2f}, dy {dy:+.2f} px, "
                    f"rotation {angle:+.2f}° ({entry.get('matches')} cells)"
                )
            else:
                lines.append(f"S{sid} {name}: not refined — {entry.get('reason')}")

        largest = max(
            (np.hypot(*entry["parameters"][:2]) for entry in results.values() if entry and entry["accepted"]),
            default=0.0,
        )

        box = QMessageBox(self)
        box.setIcon(QMessageBox.Icon.Question)
        box.setWindowTitle("Refine alignments from footprints")
        box.setText(
            f"Corrections for {accepted} of {len(results)} sessions "
            f"(largest shift {largest:.2f} px)."
        )
        box.setInformativeText(
            "Each session is corrected against the footprints of all others; "
            "the mean correction is zero. Applying rebuilds the corrected "
            "sessions from their original footprints; their model evidence "
            "and assignments become outdated. Later alignments stay current."
        )
        box.setDetailedText("\n".join(lines))
        apply = box.addButton(
            f"Apply to {accepted} sessions", QMessageBox.ButtonRole.AcceptRole
        )
        apply.setEnabled(accepted > 0)
        box.addButton(QMessageBox.StandardButton.Cancel)
        fit_message_box_buttons(box)
        box.exec()

        if box.clickedButton() is not apply or self._disposed:
            return

        try:
            self.data.queue_apply_footprint_corrections(
                results,
                finished=lambda count: self.state.issue(
                    "info",
                    "Footprint refinement",
                    f"Refined the alignment of {count} sessions.",
                ),
            )
        except ValueError as exc:
            self.state.issue("warning", "Footprint refinement", str(exc), parent=self)

    def _update_stale_alignments(self):
        stale = self._stale_sessions()
        if stale:
            self.data.queue_process_alignments(session_ids=stale)

    def _realign_all(self):
        sessions = self.data.sessions
        manual = [
            session_id
            for session_id, session in enumerate(sessions)
            if session.remap is not None and session.remap.method == "manual"
        ]

        box = QMessageBox(self)
        box.setIcon(QMessageBox.Icon.Question)
        box.setWindowTitle("Realign all sessions")
        box.setText(
            f"Realign sessions 1–{len(sessions) - 1} automatically, in session "
            "order? Session 0 is the reference."
        )
        box.setInformativeText(
            "Model evidence and assignments of all sessions become outdated. "
            "Deviations from the default alignment are shown for confirmation."
        )
        replace_manual = None
        if manual:
            replace_manual = QCheckBox(
                f"Also recompute manual alignments ({len(manual)}: "
                + ", ".join(f"S{session_id}" for session_id in manual[:8])
                + (" …" if len(manual) > 8 else "")
                + ")"
            )
            box.setCheckBox(replace_manual)
        start = box.addButton("Realign all", QMessageBox.ButtonRole.AcceptRole)
        box.addButton(QMessageBox.StandardButton.Cancel)
        fit_message_box_buttons(box)
        box.exec()

        if box.clickedButton() is not start or self._disposed:
            return

        self.data.queue_realign_all(
            replace_manual=replace_manual is not None and replace_manual.isChecked()
        )

    def _update_target_options(self):
        available = getattr(self.data, "overview_background", None) is not None
        self.target.model().item(1).setEnabled(available)

        if not available and self.target.currentData() == "overview":
            self.target.setCurrentIndex(0)

    @property
    def _overview_target(self):
        return self.target.currentData() == "overview"

    def _schedule_current(self, *_):
        if not self._disposed:
            self._follow_timer.start(0)

    def _follow_current(self):
        if self._disposed:
            return

        if self._overview_target:
            self._follow_overview_background()
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
            self._hide_editor()
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
                self._hide_editor()
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

    def _follow_overview_background(self):
        old = self.state.alignment_draft

        if (
            old is not None
            and getattr(old, "external", False)
            and old.is_current(self.data)
        ):
            self._sync_draft()
            return

        draft = select_overview_background_draft(self.data, self)

        if draft is None:
            self.placeholder.setText(
                "Load an overview background first "
                "(Overview options: Background from file)."
            )
            self.placeholder.show()
            self.toolbar.hide()
            self._hide_editor()
            return

        self._sync_draft()

    def _open_window(self):
        open_manual_alignment(
            self.data,
            self.state.current_session_id,
            self,
            overview_background=self._overview_target,
        )

    def _sync_draft(self):
        if self._disposed:
            return

        draft = self.state.alignment_draft
        external = draft is not None and getattr(draft, "external", False)

        # The draft must match the selected target (and, for sessions,
        # the current session).
        if (
            draft is None
            or external != self._overview_target
            or (not external and draft.session_id != self.state.current_session_id)
        ):
            self._schedule_current()
            return

        if self.editor is None:
            self.editor = ManualAlignmentEditor(self.data, draft, self)
            self.toolbar_layout.addWidget(self.editor.view_controls)
            self.editor.apply_requested.connect(self._apply)
            self.editor.popout_requested.connect(self._open_window)
            self.editor.refine_all_requested.connect(self._refine_from_footprints)
            self.editor_layout.addWidget(self.editor)
            self._refresh_stale_button()
        else:
            self.editor.set_draft(draft)

        self.placeholder.hide()
        self.editor.show()
        self.toolbar.show()
        self._place_target()

    def _hide_editor(self):
        if self.editor is not None:
            self.editor.hide()
        self._place_target()

    def _place_target(self):
        """Target selection in the editor's options while it is shown, else
        in the panel's options row (to switch to the overview background)."""
        if self.editor is not None and not self.editor.isHidden():
            self.editor.set_target_widget(self.target)
        else:
            if self.editor is not None:
                self.editor.release_target_widget(self.target)
            if self._x_options_layout.indexOf(self.target) < 0:
                self._x_options_layout.insertWidget(0, self.target)
        self.target.show()

    def _remove_editor(self):
        if self.editor is not None:
            self.editor.release_target_widget(self.target)
            self.target.setParent(self.toolbar.parentWidget())
            self.editor.dispose()
            self.editor_layout.removeWidget(self.editor)
            self.editor.deleteLater()
            self.editor = None

    def _apply(self):
        if self.editor is None or not self.editor.can_apply():
            return

        draft = self.editor.draft

        if getattr(draft, "external", False):
            self.data.set_overview_background_alignment(
                draft.template, draft.make_remap()
            )
            # A fresh draft starts from the applied alignment.
            self.state.alignment_draft = None
            self.state.alignment_draft_changed.emit()
            return

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
        self.state.data_changed.disconnect(self._refresh_stale_button)
        self.state.tasks.queue_changed.disconnect(self._refresh_stale_button)
        self.state.alignment_draft_changed.disconnect(self._sync_draft)
        self._remove_editor()
        self.toolbar.hide()
        self.toolbar.deleteLater()
        self.target.hide()
        self.target.deleteLater()
        self.update_stale.hide()
        self.update_stale.deleteLater()


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
