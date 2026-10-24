from PySide6.QtCore import Signal, Qt, QSize, QTimer
from PySide6.QtWidgets import (
    QWidget,
    QLabel,
    QPushButton,
    QToolButton,
    QHBoxLayout,
    QVBoxLayout,
    QListWidget,
    QListWidgetItem,
    QAbstractItemView,
    QSizePolicy,
    QMessageBox,
)
from .toggle_option import ToggleOption
from catan.gui.GUI_elements.utils.menu_creation import fit_message_box_buttons


class TaskItemWidget(QWidget):
    cancel_requested = Signal(str)

    def __init__(
        self,
        task_id: str,
        name: str,
        running: bool = False,
        cancelling: bool = False,
        parent=None,
    ):
        super().__init__(parent)

        self.task_id = task_id

        if cancelling:
            status = "◐"
            text = f"{status} {name} — Cancelling..."
        else:
            status = "●" if running else "○"
            text = f"{status} {name}"

        self.label = QLabel(text)

        self.cancel_button = QPushButton("×")
        self.cancel_button.setFixedWidth(28)
        self.cancel_button.setToolTip("Cancel task")

        if cancelling:
            self.cancel_button.setEnabled(False)

        self.cancel_button.clicked.connect(
            lambda: self.cancel_requested.emit(self.task_id)
        )

        layout = QHBoxLayout(self)
        layout.setContentsMargins(2, 2, 2, 2)

        layout.addWidget(self.label)
        layout.addStretch()
        layout.addWidget(self.cancel_button)


class ReorderableTaskList(QListWidget):
    task_moved = Signal(int, int)

    def __init__(self, parent=None):
        super().__init__(parent)

        self.setDragDropMode(QAbstractItemView.DragDropMode.InternalMove)

        self.setSelectionMode(QAbstractItemView.SelectionMode.SingleSelection)

    def dropEvent(self, event):
        item = self.currentItem()

        if item is None:
            super().dropEvent(event)
            return

        old_index = self.row(item)

        super().dropEvent(event)

        new_index = self.row(item)

        if old_index != new_index:
            self.task_moved.emit(
                old_index,
                new_index,
            )


class TaskQueueDisplay(QWidget):
    def __init__(
        self,
        task_manager,
        group: str,
        parent=None,
    ):
        super().__init__(parent)

        self.task_manager = task_manager
        self.group = group

        self.title = QLabel(group.capitalize())

        # Current task area
        self.current_container = QWidget()
        self.current_layout = QVBoxLayout(self.current_container)
        self.current_layout.setContentsMargins(0, 0, 0, 0)
        self.current_layout.setSpacing(0)

        # Queued tasks
        self.queue_list = ReorderableTaskList()
        self.queue_list.task_moved.connect(self._move_task)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        layout.addWidget(self.title)
        layout.addWidget(self.current_container)
        layout.addWidget(self.queue_list)

        # What is displayed, to update in place instead of rebuilding.
        self._current_key = None
        self._queued_ids = []

        # Task bursts (e.g. queueing a recipe) emit many signals; refresh
        # once per burst, and not at all while hidden (refreshed on show).
        self._refresh_pending = False
        self._refresh_timer = QTimer(self)
        self._refresh_timer.setSingleShot(True)
        self._refresh_timer.setInterval(50)
        self._refresh_timer.timeout.connect(self.refresh)

        # ---- TaskManager signals ----

        self.task_manager.queue_changed.connect(self._queue_changed)

        self.task_manager.task_started.connect(self._task_changed)

        self.task_manager.task_finished.connect(self._task_changed)

        self.task_manager.task_cancelled.connect(self._task_changed)

        self.task_manager.task_failed.connect(self._task_changed)

        self.refresh()

    def _queue_changed(
        self,
        group: str,
    ):
        if group == self.group:
            self._schedule_refresh()

    def _task_changed(
        self,
        group: str,
        task_id: str,
    ):
        if group == self.group:
            self._schedule_refresh()

    def _schedule_refresh(self):
        self._refresh_pending = True

        if self.isVisible() and not self._refresh_timer.isActive():
            self._refresh_timer.start()

    def showEvent(self, event):
        super().showEvent(event)

        if self._refresh_pending:
            self.refresh()

    def _move_task(
        self,
        old_index: int,
        new_index: int,
    ):
        self.task_manager.move_queued_task(
            self.group,
            old_index,
            new_index,
        )

    def _clear_layout(
        self,
        layout,
    ):
        while layout.count():
            item = layout.takeAt(0)

            widget = item.widget()

            if widget is not None:
                widget.deleteLater()

    def refresh(self):
        self._refresh_pending = False
        self._refresh_timer.stop()

        # ============================================================
        # Current task
        # ============================================================

        summary = self.task_manager.group_summary(self.group)

        if summary["running"]:
            if summary["current_cancelling"]:
                state = "cancelling"
            else:
                state = "1 running"
        else:
            state = "idle"

        label = (
            f"{self.group.capitalize()}: "
            f"{state}, "
            f"{summary['queued_count']} queued"
        )
        self.title.setText(label)

        current = self.task_manager.current_task(self.group)

        current_key = (
            None
            if current is None
            else (current.id, current.name, current.worker.is_cancelled())
        )

        if current_key != self._current_key:
            self._current_key = current_key
            self._clear_layout(self.current_layout)

            if current is not None:
                current_widget = TaskItemWidget(
                    task_id=current.id,
                    name=current.name,
                    running=True,
                    cancelling=current.worker.is_cancelled(),
                )

                current_widget.cancel_requested.connect(self.task_manager.cancel)

                self.current_layout.addWidget(current_widget)

        # ============================================================
        # Queued tasks
        # ============================================================

        queued = self.task_manager.queued_tasks(self.group)
        queued_ids = [task.id for task in queued]
        shown = self._queued_ids

        self.queue_list.setVisible(bool(queued))

        if queued_ids == shown:
            return

        n_shown = len(shown)
        if n_shown and queued_ids == shown[n_shown - len(queued_ids) :]:
            # Tasks left the front of the queue (started or cancelled).
            for _ in range(n_shown - len(queued_ids)):
                self.queue_list.takeItem(0)
        elif queued_ids[:n_shown] == shown:
            # Tasks were appended.
            for task in queued[n_shown:]:
                self._add_queued_item(task)
        else:
            # Reordered, or removed from the middle.
            self.queue_list.clear()
            for task in queued:
                self._add_queued_item(task)

        self._queued_ids = queued_ids

    def _add_queued_item(self, task):
        item = QListWidgetItem()

        item.setData(
            Qt.ItemDataRole.UserRole,
            task.id,
        )

        widget = TaskItemWidget(
            task_id=task.id,
            name=task.name,
            running=False,
            cancelling=False,
        )

        widget.cancel_requested.connect(self.task_manager.cancel)

        item.setSizeHint(widget.sizeHint())

        self.queue_list.addItem(item)

        self.queue_list.setItemWidget(
            item,
            widget,
        )

        # if not has_current and not has_queued:
        #     self.title.setText(
        #         f"{self.group.capitalize()} (idle)"
        #     )


class ElidedSummaryLabel(QLabel):
    def __init__(self, parent=None):
        super().__init__(parent)
        self._full_text = ""
        self.setWordWrap(False)
        self.setTextFormat(Qt.TextFormat.PlainText)
        self.setSizePolicy(
            QSizePolicy.Policy.Ignored,
            QSizePolicy.Policy.Fixed,
        )
        self.setFixedHeight(self.fontMetrics().height() + 6)

    def setText(self, text):
        self._full_text = str(text)
        self.setToolTip(self._full_text)
        self._update_display()

    def _update_display(self):
        super().setText(
            self.fontMetrics().elidedText(
                self._full_text,
                Qt.TextElideMode.ElideRight,
                max(0, self.contentsRect().width()),
            )
        )

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self._update_display()

    def minimumSizeHint(self):
        return QSize(0, self.height())

    def sizeHint(self):
        return QSize(140, self.height())


class TaskOverviewDisplay(QWidget):
    def __init__(self, task_manager, parent=None):
        super().__init__(parent)
        self.task_manager = task_manager
        self.setSizePolicy(
            QSizePolicy.Policy.Expanding,
            QSizePolicy.Policy.Fixed,
        )

        self.summary_label = ElidedSummaryLabel(self)

        self.details = QWidget()
        details_layout = QVBoxLayout(self.details)
        details_layout.setContentsMargins(8, 8, 8, 8)
        details_layout.setSpacing(10)

        self.queue_displays = {}
        for group in task_manager.GROUPS:
            display = TaskQueueDisplay(task_manager, group)
            self.queue_displays[group] = display
            details_layout.addWidget(display)

        self.toggle_button = ToggleOption(
            container=self.details,
            tooltip="Show task queues",
            popup=True,
            popup_size=(560, 560),
            parent=self,
        )
        self.toggle_button.setFixedSize(26, 26)
        self.toggle_button.toggled.connect(self._on_popup_toggled)

        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(6)
        # Always shown (fixed layout), enabled while tasks exist.
        self.cancel_all_button = QToolButton(self)
        self.cancel_all_button.setText("Cancel all")
        self.cancel_all_button.setToolTip(
            "Cancel all running and queued tasks (including queued session "
            "registrations)."
        )
        self.cancel_all_button.clicked.connect(self._cancel_all)

        layout.addWidget(QLabel("Tasks"))
        layout.addWidget(self.summary_label, 1)
        layout.addWidget(self.cancel_all_button)
        layout.addWidget(self.toggle_button)

        task_manager.queue_changed.connect(self.refresh_summary)
        task_manager.task_started.connect(self.refresh_summary)
        task_manager.task_finished.connect(self.refresh_summary)
        task_manager.task_cancelled.connect(self.refresh_summary)
        task_manager.task_failed.connect(self.refresh_summary)
        task_manager.scheduling_settled.connect(self.refresh_summary)

        self.refresh_summary()

    def _cancel_all(self):
        count = len(self.task_manager.tasks)
        if not count:
            return

        box = QMessageBox(self)
        box.setIcon(QMessageBox.Icon.Question)
        box.setWindowTitle("Cancel all tasks")
        box.setText(f"Cancel all {count} running and queued tasks?")
        box.setInformativeText(
            "Running tasks stop at their next safe point; completed work "
            "(e.g. already registered sessions) is kept."
        )
        cancel_all = box.addButton("Cancel all", QMessageBox.ButtonRole.DestructiveRole)
        box.addButton("Keep running", QMessageBox.ButtonRole.RejectRole)
        fit_message_box_buttons(box)
        box.exec()

        if box.clickedButton() is cancel_all:
            self.task_manager.cancel_all()

    def _on_popup_toggled(self, expanded):
        if expanded:
            for display in self.queue_displays.values():
                display.refresh()

    def refresh_summary(self, *_):
        parts = []

        for group in self.task_manager.GROUPS:
            summary = self.task_manager.group_summary(group)
            queued = summary["queued_count"]

            if not summary["running"] and not queued:
                continue

            if summary["current_cancelling"]:
                status = "cancelling"
            elif summary["running"]:
                status = "running"
            else:
                status = "waiting"

            parts.append(f"{group}: {status}, {queued} queued")

        if self.task_manager.processing_paused:
            parts.append("Waiting for source check or input")

        if self.task_manager.processing_requested:
            parts.append("Waiting for display calculation")

        self.summary_label.setText(" · ".join(parts) or "Idle")
        self.cancel_all_button.setEnabled(bool(self.task_manager.tasks))
