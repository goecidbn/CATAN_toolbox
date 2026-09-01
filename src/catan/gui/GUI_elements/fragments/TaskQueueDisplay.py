from PySide6.QtCore import Signal, Qt
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
)


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

        self.setDragDropMode(
            QAbstractItemView.DragDropMode.InternalMove
        )

        self.setSelectionMode(
            QAbstractItemView.SelectionMode.SingleSelection
        )

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
        self.current_layout = QVBoxLayout(
            self.current_container
        )
        self.current_layout.setContentsMargins(
            0, 0, 0, 0
        )
        self.current_layout.setSpacing(0)

        # Queued tasks
        self.queue_list = ReorderableTaskList()
        self.queue_list.task_moved.connect(
            self._move_task
        )

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        layout.addWidget(self.title)
        layout.addWidget(self.current_container)
        layout.addWidget(self.queue_list)

        # ---- TaskManager signals ----

        self.task_manager.queue_changed.connect(
            self._queue_changed
        )

        self.task_manager.task_started.connect(
            self._task_changed
        )

        self.task_manager.task_finished.connect(
            self._task_changed
        )

        self.task_manager.task_cancelled.connect(
            self._task_changed
        )

        self.task_manager.task_failed.connect(
            self._task_changed
        )

        self.refresh()

    def _queue_changed(
        self,
        group: str,
    ):
        if group == self.group:
            self.refresh()

    def _task_changed(
        self,
        group: str,
        task_id: str,
    ):
        if group == self.group:
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
        # ============================================================
        # Current task
        # ============================================================

        summary = (
            self.task_manager.group_summary(
                self.group
            )
        )

        if summary["running"]:
            if summary[
                "current_cancelling"
            ]:
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


        current = self.task_manager.current_task(
            self.group
        )

        self._clear_layout(
            self.current_layout
        )

        has_current = current is not None
        if current is None:
            pass
            # self.current_layout.addWidget(
            #     QLabel("Idle")
            # )

        else:
            current_widget = TaskItemWidget(
                task_id=current.id,
                name=current.name,
                running=True,
                cancelling=current.worker.is_cancelled(),
            )

            current_widget.cancel_requested.connect(
                self.task_manager.cancel
            )

            self.current_layout.addWidget(
                current_widget
            )

        # ============================================================
        # Queued tasks
        # ============================================================

        self.queue_list.clear()

        n_queued = len(self.task_manager.queued_tasks(self.group))
        has_queued = n_queued > 0
        self.queue_list.setVisible(has_queued)

        for task in self.task_manager.queued_tasks(
            self.group
        ):
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

            widget.cancel_requested.connect(
                self.task_manager.cancel
            )

            item.setSizeHint(
                widget.sizeHint()
            )

            self.queue_list.addItem(item)

            self.queue_list.setItemWidget(
                item,
                widget,
            )

        # if not has_current and not has_queued:
        #     self.title.setText(
        #         f"{self.group.capitalize()} (idle)"
        #     )


class TaskOverviewDisplay(QWidget):
    def __init__(
        self,
        task_manager,
        parent=None,
    ):
        super().__init__(parent)

        self.task_manager = task_manager

        # self.summary_label = QLabel()

        # ============================================================
        # Expand/collapse button
        # ============================================================

        self.toggle_button = QToolButton()

        self.toggle_button.setCheckable(True)
        self.toggle_button.setChecked(False)

        self.toggle_button.setArrowType(
            Qt.ArrowType.RightArrow
        )

        self.toggle_button.toggled.connect(
            self._toggle_details
        )

        # ============================================================
        # Header
        # ============================================================

        header = QHBoxLayout()

        header.addWidget(
            QLabel("Tasks")
        )

        # header.addWidget(
        #     self.summary_label
        # )

        header.addStretch()

        header.addWidget(
            self.toggle_button
        )

        # ============================================================
        # Detailed task queues
        # ============================================================

        self.details = QWidget()

        details_layout = QVBoxLayout(
            self.details
        )

        details_layout.setContentsMargins(
            0, 0, 0, 0
        )

        self.queue_displays = {}

        for group in self.task_manager.GROUPS:
            display = TaskQueueDisplay(
                task_manager,
                group,
            )

            self.queue_displays[group] = display

            details_layout.addWidget(
                display
            )

        self.details.setVisible(False)

        # ============================================================
        # Main layout
        # ============================================================

        layout = QVBoxLayout(self)

        layout.setContentsMargins(
            0, 0, 0, 0
        )

        layout.addLayout(header)
        layout.addWidget(self.details)

        # ============================================================
        # TaskManager signals
        # ============================================================

        # self.task_manager.queue_changed.connect(
        #     lambda group:
        #         self.refresh_summary()
        # )

        # self.task_manager.task_started.connect(
        #     lambda group, task_id:
        #         self.refresh_summary()
        # )

        # self.task_manager.task_finished.connect(
        #     lambda group, task_id:
        #         self.refresh_summary()
        # )

        # self.task_manager.task_cancelled.connect(
        #     lambda group, task_id:
        #         self.refresh_summary()
        # )

        # self.task_manager.task_failed.connect(
        #     lambda group, task_id:
        #         self.refresh_summary()
        # )

        # self.refresh_summary()

    def _toggle_details(
        self,
        expanded: bool,
    ):
        self.details.setVisible(
            expanded
        )

        self.toggle_button.setArrowType(
            Qt.ArrowType.DownArrow
            if expanded
            else Qt.ArrowType.RightArrow
        )

    # def refresh_summary(self):
    #     parts = []

    #     for group in self.task_manager.GROUPS:

    #         summary = (
    #             self.task_manager.group_summary(
    #                 group
    #             )
    #         )

    #         if summary["running"]:
    #             if summary[
    #                 "current_cancelling"
    #             ]:
    #                 state = "cancelling"
    #             else:
    #                 state = "1 running"
    #         else:
    #             state = "idle"

    #         parts.append(
    #             f"{group.capitalize()}: "
    #             f"{state}, "
    #             f"{summary['queued_count']} queued"
    #         )

    #     self.summary_label.setText(
    #         "\n".join(parts)
    #     )