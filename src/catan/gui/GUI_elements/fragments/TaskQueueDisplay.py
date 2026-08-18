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
        parent=None,
    ):
        super().__init__(parent)

        self.task_id = task_id

        status = "●" if running else "○"

        self.label = QLabel(f"{status} {name}")

        self.cancel_button = QPushButton("×")
        self.cancel_button.setFixedWidth(28)
        self.cancel_button.setToolTip("Cancel task")

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

        self.current_label = QLabel("Idle")

        self.queue_list = ReorderableTaskList()

        self.queue_list.task_moved.connect(
            self._move_task
        )

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        layout.addWidget(self.title)
        layout.addWidget(self.current_label)
        layout.addWidget(self.queue_list)

        self.task_manager.queue_changed.connect(
            self._queue_changed
        )

        self.task_manager.task_started.connect(
            self._task_started
        )

        self.task_manager.task_finished.connect(
            self._task_finished
        )

        self.refresh()

    def _queue_changed(self, group: str):
        if group == self.group:
            self.refresh()

    def _task_started(self, group: str, task_id: str):
        if group == self.group:
            self.refresh()

    def _task_finished(self, group: str, task_id: str):
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

    def refresh(self):
        current = self.task_manager.current_task(
            self.group
        )

        if current is None:
            self.current_label.setText("Idle")

        else:
            self.current_label.setText(
                f"● {current.name}"
            )

        self.queue_list.clear()

        for task in self.task_manager.queued_tasks(
            self.group
        ):
            item = QListWidgetItem()

            item.setData(
                Qt.ItemDataRole.UserRole,
                task.id,
            )

            widget = TaskItemWidget(
                task.id,
                task.name,
                running=False,
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


class TaskOverviewDisplay(QWidget):
    def __init__(
        self,
        task_manager,
        parent=None,
    ):
        super().__init__(parent)

        self.task_manager = task_manager

        self.summary_label = QLabel()

        self.toggle_button = QToolButton()
        self.toggle_button.setCheckable(True)
        self.toggle_button.setChecked(False)

        self.toggle_button.setArrowType(
            Qt.ArrowType.RightArrow
        )

        self.toggle_button.toggled.connect(
            self._toggle_details
        )

        header = QHBoxLayout()

        header.addWidget(QLabel("Tasks"))
        header.addWidget(self.summary_label)
        header.addStretch()
        header.addWidget(self.toggle_button)

        self.details = QWidget()

        details_layout = QVBoxLayout(
            self.details
        )

        self.loading_widget = TaskQueueDisplay(
            task_manager,
            "loading",
        )

        self.model_update_widget = TaskQueueDisplay(
            task_manager,
            "model update",
        )

        self.calculating_widget = TaskQueueDisplay(
            task_manager,
            "calculating",
        )

        details_layout.addWidget(
            self.loading_widget
        )

        details_layout.addWidget(
            self.model_update_widget
        )

        details_layout.addWidget(
            self.calculating_widget
        )

        self.details.setVisible(False)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        layout.addLayout(header)
        layout.addWidget(self.details)

        self.task_manager.queue_changed.connect(
            lambda group: self.refresh_summary()
        )

        self.task_manager.task_started.connect(
            lambda group, task_id:
                self.refresh_summary()
        )

        self.task_manager.task_finished.connect(
            lambda group, task_id:
                self.refresh_summary()
        )

        self.refresh_summary()

    def _toggle_details(
        self,
        expanded: bool,
    ):
        self.details.setVisible(expanded)

        self.toggle_button.setArrowType(
            Qt.ArrowType.DownArrow
            if expanded
            else Qt.ArrowType.RightArrow
        )

    def refresh_summary(self):
        loading = self.task_manager.group_summary(
            "loading"
        )

        calculating = self.task_manager.group_summary(
            "calculating"
        )

        loading_text = (
            f"Loading: "
            f"{'1 running' if loading['running'] else 'idle'}, "
            f"{loading['queued_count']} queued"
        )

        calc_text = (
            f"Calculating: "
            f"{'1 running' if calculating['running'] else 'idle'}, "
            f"{calculating['queued_count']} queued"
        )

        self.summary_label.setText(
            f"{loading_text}   |   {calc_text}"
        )
