from __future__ import annotations
from PySide6.QtCore import Signal
import os
from pathlib import Path

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QAbstractItemView,
    QDialog,
    QDialogButtonBox,
    QHBoxLayout,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QPushButton,
    QVBoxLayout,
)


class SessionListWidget(QListWidget):
    delete_requested = Signal(int)
    move_up_requested = Signal(int)
    move_down_requested = Signal(int)

    def keyPressEvent(self, event):

        item = self.currentItem()

        if item is None:
            super().keyPressEvent(event)
            return

        row = self.row(item)

        if event.key() == Qt.Key.Key_Delete:
            self.delete_requested.emit(row)
            return

        if (
            event.key() == Qt.Key.Key_Up
            and event.modifiers() & Qt.KeyboardModifier.ControlModifier
        ):
            self.move_up_requested.emit(row)
            return

        if (
            event.key() == Qt.Key.Key_Down
            and event.modifiers() & Qt.KeyboardModifier.ControlModifier
        ):
            self.move_down_requested.emit(row)
            return

        super().keyPressEvent(event)

class GlobReviewDialog(QDialog):
    def __init__(
        self,
        paths: list[Path],
        parent=None,
    ):
        super().__init__(parent)

        self.setWindowTitle("Review matched session files")
        self.resize(800, 500)

        self.common_path = Path(os.path.commonpath(paths))
        self._original_paths = [path.relative_to(self.common_path) for path in paths]
        self._original_paths.sort()

        self.info_label = QLabel(
            f"{len(self._original_paths)} matching files found with common path {self.common_path}."
        )

        self.list_widget = SessionListWidget()

        # Enable manual drag-and-drop reordering
        self.list_widget.setDragDropMode(
            QAbstractItemView.DragDropMode.InternalMove
        )
        self.list_widget.setDefaultDropAction(
            Qt.DropAction.MoveAction
        )

        self.list_widget.setSelectionMode(QAbstractItemView.SelectionMode.ExtendedSelection)

        for path in self._original_paths:
            self._add_path(path)

        sort_name_button = QPushButton("Sort by name")
        sort_path_button = QPushButton("Sort by path")
        reset_button = QPushButton("Reset order")
        remove_button = QPushButton("Remove selected")

        sort_name_button.clicked.connect(self.sort_by_name)
        sort_path_button.clicked.connect(self.sort_by_path)
        reset_button.clicked.connect(self.reset_order)
        remove_button.clicked.connect(self.remove_selected)
        self.list_widget.delete_requested.connect(self.remove_selected)
        self.list_widget.move_up_requested.connect(
            lambda row: self.move_item(row, row - 1)
        )
        self.list_widget.move_down_requested.connect(
            lambda row: self.move_item(row, row + 1)
        )

        controls = QHBoxLayout()
        controls.addWidget(sort_name_button)
        controls.addWidget(sort_path_button)
        controls.addWidget(reset_button)
        controls.addStretch()
        controls.addWidget(remove_button)

        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok
            | QDialogButtonBox.StandardButton.Cancel
        )

        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)

        layout = QVBoxLayout(self)
        layout.addWidget(self.info_label)
        layout.addWidget(self.list_widget)
        layout.addLayout(controls)
        layout.addWidget(buttons)

    def _add_path(self, path: Path) -> None:
        # item = QListWidgetItem(path.name)
        item = QListWidgetItem(str(path))

        # Keep the actual Path separate from what is displayed.
        item.setData(Qt.ItemDataRole.UserRole, path)

        # Full path on hover.
        item.setToolTip(str(path))

        self.list_widget.addItem(item)

    def paths(self) -> list[Path]:
        return [
            self.common_path / self.list_widget.item(i).data(
                Qt.ItemDataRole.UserRole
            )
            for i in range(self.list_widget.count())
        ]

    def sort_by_name(self) -> None:
        paths = sorted(
            self.paths(),
            key=lambda path: path.name.lower(),
        )
        self._replace_paths(paths)

    def sort_by_path(self) -> None:
        paths = sorted(
            self.paths(),
            key=lambda path: str(path).lower(),
        )
        self._replace_paths(paths)

    def reset_order(self) -> None:
        self._replace_paths(self._original_paths)
    
    def move_item(self, from_row: int, to_row: int) -> None:

        if from_row < 0 or from_row >= self.list_widget.count():
            return
        if to_row < 0 or to_row >= self.list_widget.count():
            return

        item = self.list_widget.takeItem(from_row)
        self.list_widget.insertItem(to_row, item)
        self.list_widget.setCurrentRow(to_row)

    def remove_selected(self) -> None:
        for item in self.list_widget.selectedItems():
            row = self.list_widget.row(item)
            self.list_widget.takeItem(row)

        self._update_info()

    def _replace_paths(self, paths: list[Path]) -> None:
        self.list_widget.clear()

        for path in paths:
            self._add_path(path)

        self._update_info()

    def _update_info(self) -> None:
        self.info_label.setText(
            f"{self.list_widget.count()} files selected."
        )