from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np
from pathlib import Path, PurePosixPath

from PySide6.QtGui import QAction
from PySide6.QtWidgets import (
    QAbstractItemView,
    QTreeWidgetItem,
    QTreeWidget,
    QVBoxLayout,
    QLabel,
    QDialog,
    QListWidget,
    QListWidgetItem,
    QDialogButtonBox,
    QHeaderView,
    QFileDialog,
    QHBoxLayout,
    QToolButton,
    QMenu,
)
from PySide6.QtCore import Qt

from catan.core.io import resolve_source_path
from catan.core.io.types import FieldSource
from catan.core.io.inspection import browse_file_fields
from catan.core.structures.load_config import FieldSpec


@dataclass(frozen=True)
class FieldSelection:

    path: str
    source_path: str | None = None
    source: FieldSource = "dataset"
    attribute: str | None = None


class FieldSelectDialog(QDialog):

    def __init__(
        self,
        path: str,
        title: str = "Select field",
        parent=None,
        key: str | None = None,
        subpath: str = "/",
        spec: FieldSpec | None = None,
        context: str | None = None,
    ):
        super().__init__(parent)

        self.primary_path = str(path)
        self.source_path = None if spec is None else spec.source_path

        self.path = str(resolve_source_path(self.primary_path, self.source_path))

        self.selected_field: FieldSelection | None = None

        if spec is not None:
            selected_path = PurePosixPath(spec.path)
            parent_path = str(selected_path.parent)
            self.current_path = (
                "/" if parent_path in ("", ".") else self._normalize_path(parent_path)
            )
            self.initial_field_path = spec.path

        else:
            self.current_path = self._normalize_path(subpath)
            self.initial_field_path = None

        self.initial_key = key

        self.setWindowTitle(title)
        self.resize(500, 400)

        layout = QVBoxLayout(self)
        if context:
            self.context_label = QLabel(context)
            self.context_label.setStyleSheet("font-weight: 600;")
            layout.addWidget(self.context_label)

        source_layout = QHBoxLayout()

        self.source_label = QLabel()
        source_layout.addWidget(self.source_label, stretch=1)

        self.source_button = QToolButton()
        self.source_button.setText("Source…")
        self.source_button.setPopupMode(QToolButton.ToolButtonPopupMode.InstantPopup)

        self.source_menu = QMenu(self.source_button)

        choose_file = QAction("Load from different file…", self.source_menu)
        choose_file.triggered.connect(self._choose_source_file)
        self.source_menu.addAction(choose_file)

        choose_directory = QAction("Load from source directory…", self.source_menu)
        choose_directory.triggered.connect(self._choose_source_directory)
        self.source_menu.addAction(choose_directory)

        self.source_menu.addSeparator()

        self.revert_source_action = QAction(
            "Revert to session source", self.source_menu
        )
        self.revert_source_action.triggered.connect(self._revert_source)
        self.source_menu.addAction(self.revert_source_action)

        self.source_button.setMenu(self.source_menu)
        source_layout.addWidget(self.source_button)
        layout.addLayout(source_layout)
        self._update_source_display()

        self.path_label = QLabel()
        layout.addWidget(self.path_label)

        self.tree_widget = QTreeWidget()

        self.tree_widget.setColumnCount(3)
        self.tree_widget.setHeaderLabels(["Field", "Shape", "dtype"])

        self.tree_widget.setSelectionMode(
            QAbstractItemView.SelectionMode.SingleSelection
        )

        self.tree_widget.setRootIsDecorated(False)
        self.tree_widget.setAlternatingRowColors(True)

        # Give most space to the field name

        header = self.tree_widget.header()

        header.setSectionResizeMode(
            0,
            QHeaderView.ResizeMode.Stretch,
        )
        header.setSectionResizeMode(
            1,
            QHeaderView.ResizeMode.ResizeToContents,
        )
        header.setSectionResizeMode(
            2,
            QHeaderView.ResizeMode.ResizeToContents,
        )

        # self.list_widget = QListWidget()
        # self.list_widget.setSelectionMode(
        #     QListWidget.SelectionMode.SingleSelection
        # )

        layout.addWidget(self.tree_widget)

        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel,
            parent=self,
        )

        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)

        layout.addWidget(buttons)

        self.tree_widget.itemDoubleClicked.connect(self._item_double_clicked)

        self._populate()

    @staticmethod
    def _normalize_path(path: str) -> str:
        if not path:
            return "/"

        path = "/" + path.strip("/")

        return path if path != "" else "/"

    def _populate(self):

        self.tree_widget.clear()
        self.path_label.setText(f"Path: {self.current_path}")
        fields = browse_file_fields(self.path, subpath=self.current_path)

        selected_item = None

        # ---------------------------------------------------------
        # Parent directory entry
        # ---------------------------------------------------------

        if self.current_path != "/":

            item = QTreeWidgetItem(["📁  ..", "", ""])

            item.setData(
                0,
                Qt.ItemDataRole.UserRole,
                {
                    "kind": "parent",
                    "name": "..",
                    "path": None,
                    "attribute": None,
                },
            )

            self.tree_widget.addTopLevelItem(item)

        # ---------------------------------------------------------
        # Groups first, then selectable fields / attributes
        # ---------------------------------------------------------

        fields = sorted(
            fields,
            key=lambda field: (field.kind != "group", field.name.lower()),
        )

        for field in fields:

            if field.kind == "group":

                item = QTreeWidgetItem([f"📁  {field.name}", "", ""])

                font = item.font(0)
                font.setBold(True)
                item.setFont(0, font)

            else:

                item = QTreeWidgetItem(
                    [field.name, field.display_shape, field.display_dtype]
                )

            # Keep the complete FieldInfo identity on the tree item.
            # In particular, do not reconstruct the path later from
            # current_path + name.
            item.setData(
                0,
                Qt.ItemDataRole.UserRole,
                {
                    "kind": field.kind,
                    "name": field.name,
                    "path": field.path,
                    "attribute": field.attribute,
                },
            )

            self.tree_widget.addTopLevelItem(item)

            # -----------------------------------------------------
            # Preselect the currently configured field
            # -----------------------------------------------------

            if (
                self.initial_field_path is not None
                and field.path == self.initial_field_path
            ):
                selected_item = item

            elif (
                self.initial_field_path is None
                and self.initial_key is not None
                and field.name == self.initial_key
            ):
                selected_item = item

        # ---------------------------------------------------------
        # Apply preselection
        # ---------------------------------------------------------

        if selected_item is not None:

            self.tree_widget.setCurrentItem(selected_item)
            selected_item.setSelected(True)
            self.tree_widget.scrollToItem(selected_item)

    def _item_double_clicked(
        self,
        item: QTreeWidgetItem,
        column: int,
    ):
        data = item.data(
            0,
            Qt.ItemDataRole.UserRole,
        )

        kind = data["kind"]
        name = data["name"]

        if kind == "parent":
            self._go_up()

        elif kind == "group":
            self._enter_group(name)

        elif kind in ("field", "attribute"):
            self.accept()

    def _enter_group(self, name: str):
        if self.current_path == "/":
            self.current_path = f"/{name}"
        else:
            self.current_path = f"{self.current_path}/{name}"

        # Preselection should only apply initially
        self.initial_key = None

        self._populate()

    def _go_up(self):

        path = PurePosixPath(self.current_path)

        parent = str(path.parent)

        if parent == ".":
            parent = "/"

        self.current_path = parent
        self.initial_key = None

        self._populate()

    def accept(self):

        item = self.tree_widget.currentItem()

        if item is None:
            return

        data = item.data(0, Qt.ItemDataRole.UserRole)

        kind = data["kind"]
        if kind not in ("field", "attribute"):
            return

        if kind == "attribute":

            self.selected_field = FieldSelection(
                path=data["path"],
                source_path=self.source_path,
                source="attribute",
                attribute=data["attribute"],
            )

        else:

            self.selected_field = FieldSelection(
                path=data["path"],
                source_path=self.source_path,
                source="dataset",
                attribute=None,
            )

        super().accept()

    @staticmethod
    def get_field(
        path: str,
        title: str = "Select field",
        parent=None,
        key: str | None = None,
        subpath: str = "/",
        spec: FieldSpec | None = None,
        context: str | None = None,
    ):
        dlg = FieldSelectDialog(
            path=path,
            title=title,
            parent=parent,
            key=key,
            subpath=subpath,
            spec=spec,
            context=context,
        )

        result = dlg.exec()

        if result == QDialog.DialogCode.Accepted:
            # return dlg.current_path, dlg.selected_field
            return dlg.selected_field

        return None

    def _update_source_display(self):

        if self.source_path is None:
            self.source_label.setText("Source: session source")
            self.source_label.setToolTip(self.primary_path)
            self.revert_source_action.setEnabled(False)

        else:
            self.source_label.setText(f"Source: {Path(self.path).name}")
            self.source_label.setToolTip(self.path)
            self.revert_source_action.setEnabled(True)

    def _set_source(self, source_path: str | None):

        self.source_path = source_path

        self.path = str(resolve_source_path(self.primary_path, source_path))

        self.current_path = "/"
        self.initial_key = None
        self.initial_field_path = None

        self._update_source_display()
        self._populate()

    def _choose_source_file(self):

        start = str(Path(self.path).parent)

        path, _ = QFileDialog.getOpenFileName(self, "Select field source", start)

        if not path:
            return

        self._set_source(path)

    def _choose_source_directory(self):

        start = str(Path(self.path).parent)

        path = QFileDialog.getExistingDirectory(
            self, "Select field source directory", start
        )

        if not path:
            return

        self._set_source(path)

    def _revert_source(self):
        self._set_source(None)
