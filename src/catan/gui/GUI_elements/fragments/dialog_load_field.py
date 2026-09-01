from dataclasses import dataclass
from catan.core.io.inspection import browse_file_fields
# import h5py
import numpy as np
from pathlib import Path, PurePosixPath
# from scipy.io import loadmat
from typing import List, Optional, Tuple
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
)
from PySide6.QtCore import Qt


class FieldSelectDialog(QDialog):

    def __init__(
        self,
        path: str,
        title: str = "Select field",
        parent=None,
        key: str | None = None,
        subpath: str = "/",
    ):
        super().__init__(parent)

        self.path = path
        self.current_path = self._normalize_path(subpath)

        self.selected_field: str | None = None
        self.initial_key = key

        self.setWindowTitle(title)
        self.resize(500, 400)

        layout = QVBoxLayout(self)

        self.path_label = QLabel()
        layout.addWidget(self.path_label)

        self.tree_widget = QTreeWidget()

        self.tree_widget.setColumnCount(3)
        self.tree_widget.setHeaderLabels(
            ["Field", "Shape", "dtype"]
        )

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
            QDialogButtonBox.StandardButton.Ok
            | QDialogButtonBox.StandardButton.Cancel,
            parent=self,
        )

        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)

        layout.addWidget(buttons)

        self.tree_widget.itemDoubleClicked.connect(
            self._item_double_clicked
        )

        self._populate()

    @staticmethod
    def _normalize_path(path: str) -> str:
        if not path:
            return "/"

        path = "/" + path.strip("/")

        return path if path != "" else "/"

    def _populate(self):
        self.tree_widget.clear()

        self.path_label.setText(
            f"Path: {self.current_path}"
        )

        fields = browse_file_fields(
            self.path,
            subpath=self.current_path,
        )
        # list_file_fields(
        #     self.path,
        #     subpath=self.current_path,
        # )

        selected_item = None

        # Add '..' first
        if self.current_path != "/":
            item = QTreeWidgetItem(
                ["📁  ..", "", ""]
            )

            item.setData(
                0,
                Qt.ItemDataRole.UserRole,
                {
                    "kind": "parent",
                    "name": "..",
                },
            )

            self.tree_widget.addTopLevelItem(item)

        # groups first, then datasets
        fields = sorted(
            fields,
            key=lambda f: (
                f.kind != "group",
                f.name.lower(),
            ),
        )

        for field in fields:

            if field.kind == "group":
                item = QTreeWidgetItem(
                    [
                        f"📁  {field.name}",
                        "",
                        "",
                    ]
                )
                font = item.font(0)
                font.setBold(True)
                item.setFont(0, font)

            else:
                item = QTreeWidgetItem(
                    [
                        field.name,
                        field.display_shape,
                        field.display_dtype,
                    ]
                )

            item.setData(
                0,
                Qt.ItemDataRole.UserRole,
                {
                    "kind": field.kind,
                    "name": field.name,
                },
            )

            self.tree_widget.addTopLevelItem(item)

            if (
                self.initial_key is not None
                and field.name == self.initial_key
            ):
                selected_item = item

        if selected_item is not None:
            self.tree_widget.setCurrentItem(
                selected_item
            )
            selected_item.setSelected(True)

            self.tree_widget.scrollToItem(
                selected_item
            )

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

        print(f"Double-clicked on {kind}: {name}")
        print("data:", data)

        if kind == "parent":
            self._go_up()

        elif kind == "group":
            self._enter_group(name)

        elif kind == "field":
            self.accept()

    def _enter_group(self, name: str):
        if self.current_path == "/":
            self.current_path = f"/{name}"
        else:
            self.current_path = (
                f"{self.current_path}/{name}"
            )

        # Preselection should only apply initially
        self.initial_key = None

        self._populate()

    def _go_up(self):
        
        path = PurePosixPath(
            self.current_path
        )

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

        data = item.data(
            0,
            Qt.ItemDataRole.UserRole,
        )

        if data["kind"] != "field":
            return

        path = PurePosixPath(self.current_path,data["name"])
        self.selected_field = str(path)

        # if self.current_path == "/":
        #     self.selected_field = name
        # else:
        #     self.selected_field = (
        #         f"{self.current_path.strip('/')}/{name}"
        #     )

        super().accept()

    @staticmethod
    def get_field(
        path: str,
        title: str = "Select field",
        parent=None,
        key: str | None = None,
        subpath: str = "/",
    ):
        dlg = FieldSelectDialog(
            path=path,
            title=title,
            parent=parent,
            key=key,
            subpath=subpath,
        )

        result = dlg.exec()

        if result == QDialog.DialogCode.Accepted:
            # return dlg.current_path, dlg.selected_field
            return dlg.selected_field

        return None

# @dataclass
# class FieldInfo:
#     name: str
#     kind: str          # "group" or "dataset"
#     shape: str = ""
#     dtype: str = ""

# def list_hdf5_datasets(path: str, subpath: str = "/") -> List[FieldInfo]:
#     """
#     Return a list of (name, shape, dtype) for an HDF5 file.

#     Special case:
#       - Groups that look like CaImAn sparse matrices (with datasets
#         'indptr', 'indices', 'data', 'shape') are shown as a *single*
#         logical field with:
#             name  = group name (e.g. "A")
#             shape = tuple from 'shape' dataset
#             dtype = dtype of 'data' dataset

#       - Other groups: we list their immediate datasets as 'group/dset'.
#       - Top-level datasets are listed as usual.
#     """
#     fields = []

#     with h5py.File(path, "r") as f:
#         group = f[subpath]

#         if not isinstance(group, h5py.Group):
#             raise ValueError(
#                 f"{subpath!r} is not an HDF5 group."
#             )

#         for name, obj in group.items():

#             if isinstance(obj, h5py.Group):

#                 # CaImAn sparse matrix:
#                 # treat the whole group as one selectable logical field
#                 if all(
#                     key in obj
#                     for key in ("indptr", "indices", "data", "shape")
#                 ):
#                     shape_ds = obj["shape"][()]
#                     shape_tuple = tuple(
#                         int(x)
#                         for x in np.atleast_1d(shape_ds)
#                     )

#                     fields.append(
#                         FieldInfo(
#                             name=name,
#                             kind="dataset",
#                             shape=str(shape_tuple),
#                             dtype=str(obj["data"].dtype),
#                         )
#                     )

#                 else:
#                     fields.append(
#                         FieldInfo(
#                             name=name,
#                             kind="group",
#                         )
#                     )

#             elif isinstance(obj, h5py.Dataset):
#                 fields.append(
#                     FieldInfo(
#                         name=name,
#                         kind="dataset",
#                         shape=str(obj.shape),
#                         dtype=str(obj.dtype),
#                     )
#                 )

#     return fields


# def list_mat_fields(path: str, subpath="/") -> List[FieldInfo]:
#     """Return top-level variables (name, shape, dtype) from a MAT file."""
#     data = loadmat(path)
#     fields: List[FieldInfo] = []
#     for k, v in data.items():
#         if k.startswith("__"):
#             continue
#         if isinstance(v, np.ndarray):
#             shape_str = str(v.shape)
#             dtype_str = str(v.dtype)
#         else:
#             shape_str = "-"
#             dtype_str = type(v).__name__
#         fields.append(
#             FieldInfo(
#                 name=k,
#                 kind="dataset",
#                 shape=shape_str,
#                 dtype=dtype_str,
#             )
#         )
#     return fields


# def list_file_fields(path: str, subpath="/") -> List[FieldInfo]:
#     """Dispatch depending on extension (.h5/.hdf5/.mat)."""
#     ext = Path(path).suffix.lower()
#     if ext in (".h5", ".hdf5"):
#         return list_hdf5_datasets(path, subpath)
#     elif ext == ".mat":
#         return list_mat_fields(path, subpath)
#     else:
#         raise ValueError(f"Unsupported file type for field listing: {ext}")