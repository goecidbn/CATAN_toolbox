import h5py
import numpy as np
from pathlib import Path
from scipy.io import loadmat
from typing import List, Optional, Tuple
from PySide6.QtWidgets import (
    QVBoxLayout,
    QLabel,
    QDialog,
    QListWidget,
    QListWidgetItem,
    QDialogButtonBox,
)
from PySide6.QtCore import Qt

class FieldSelectDialog(QDialog):
    def __init__(self, fields, title="Select field", parent=None):
        """
        fields: iterable of (name, shape_str, dtype_str)
        """
        super().__init__(parent)
        self.setWindowTitle(title)
        self.selected_field = None  # will hold just the *name*

        layout = QVBoxLayout(self)
        layout.addWidget(QLabel("Available fields:"))

        self.list_widget = QListWidget()
        for name, shape_str, dtype_str in fields:
            text = f"{name}  —  shape={shape_str}, dtype={dtype_str}"
            item = QListWidgetItem(text)
            # store the raw field name in user data
            item.setData(Qt.UserRole, name)
            self.list_widget.addItem(item)

        self.list_widget.setSelectionMode(QListWidget.SingleSelection)
        layout.addWidget(self.list_widget)

        buttons = QDialogButtonBox(
            QDialogButtonBox.Ok | QDialogButtonBox.Cancel,
            parent=self,
        )
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

        # double-click = OK
        self.list_widget.itemDoubleClicked.connect(self.accept)

    def accept(self):
        items = self.list_widget.selectedItems()
        if items:
            item = items[0]
            # retrieve *only* the raw name
            self.selected_field = item.data(Qt.UserRole)
        super().accept()

    @staticmethod
    def get_field(fields, title="Select field", parent=None):
        """
        fields: list of (name, shape_str, dtype_str)
        returns: selected field name (str) or None
        """
        dlg = FieldSelectDialog(fields, title, parent)
        result = dlg.exec()
        if result == QDialog.Accepted:
            return dlg.selected_field
        return None


FieldInfo = Tuple[str, str, str]   # (name, shape_str, dtype_str)

def list_hdf5_datasets(path: str) -> List[FieldInfo]:
    """
    Return a list of (name, shape, dtype) for an HDF5 file.

    Special case:
      - Groups that look like CaImAn sparse matrices (with datasets
        'indptr', 'indices', 'data', 'shape') are shown as a *single*
        logical field with:
            name  = group name (e.g. "A")
            shape = tuple from 'shape' dataset
            dtype = dtype of 'data' dataset

      - Other groups: we list their immediate datasets as 'group/dset'.
      - Top-level datasets are listed as usual.
    """
    fields: List[FieldInfo] = []

    with h5py.File(path, "r") as f:
        for key in f.keys():
            obj = f[key]

            # --- Sparse matrix group (CaImAn style) ---
            if isinstance(obj, h5py.Group):
                if all(d in obj for d in ("indptr", "indices", "data", "shape")):
                    # Treat as one field
                    shape_ds = obj["shape"][()]   # e.g. array([d, n])
                    # ensure we turn it into a nice Python tuple
                    shape_tuple = tuple(int(x) for x in np.atleast_1d(shape_ds))
                    dtype_str = str(obj["data"].dtype)
                    fields.append((key, str(shape_tuple), dtype_str))
                else:
                    # For "normal" groups: list immediate datasets inside that group
                    for subkey, subobj in obj.items():
                        if isinstance(subobj, h5py.Dataset):
                            name = f"{key}/{subkey}"
                            shape_str = str(subobj.shape)
                            dtype_str = str(subobj.dtype)
                            fields.append((name, shape_str, dtype_str))

            # --- Top-level dataset ---
            elif isinstance(obj, h5py.Dataset):
                shape_str = str(obj.shape)
                dtype_str = str(obj.dtype)
                fields.append((key, shape_str, dtype_str))

    return fields


def list_mat_fields(path: str) -> List[FieldInfo]:
    """Return top-level variables (name, shape, dtype) from a MAT file."""
    data = loadmat(path)
    fields: List[FieldInfo] = []
    for k, v in data.items():
        if k.startswith("__"):
            continue
        if isinstance(v, np.ndarray):
            shape_str = str(v.shape)
            dtype_str = str(v.dtype)
        else:
            shape_str = "-"
            dtype_str = type(v).__name__
        fields.append((k, shape_str, dtype_str))
    return fields


def list_file_fields(path: str) -> List[FieldInfo]:
    """Dispatch depending on extension (.h5/.hdf5/.mat)."""
    ext = Path(path).suffix.lower()
    if ext in (".h5", ".hdf5"):
        return list_hdf5_datasets(path)
    elif ext == ".mat":
        return list_mat_fields(path)
    else:
        raise ValueError(f"Unsupported file type for field listing: {ext}")