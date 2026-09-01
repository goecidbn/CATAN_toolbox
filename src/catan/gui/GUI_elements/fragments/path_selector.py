from pathlib import Path
from typing import Optional
from PySide6.QtWidgets import (
    QLineEdit,
    QFileDialog,
)


def choose_path(
    parent,
    pick_dir: bool = False,
    init_path: str = "",
    only_tail: bool = False,
    edit_line: Optional[QLineEdit] = None,
    display_text: str = "Select file",
    only_existing: bool = True,
) -> Optional[str]:
    if pick_dir:
        print(f"Choosing directory with initial path: {init_path}")
        path = QFileDialog.getExistingDirectory(
            parent=parent,
            caption=display_text,
            dir=init_path,  # initial directory ("" = current)
            options=QFileDialog.Option.DontUseNativeDialog,
        )
    else:
        opts = dict(
            parent=parent,
            caption=display_text,
            dir=init_path,  # initial directory ("" = current)
            filter="HDF5 files (*.hdf5 *.h5);;MATLAB files (*.mat);;All files (*)",
            options=QFileDialog.Option.DontUseNativeDialog,
        )
        if only_existing:
            path, _ = QFileDialog.getOpenFileName(**opts)
        else:
            path, _ = QFileDialog.getSaveFileName(**opts)
    if not path:
        return

    if path and edit_line is not None:
        relative_path = str(Path(path).relative_to(init_path)) if only_tail else path
        edit_line.setText(relative_path)
    elif path:
        return path
