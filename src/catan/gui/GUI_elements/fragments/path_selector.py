from pathlib import Path
from typing import Optional

from PySide6.QtWidgets import QLineEdit

from .guarded_path_dialog import GuardedPathDialog


def _find_state(parent):
    widget = parent
    while widget is not None:
        state = getattr(widget, "state", None)
        if state is not None and callable(getattr(state, "issue", None)):
            return state
        widget = widget.parentWidget()
    return None


def choose_path(
    parent,
    pick_dir: bool = False,
    init_path: str = "",
    only_tail: bool = False,
    edit_line: Optional[QLineEdit] = None,
    display_text: str = "Select file",
    only_existing: bool = True,
    *,
    state=None,
    file_filters=None,
    default_suffix="",
    default_filename="",
) -> Optional[str]:
    if file_filters is None:
        file_filters = [
            (
                "Supported files",
                (
                    "*.hdf5",
                    "*.h5",
                    "*.mat",
                    "*.npz",
                    "*.json",
                    "*.png",
                    "*.jpg",
                    "*.jpeg",
                    "*.tif",
                    "*.tiff",
                    "*.bmp",
                ),
            ),
            ("HDF5 files", ("*.hdf5", "*.h5")),
            ("MATLAB files", ("*.mat",)),
            ("CATAN loading recipe", ("*.json",)),
            ("All files", ("*",)),
        ]

    path = GuardedPathDialog.get_path(
        parent,
        title=display_text,
        initial_path=init_path,
        pick_dir=pick_dir,
        only_existing=only_existing,
        file_filters=file_filters,
        state=state if state is not None else _find_state(parent),
        default_suffix=default_suffix,
        default_filename=default_filename,
    )

    if path is None:
        return None

    if edit_line is not None:
        if only_tail:
            try:
                displayed = str(Path(path).relative_to(init_path))
            except ValueError:
                # Selection outside the initial folder remains valid.
                displayed = path
        else:
            displayed = path

        edit_line.setText(displayed)
        return None

    return path
