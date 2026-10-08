import os
from fnmatch import fnmatchcase

from PySide6.QtGui import QKeySequence, QShortcut
from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMessageBox,
    QPushButton,
    QTreeWidget,
    QTreeWidgetItem,
    QVBoxLayout,
    QInputDialog,
)

from .file_inspection import FileInspection


class GuardedPathDialog(QDialog):
    def __init__(
        self,
        parent=None,
        *,
        title="Select file",
        initial_path="",
        pick_dir=False,
        only_existing=True,
        file_filters=None,
        state=None,
        default_suffix="",
        default_filename="",
    ):
        super().__init__(parent)

        self.setWindowTitle(title)
        self.resize(760, 520)

        self.state = state
        self.pick_dir = pick_dir
        self.only_existing = only_existing
        self._save_filename = default_filename
        self.selected_path = None

        # These operations manipulate strings; they do not inspect the path.
        self._directory = os.path.abspath(
            os.path.expanduser(os.fspath(initial_path or "."))
        )
        self._entries = []
        self._action = None

        self.default_suffix = default_suffix.lstrip(".")

        layout = QVBoxLayout(self)

        navigation = QHBoxLayout()
        self.path_edit = QLineEdit(self._directory)
        self.path_edit.setPlaceholderText(
            "Type or paste a full folder/file path, then press Enter"
        )
        self.path_edit.setToolTip(
            "Navigate directly to any accessible path. "
            "Press Ctrl+L to edit this location."
        )

        self._location_shortcut = QShortcut(
            QKeySequence("Ctrl+L"),
            self,
        )
        self._location_shortcut.activated.connect(self._focus_location)
        self.up_button = QPushButton("Up")
        self.go_button = QPushButton("Go")

        self.new_folder_button = QPushButton("New folder…")

        self.home_button = QPushButton("Home")
        self.root_button = QPushButton("Root")

        for button in (self.home_button, self.root_button):
            button.setAutoDefault(False)
            button.setDefault(False)

        self.home_button.clicked.connect(lambda: self._browse(os.path.expanduser("~")))
        self.root_button.clicked.connect(
            lambda: self._browse(os.path.splitdrive(self._directory)[0] + os.sep)
        )

        navigation.addWidget(self.up_button)
        navigation.addWidget(self.home_button)
        navigation.addWidget(self.root_button)
        navigation.addWidget(self.path_edit, 1)
        navigation.addWidget(self.go_button)
        navigation.addWidget(self.new_folder_button)
        layout.addLayout(navigation)

        self.tree = QTreeWidget()
        self.tree.setHeaderLabels(["Name", "Type"])
        self.tree.setRootIsDecorated(False)
        self.tree.setColumnWidth(0, 520)
        layout.addWidget(self.tree, 1)

        selection = QHBoxLayout()
        selection.addWidget(QLabel("Folder:" if pick_dir else "File:"))
        self.name_edit = QLineEdit()
        self.name_edit.setPlaceholderText(
            "Leave empty to select this folder"
            if pick_dir
            else "File name or full path"
        )
        self.name_edit.setText(default_filename)
        self.name_edit.textEdited.connect(self._remember_save_filename)
        selection.addWidget(self.name_edit, 1)
        layout.addLayout(selection)

        options = QHBoxLayout()
        self.hidden_checkbox = QCheckBox("Show dotfiles")
        options.addWidget(self.hidden_checkbox)

        self.filter_combo = QComboBox()
        for label, patterns in file_filters or [("All files", ("*",))]:
            self.filter_combo.addItem(label, tuple(patterns))
        self.filter_combo.setVisible(not pick_dir)
        options.addWidget(self.filter_combo, 1)
        layout.addLayout(options)

        self.inspection = FileInspection(self)
        layout.addWidget(self.inspection)

        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel
        )
        self.ok_button = buttons.button(QDialogButtonBox.StandardButton.Ok)
        self.ok_button.setText(
            "Select folder" if pick_dir else "Open" if only_existing else "Save"
        )
        layout.addWidget(buttons)

        # Enter in a path/name field should trigger its own action.
        for button in (
            self.up_button,
            self.go_button,
            self.new_folder_button,
            *buttons.buttons(),
        ):
            button.setAutoDefault(False)
            button.setDefault(False)

        self.path_edit.returnPressed.connect(self._go)
        self.go_button.clicked.connect(self._go)
        self.up_button.clicked.connect(self._up)
        self.name_edit.returnPressed.connect(self.accept)
        self.new_folder_button.clicked.connect(self._create_directory)

        self.tree.itemSelectionChanged.connect(self._selection_changed)
        self.tree.itemDoubleClicked.connect(self._open_item)
        self.filter_combo.currentIndexChanged.connect(self._render)
        self.hidden_checkbox.toggled.connect(self._render)

        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)

        self.inspection.result.connect(self._on_result)
        self.inspection.failed.connect(self._on_failed)
        self.inspection.cancelled.connect(self.reject)
        self.inspection.busy_changed.connect(self._set_busy)

        self.finished.connect(lambda _result: self.inspection.invalidate())

        self._browse(
            self._directory,
            allow_missing_file=not pick_dir and not only_existing,
        )

    def _focus_location(self):
        self.path_edit.setFocus()
        self.path_edit.selectAll()

    def _remember_save_filename(self, text):
        if not self.pick_dir and not self.only_existing:
            self._save_filename = text

    def _set_filename(self, text):
        self.name_edit.setText(text)
        self._remember_save_filename(text)

    def _absolute(self, text):
        path = os.path.expanduser(text)
        if not os.path.isabs(path):
            path = os.path.join(self._directory, path)
        return os.path.abspath(path)

    def _set_busy(self, busy):
        self.tree.setEnabled(not busy)
        self.ok_button.setEnabled(not busy)
        self.name_edit.setEnabled(not busy)
        self.new_folder_button.setEnabled(not busy)

        # Navigation stays available so another path can replace a
        # stalled request. FileInspection discards the stale result.

    def _go(self):
        text = self.path_edit.text().strip()
        if text:
            self._browse(self._absolute(text))

    def _up(self):
        self._browse(os.path.dirname(self._directory))

    def _browse(self, path, *, allow_missing_file=False):
        self._action = "browse"
        self.path_edit.setText(path)
        self.inspection.start(
            "browse_directory",
            path,
            allow_missing_file=allow_missing_file,
        )

    def _render(self, *_):
        self.tree.clear()

        patterns = self.filter_combo.currentData() or ("*",)
        show_hidden = self.hidden_checkbox.isChecked()

        for name, kind in self._entries:
            if not show_hidden and name.startswith("."):
                continue

            if kind == "file":
                if self.pick_dir:
                    continue
                if not any(
                    fnmatchcase(name.casefold(), pattern.casefold())
                    for pattern in patterns
                ):
                    continue

            item = QTreeWidgetItem(
                [
                    name,
                    {
                        "directory": "Folder",
                        "file": "File",
                        "link": "Link",
                    }[kind],
                ]
            )
            item.setData(0, Qt.ItemDataRole.UserRole, (name, kind))
            self.tree.addTopLevelItem(item)

    def _selection_changed(self):
        items = self.tree.selectedItems()
        if not items:
            return

        name, kind = items[0].data(0, Qt.ItemDataRole.UserRole)

        if not self.pick_dir and not self.only_existing:
            # Directory/link navigation must not overwrite the save name.
            # A link's target is resolved by the guarded reader when opened.
            if kind != "file":
                return

        self._set_filename(name)

    def _open_item(self, item, _column):
        name, kind = item.data(0, Qt.ItemDataRole.UserRole)
        path = os.path.join(self._directory, name)

        if kind in {"directory", "link"}:
            self._browse(path)
        else:
            self.name_edit.setText(name)
            self.accept()

    def _create_directory(self):
        name, accepted = QInputDialog.getText(
            self,
            "Create folder",
            f"Folder name in:\n{self._directory}",
        )

        if not accepted or not name:
            return

        self._action = "create_directory"
        self.inspection.start(
            "create_directory",
            self._directory,
            name=name,
        )

    def accept(self):
        if not self.ok_button.isEnabled():
            return

        name = self.name_edit.text()
        if not name and not self.pick_dir:
            return

        path = self._absolute(name) if name else self._directory

        self._action = "select"

        suffix = self.default_suffix

        if not self.only_existing and not self.pick_dir:
            patterns = self.filter_combo.currentData() or ()
            if len(patterns):
                pattern = patterns[0]
                if pattern.startswith("*.") and not any(
                    char in pattern[2:] for char in "*?[]"
                ):
                    suffix = pattern[2:]

        self.inspection.start(
            "select_path",
            path,
            pick_dir=self.pick_dir,
            only_existing=self.only_existing,
            default_suffix=suffix,
        )

    def _on_result(self, result):
        if self._action == "create_directory":
            self._browse(result["path"])
            return

        if self._action == "browse":
            self._directory = result["path"]
            self._entries = result["entries"]

            self.path_edit.setText(self._directory)

            if not self.pick_dir and not self.only_existing:
                filename = result["selected"] or self._save_filename
            else:
                filename = result["selected"] or ""

            self._set_filename(filename)
            self._render()
            return

        if self._action != "select":
            return

        # In file mode, selecting a folder navigates into it.
        if result["directory"] and not self.pick_dir:
            self._browse(result["path"])
            return

        if not self.pick_dir and not self.only_existing and result["exists"]:
            answer = QMessageBox.question(
                self,
                "Replace existing file?",
                f"The file already exists:\n{result['path']}\n\n"
                "Use this path and replace the file when saving?",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                QMessageBox.StandardButton.No,
            )
            if answer != QMessageBox.StandardButton.Yes:
                return

        self.selected_path = result["path"]
        super().accept()

    def _on_failed(self):
        # FileInspection retains the request and offers Retry.
        info = self.inspection.failure_info or {}
        summary = info.get("summary") or "Could not access this path."
        recovery = info.get("recovery") or "Restore source access."

        if self.state is not None:
            self.state.issue(
                "warning",
                "File browsing failed",
                (
                    f"{summary}\n\n{recovery}\n\n"
                    "Restore access and click Retry, enter another path "
                    "and click Go, or cancel this selection."
                ),
                parent=self.window(),
            )

    @classmethod
    def get_path(cls, parent=None, **kwargs):
        dialog = cls(parent, **kwargs)
        try:
            if dialog.exec() == QDialog.DialogCode.Accepted:
                return dialog.selected_path
            return None
        finally:
            dialog.inspection.invalidate()
            dialog.deleteLater()
