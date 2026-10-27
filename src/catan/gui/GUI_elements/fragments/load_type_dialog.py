"""Choose what a file holds when CATAN cannot tell by itself ("+ from file")."""

from pathlib import Path

from PySide6.QtWidgets import (
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QFormLayout,
    QLabel,
    QVBoxLayout,
)

from catan.gui.GUI_elements.utils.paths import display_path

# (kind, label, preset source type or None)
LOAD_TYPES = (
    ("sessions", "Session data", "session"),
    ("assignments", "Assignments (neuron × session IDs)", "assignments"),
    ("model", "Matching model (CATAN model file)", None),
    ("recipe", "Loading recipe (CATAN .json)", None),
    ("table", "Path table (one session path per row)", None),
)


class LoadTypeDialog(QDialog):
    """Type and, for sessions / assignments, the load preset of a file.

    ``presets``: {source type: suggested LoadConfig or None}; ``names``:
    {source type: preset names}. After exec(): kind, preset_name."""

    def __init__(self, path, presets, names, parent=None):
        super().__init__(parent)
        self.setWindowTitle("What does this file contain?")
        self._presets = presets
        self._names = names

        layout = QVBoxLayout(self)
        note = QLabel(
            f"{display_path(str(path))}\n\n"
            "CATAN could not identify this file: it is no CATAN file, and "
            "none or several of the suggested load presets match it. Choose "
            "what it contains and the preset to read it with. Missing fields "
            "are then shown in the load configuration."
        )
        note.setWordWrap(True)
        layout.addWidget(note)

        form = QFormLayout()
        self.kind = QComboBox(self)
        for kind, label, _ in LOAD_TYPES:
            self.kind.addItem(label, kind)
        form.addRow("Contains", self.kind)

        self.preset = QComboBox(self)
        form.addRow("Load preset", self.preset)
        layout.addLayout(form)

        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel
        )
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

        self.kind.currentIndexChanged.connect(self._refresh_presets)
        if Path(path).suffix.lower() == ".json":
            self.kind.setCurrentIndex(self.kind.findData("recipe"))
        self._refresh_presets()

    @property
    def source_type(self):
        return dict((kind, source) for kind, _, source in LOAD_TYPES)[self.kind.currentData()]

    def _refresh_presets(self, *_):
        source_type = self.source_type
        self.preset.clear()
        self.preset.setEnabled(source_type is not None)
        if source_type is None:
            return
        names = list(self._names.get(source_type, ()))
        self.preset.addItems(names)
        suggested = self._presets.get(source_type)
        name = getattr(suggested, "name", None)
        if name in names:
            self.preset.setCurrentText(name)

    @property
    def preset_name(self):
        return self.preset.currentText() if self.preset.isEnabled() else None
