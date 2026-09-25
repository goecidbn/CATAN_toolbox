from pathlib import Path
from PySide6.QtCore import QObject, Qt, Signal

from PySide6.QtWidgets import (
    QInputDialog,
    QLabel,
    QMessageBox,
    QHBoxLayout,
    QComboBox,
    QSizePolicy,
    QVBoxLayout,
    QWidget,
)
from . import FieldSelector, ToggleOption
from catan.core.structures import SessionData
from catan.tracking.structures import Assignments

from .IconButton import make_icon_button


class FieldConfigConstructor(QObject):

    expanded_changed = Signal()  # Add this line to define the signal
    source_changed = Signal()

    ## those are the two main elements to be exposed
    toggle_config_options: ToggleOption
    config_options: QWidget

    def __init__(self, parent, source):
        super().__init__(parent)
        self._parent = parent
        self.state = parent.state

        self.source: SessionData | Assignments | None = None

        ## build the different GUI elements
        self.build_toggle_config_options()

        ## define the container for the configuration options
        self.config_options = QWidget()
        self.config_options.setSizePolicy(
            QSizePolicy.Policy.Minimum,
            QSizePolicy.Policy.Preferred,  # or minimum
        )
        self.config_layout = QVBoxLayout(self.config_options)
        self.config_layout.setContentsMargins(6, 6, 6, 6)
        self.config_layout.setSpacing(3)

        self.build_config_file_options()
        self.build_config_field_options()

        ## connect the container to the toggle
        self.toggle_config_options.set_container(self.config_options)
        self.source_changed.connect(self._on_source_changed)

        self.update_source(source)

    def update_source(self, source: SessionData | Assignments | None):
        self.source = source
        self.toggle_config_options.setEnabled(self.source is not None)

        expanded = (
            self.toggle_config_options.container is not None
            and self.toggle_config_options.container.isVisible()
            and (self.source is not None)
        )
        self.toggle_config_options.set_expanded(expanded)
        self.source_changed.emit()

    def _on_source_changed(self):
        self.config_field_options.update_source(self.source)
        self.rebuild_config_selector()

    def build_toggle_config_options(self):
        ## define and set toggle
        self.toggle_config_options = ToggleOption(
            icon_name="cog", tooltip="Select fields to load", expanded=False
        )

        def on_toggle_config_fields():
            self.expanded_changed.emit()

        self.toggle_config_options.toggled.connect(on_toggle_config_fields)

    def build_config_file_options(self):
        """
        the row containing the load configuration selector and associated buttons.
        """

        # clean previous layout if exists
        self.config_file_options = QWidget()
        self.config_file_options.setSizePolicy(
            QSizePolicy.Policy.Minimum,
            QSizePolicy.Policy.Preferred,  # or minimum
        )
        layout = QHBoxLayout(self.config_file_options)

        # layout.addWidget(QLabel("Load config:"))
        self.load_config_selector = QComboBox()
        self.load_config_selector.setMaximumWidth(120)
        self.rebuild_config_selector()

        layout.addWidget(
            self.load_config_selector, alignment=Qt.AlignmentFlag.AlignLeft
        )
        layout.addStretch()

        def load_config_changed(idx):

            if self.source is None or idx < 0:
                return

            names = self.state.config_manager.names(
                public_only=False, source_type=self.source.source_type
            )

            if idx >= len(names):
                return

            name = names[idx]

            self.source.source_config = self.state.config_manager.select(name)

            self.config_field_options.rebuild()
            self.expanded_changed.emit()

        self.load_config_selector.currentIndexChanged.connect(
            lambda idx: load_config_changed(idx)
        )

        self.load_config_save_button = make_icon_button(
            "floppy-disk",
            tooltip="Save load configuration",
            # size=28, icon_size=22
        )
        self.load_config_save_button.clicked.connect(self.on_save_load_config)

        self.load_config_delete_button = make_icon_button(
            "trash",
            tooltip="Delete load configuration",
            # size=28, icon_size=22
        )
        self.load_config_delete_button.clicked.connect(self.on_delete_load_config)

        # ext = Path(self.source.path).suffix
        self.load_config_set_default_button = make_icon_button(
            "file-circle-check",
            tooltip=f"Set load configuration as default for this filetype.",
            # size=28, icon_size=22,
        )
        self.load_config_set_default_button.clicked.connect(self.on_set_default)

        layout.addWidget(
            self.load_config_save_button, alignment=Qt.AlignmentFlag.AlignRight
        )
        layout.addWidget(
            self.load_config_delete_button, alignment=Qt.AlignmentFlag.AlignRight
        )
        layout.addWidget(
            self.load_config_set_default_button, alignment=Qt.AlignmentFlag.AlignRight
        )
        self.config_layout.addWidget(self.config_file_options)

    def rebuild_config_selector(self):
        selector = self.load_config_selector

        selector.blockSignals(True)
        selector.clear()
        if self.source is None:
            selector.blockSignals(False)
            return
        names = self.state.config_manager.names(
            public_only=False, source_type=self.source.source_type
        )
        selector.addItems(names)

        if self.source.source_config is None:
            selector.setCurrentIndex(-1)

        for i, name in enumerate(names):
            selector.setItemData(
                i,
                name,
                role=Qt.ItemDataRole.ToolTipRole,
            )
            if name == getattr(self.source.source_config, "name", None):
                selector.setCurrentIndex(i)
        selector.blockSignals(False)

    def build_config_field_options(self):

        self.config_field_options = FieldSelector(self._parent, self.source)
        self.config_field_options.fields_changed.connect(self._on_fields_changed)
        self.config_layout.addWidget(self.config_field_options)

    def on_save_load_config(self):
        if self.source is None:
            return

        if self.source.source_config is None:
            self.state.issue(
                "warning",
                "No load configuration selected",
                "Please select a load configuration to save.",
            )
            return
        new_name, ok = QInputDialog.getText(
            self._parent,
            "Save load configuration",
            "Enter a name for the load configuration:",
            text=self.source.source_config.name or "",
        )
        if ok and new_name:
            # print("saving source_config:", self.source.source_config.name)
            self.state.config_manager.save_config(self.source.source_config, new_name)

        self.rebuild_config_selector()

    def on_delete_load_config(self):
        if self.source is None:
            return

        if self.source.source_config is None:
            self.state.issue(
                "warning",
                "No load configuration selected",
                "Please select a load configuration to delete.",
            )
            return
        if self.state.config_manager.modified(self.source.source_config):
            self.state.issue(
                "warning",
                "Modified load configuration",
                "The load configuration has been modified. You can only delete unmodified configurations.",
            )
            return
        if getattr(self.source.source_config, "native", False):
            self.state.issue(
                "warning",
                "Native load configuration",
                "CATAN-native load configurations cannot be deleted.",
            )
            return

        name = getattr(self.source.source_config, "name", None)
        reply = QMessageBox.question(
            self._parent,
            "Delete load configuration",
            f"Are you sure you want to delete the load configuration '{name}'?",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
        )
        if reply == QMessageBox.StandardButton.Yes:
            self.state.config_manager.delete(name)
            self.rebuild_config_selector()

    def on_set_default(self):
        if self.source is None:
            return

        self.state.config_manager.set_default_for_format(
            self.source.path, self.source.source_config
        )
        self._on_fields_changed()

    def _on_fields_changed(self):

        if self.source is None:
            return

        ## changes to config file options
        self.expanded_changed.emit()
        if self.source.source_config is None:
            return

        is_default = self.state.config_manager.is_default(
            self.source.path, self.source.source_config
        )
        is_modified = self.state.config_manager.modified(self.source.source_config)

        ## enable / disable buttons based on current modified status
        self.load_config_save_button.setEnabled(is_modified)
        self.load_config_set_default_button.setEnabled(
            not is_default and not is_modified
        )

        ## adjust display of current config file to highlight if it is modified
        self.load_config_selector.setEditable(is_modified)
        self.load_config_selector.setCurrentText(
            "* "
            + getattr(self.source.source_config, "name", "")
            + (" (modified)" if is_modified else "")
        )

        self.state.statistics_sources_changed.emit()

        # loading_possible = self.field_selector.loading_possible
