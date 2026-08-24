from typing import Callable
from pathlib import Path, PurePosixPath

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QWidget,
    QVBoxLayout,
    QLabel,
    QInputDialog,
    QLineEdit,
    QToolButton,
    QGridLayout,
    QCheckBox,
    QHBoxLayout
)

from catan.core.structures.load_config import FieldGroupSpec, LoadConfig
from catan.gui.structures import data, state
from catan.gui.GUI_elements.utils.FlowLayout import FlowLayout

from .field_chip import FieldChip
from .dialog_load_field import FieldSelectDialog


class OptionList(QWidget):

    """
    Widget to display a list of options for a given field group. The options can be either static or dynamic, depending on the group specification.
    """

    def __init__(
        self,
        group_name: str,
        group_spec: FieldGroupSpec,
        callback: Callable,
    ):
        # container = QWidget()
        # structure_path = QLineEdit("/estimates")
        # layout.addWidget(structure_path)
        super().__init__()

        self.group_name = group_name
        self.list_type = group_spec.type

        self.callback = callback

        if self.list_type == "static":
            layout = QGridLayout(self)
            layout.setContentsMargins(0, 2, 0, 2)
            layout.setHorizontalSpacing(6)
            layout.setVerticalSpacing(3)
            self.list_layout = layout

        elif self.list_type == "dynamic":
            self.list_layout = FlowLayout(self)

        else:
            raise ValueError(f"Unknown load_data type: {self.list_type}")

        self.rebuild(group_spec)

    def rebuild(self, group_spec: FieldGroupSpec):

        if self.list_type != group_spec.type:
            raise ValueError(
                f"Cannot rebuild OptionList of type {self.list_type} with group_spec of type {group_spec.type}"
            )
        if group_spec.type == "static":
            self.rebuild_static_options(group_spec)
        elif group_spec.type == "dynamic":
            self.rebuild_dynamic_options(group_spec)
        else:
            raise ValueError(f"Unknown group_spec type: {group_spec.type}")

    def clear_chips(self):

        # Clear existing chips
        while (child := self.list_layout.takeAt(0)) is not None:
            if child.widget() is not None:
                child.widget().deleteLater()

    def rebuild_static_options(self, group_spec: FieldGroupSpec):

        if not isinstance(self.list_layout, QGridLayout):
            raise ValueError("Cannot rebuild static options on non-grid layout.")

        self.clear_chips()

        self.edit = {}

        for row, (name, spec) in enumerate(group_spec.fields.items()):
            self.edit[name] = QLineEdit(spec.path)

            browse = QToolButton()
            browse.setText("…")
            browse.setToolTip(f"Find {name.lower()} field")
            browse.setFixedWidth(25)
            browse.clicked.connect(
                lambda _, label=name: self.callback(
                    self.group_name, label, method="edit"
                )
            )

            self.list_layout.addWidget(
                QLabel(name.capitalize() + ":"),row,0,
            )
            self.list_layout.addWidget(
                self.edit[name],row,1,
            )
            self.list_layout.addWidget(
                browse,row,2,
            )

            # self.spatial_edits[label.lower()] = edit
        self.list_layout.setColumnStretch(1, 1)

    def rebuild_dynamic_options(self, group_spec: FieldGroupSpec):

        if not isinstance(self.list_layout, FlowLayout):
            raise ValueError("Cannot rebuild dynamic options on non-flow layout.")

        self.clear_chips()

        for name, spec in group_spec.fields.items():
            chip = FieldChip(name, spec)
            self.list_layout.addWidget(chip)

            chip.field_button.clicked.connect(
                lambda _, field_name=chip.name: self.callback(
                    self.group_name, field_name, method="rename"
                )
            )
            chip.remove_requested.connect(
                lambda chip_field_name : self.callback(
                    self.group_name, chip_field_name, method="remove"
                )
            )

        add_button = QToolButton()
        add_button.setText("+")
        add_button.setToolTip(f"Add {self.group_name} field")
        add_button.clicked.connect(lambda _, field_name=None: self.callback(self.group_name, field_name, method="add"))
        self.list_layout.addWidget(add_button)


class FieldSelector(QWidget):
    """
    Widget to manage field selection for different groups. Provides an interface to edit, add, rename, and remove fields.
    """
    
    def __init__(self, parent, config: LoadConfig):
        super().__init__(parent)

        self.state: state.AppState = parent.state
        self.data: data.Data = parent.data

        self.field_options: dict[str, OptionList] = {}
        self.opts_layout = QVBoxLayout(self)
        self.opts_layout.setContentsMargins(4, 4, 4, 4)
        self.opts_layout.setSpacing(3)

        self.rebuild(config)
        self.state.data_changed.connect(self._on_data_changed)

    def rebuild(self, config: LoadConfig):
        self.config = config

        self.clear()

        # self.opts_layout.addWidget(QLabel("Load options on registration:"))

        ## loading options
        if self.config is None:
            raise ValueError("Load config is not initialized.")

        for group_name, group_spec in self.config.groups.items():
            self.field_options[group_name] = OptionList(group_name, group_spec, self.manipulate_fields)

            opts_widget = checkbox_with_options(
                group_spec,
                self.field_options[group_name],
                self.config.groups[group_name].enabled,
                lambda checked, group_name=group_name: self.config.set_group_enabled(group_name,checked)
                )
            self.opts_layout.addWidget(opts_widget)
        self._on_data_changed(("config",-1))

    def clear(self):
        while (child := self.opts_layout.takeAt(0)) is not None:
            if child.widget() is not None:
                child.widget().deleteLater()

    def _on_data_changed(self, input: tuple[str, int]):
        """
            updates GUI element availability based on the current state of the data
        """
        ## disable changing root path, when sessions are loaded,
        ## to avoid path inconsistencies
        if input[0] not in ["sessions", "config"]:
            return
        sessions_loaded = len(self.data.sessions) > 0

        for opt in self.field_options.values():
            opt.setEnabled(sessions_loaded)
    
    def manipulate_fields(self, group_name: str, field_name: str, method="edit"):

        if self.data.current_session is None:
            path = self.data.sessions[0].path
        else:
            path = self.data.current_session.path

        assert isinstance(
            path, str | Path
        ), "No valid session path found for field selection."

        assert self.config is not None, "Load config is not initialized."


        if method == "remove":
            self.config.remove_field(
                group_name,
                field_name,
            )

        if method == "rename":
            new_name, ok = QInputDialog.getText(
                self,
                "Change field title",
                "Enter new title:",
                text=field_name,
            )
            if ok and new_name:
                self.config.rename_field(
                    group_name,
                    field_name,
                    new_name=new_name,
                )

        if method in ["edit", "add"]:
            field_path = FieldSelectDialog.get_field(path=path, key=field_name)
            if field_path is None:
                return

            if method == "edit":
                self.config.update_field(
                    group_name,
                    field_name,
                    path=field_path,
                )
            
            if method == "add":
                ## check, if a field with the same path already exists in the group
                fields_to_load = self.config.get_fields_to_load([group_name])
                if field_path in [field.path for field in fields_to_load[group_name].values()]:
                    self.state.issue(
                        "warning",
                        "Adding field is not possible",
                        f"Field {field_path} already exists in load options for {group_name}.",
                    )
                    return

                field_name = QInputDialog.getText(
                    self,
                    "Add new field",
                    "Enter field title:",
                    text=PurePosixPath(field_path).name,
                )[0]
                if not field_name:
                    field_name = PurePosixPath(field_path).name
                
                self.config.add_field(
                    group_name,
                    field_name,
                    path=field_path,
                )

        self.state.data_changed.emit(("config",-1))
        self.field_options[group_name].rebuild(
            self.config.groups[group_name]
        )


def checkbox_with_options(group_spec: FieldGroupSpec, opts_ref: QWidget, active: bool, callback: Callable) -> QWidget:
    label = group_spec.title

    container = QWidget()
    layout = QVBoxLayout(container)
    layout.setContentsMargins(0, 0, 0, 0)
    layout.setSpacing(2)

    header = QWidget()
    header_layout = QHBoxLayout(header)
    header_layout.setContentsMargins(0, 0, 0, 0)
    header_layout.setSpacing(4)

    chk = QCheckBox(label)
    chk.setChecked(active)
    chk.stateChanged.connect(
        lambda state: callback(state == Qt.CheckState.Checked.value)
    )

    toggle_button = QToolButton()
    toggle_button.setCheckable(True)
    toggle_button.setChecked(False)
    toggle_button.setAutoRaise(True)
    toggle_button.setFixedWidth(22)

    header_layout.addWidget(chk)
    header_layout.addStretch()
    header_layout.addWidget(toggle_button)

    layout.addWidget(header)

    # Indented child area
    options_container = QWidget()
    options_layout = QHBoxLayout(options_container)
    options_layout.setContentsMargins(18, 0, 0, 4)
    options_layout.addWidget(opts_ref)
    # opts_ref.setVisible(False)

    layout.addWidget(options_container)

    def toggle_options(expanded: bool):
        options_container.setVisible(expanded)
        toggle_button.setArrowType(
            Qt.ArrowType.DownArrow if expanded else Qt.ArrowType.RightArrow
        )

    toggle_button.toggled.connect(toggle_options)
    toggle_options(False)

    return container


