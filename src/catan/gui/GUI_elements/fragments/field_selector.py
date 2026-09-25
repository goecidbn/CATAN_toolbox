from typing import Callable, Optional
from pathlib import Path, PurePosixPath

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QSizePolicy,
    QWidget,
    QVBoxLayout,
    QLabel,
    QInputDialog,
    QLineEdit,
    QToolButton,
    QGridLayout,
    QCheckBox,
    QHBoxLayout,
)

from catan.core.io.inspection import check_fields_compatibility
from catan.core.structures.load_config import FieldGroupSpec, FieldSpec
from catan.tracking.structures import Assignments
from catan.gui.structures import SessionData, data, state
from catan.gui.GUI_elements.utils.FlowLayout import FlowLayout, QSizePolicy

from .field_chip import FieldChip
from .dialog_load_field import FieldSelectDialog

COMPAT_COLORS = {
    "available": "#4F7F5A",  # muted green
    "optional_missing": "#8A7040",  # muted amber
    "required_missing": "#8A4F52",  # muted red
}
# green  = "#3F6548"
# amber  = "#6F5B35"
# red    = "#6F4144"
COMPAT_BORDERS = {
    "available": "#6A9A74",
    "optional_missing": "#A88A52",
    "required_missing": "#A8676B",
}
TEXT_COLOR = "#E8E8E8"


class FieldEditor(QLineEdit):
    name: str
    _field_path: str

    def __init__(
        self,
        name: str,
        spec: FieldSpec,
    ):
        super().__init__(spec.path)

        self.name = name
        self.set(path=spec.path)

        # self.setText(self.field_path)

    @property
    def field_path(self) -> str:
        return self.text()

    @field_path.setter
    def field_path(self, value: str):
        self._field_path = value
        self.setText(value)
        self.setToolTip(f"{self.name}: {value}")

    def set(self, *, name: Optional[str] = None, path: Optional[str] = None):

        if name is not None:
            self.name = name
        if path is not None:
            self.field_path = path


class OptionList(QWidget):
    """
    Widget to display a list of options for a given field group. The options can be either static or dynamic, depending on the group specification.
    """

    def __init__(
        self,
        group_name: str,
        group_spec: FieldGroupSpec,
        callback: Callable,
        enabled: bool = True,
    ):
        # container = QWidget()
        # structure_path = QLineEdit("/estimates")
        # layout.addWidget(structure_path)
        super().__init__()

        self.group_name = group_name
        self.list_type = group_spec.type
        self.group_spec = group_spec

        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)

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

        self.rebuild(group_spec, enabled=enabled)

    def rebuild(self, group_spec: FieldGroupSpec, enabled: bool = True):
        self.group_spec = group_spec
        if self.list_type != group_spec.type:
            raise ValueError(
                f"Cannot rebuild OptionList of type {self.list_type} with group_spec of type {group_spec.type}"
            )
        if group_spec.type == "static":
            self.rebuild_static_options(group_spec, enabled=enabled)
        elif group_spec.type == "dynamic":
            self.rebuild_dynamic_options(group_spec, enabled=enabled)
        else:
            raise ValueError(f"Unknown group_spec type: {group_spec.type}")

    def clear_options(self):

        # Clear existing chips
        while (child := self.list_layout.takeAt(0)) is not None:
            if child.widget() is not None:
                child.widget().deleteLater()

        self.option: dict[str, FieldEditor | FieldChip] = {}

    def rebuild_static_options(self, group_spec: FieldGroupSpec, enabled: bool = True):

        if not isinstance(self.list_layout, QGridLayout):
            raise ValueError("Cannot rebuild static options on non-grid layout.")

        self.clear_options()

        for row, (name, spec) in enumerate(group_spec.fields.items()):
            self.add_editor(row, name, spec, enabled=enabled)

        self.list_layout.setColumnStretch(1, 1)

    def add_editor(self, row: int, name: str, spec: FieldSpec, enabled: bool = True):
        path_edit = FieldEditor(name, spec)
        # self.option[name] = QLineEdit(spec.path)
        path_edit.setMinimumWidth(30)
        path_edit.editingFinished.connect(
            lambda field_name=name: self.callback(
                self.group_name,
                field_name,
                method="edit_path",
                field_path=path_edit.field_path,
            )
        )

        browse = QToolButton()
        browse.setText("…")
        browse.setToolTip(f"Find {name.lower()} field")
        browse.setFixedWidth(25)
        browse.clicked.connect(
            lambda _, label=name: self.callback(
                self.group_name, label, method="edit_path"
            )
        )

        self.list_layout.addWidget(QLabel(name.capitalize() + ":"), row, 0)
        self.list_layout.addWidget(path_edit, row, 1)

        self.list_layout.addWidget(browse, row, 2)
        path_edit.setEnabled(enabled)
        self.option[name] = path_edit

    def rebuild_dynamic_options(self, group_spec: FieldGroupSpec, enabled: bool = True):

        if not isinstance(self.list_layout, FlowLayout):
            raise ValueError("Cannot rebuild dynamic options on non-flow layout.")

        self.clear_options()

        for name, spec in group_spec.fields.items():
            self.add_chip(name, spec, enabled)

        add_button = QToolButton()
        add_button.setText("+")
        add_button.setToolTip(f"Add {self.group_name} field")
        add_button.clicked.connect(
            lambda _, field_name=None: self.callback(
                self.group_name, field_name, method="add"
            )
        )
        self.list_layout.addWidget(add_button)

    def add_chip(self, name: str, spec: FieldSpec, enabled: bool = True):

        assert isinstance(
            self.list_layout, FlowLayout
        ), "Cannot add chip to non-flow layout."
        chip = FieldChip(name, spec)
        self.list_layout.insertBeforeLast(chip)

        chip.rename_requested.connect(
            lambda field_name: self.callback(
                self.group_name, field_name, method="rename"
            )
        )
        chip.remove_requested.connect(
            lambda field_name: self.callback(
                self.group_name, field_name, method="remove"
            )
        )
        chip.change_path_requested.connect(
            lambda field_name: self.callback(
                self.group_name, field_name, method="edit_path"
            )
        )
        chip.setEnabled(enabled)
        self.option[name] = chip

    def refresh(self, name: str, path: Optional[str] = None):

        if name not in self.group_spec.fields:
            self.option[name].deleteLater()
            del self.option[name]
            return

        self.option[name].set(name=name, path=path)

    def update_style(self, name, status):

        self.option[name].setStyleSheet(
            f"background-color: {COMPAT_COLORS[status]}; color: {TEXT_COLOR}; border: 1px solid {COMPAT_BORDERS[status]}; border-radius: 3px; padding: 2px;"
        )


class FieldSelector(QWidget):
    """
    Widget to manage field selection for different groups. Provides an interface to edit, add, rename, and remove fields.
    """

    fields_changed = Signal()
    loading_possible = bool

    def __init__(self, parent, source: SessionData | Assignments):
        super().__init__(parent)

        self.state: state.AppState = parent.state
        self.data: data.Data = parent.data

        self.field_options: dict[str, OptionList] = {}
        self.opts_layout = QVBoxLayout(self)
        self.opts_layout.setContentsMargins(6, 6, 6, 6)
        self.opts_layout.setSpacing(3)

        self.fields_changed.connect(self._on_fields_changed)

        self.update_source(source)
        # self.rebuild()
        # self.state.data_changed.connect(self._on_data_changed)

    def update_source(self, source: SessionData | Assignments | None):
        self.source = source
        self.rebuild()

    def rebuild(self):

        self.clear()

        if self.source is None or self.source.source_config is None:
            return
        ## loading options
        for group_name, group_spec in self.source.source_config.groups.items():
            self.field_options[group_name] = OptionList(
                group_name,
                group_spec,
                self.manipulate_fields,
                # enabled=not self.source.status.get(f"{group_name}_loaded", False)
            )

            opts_widget = checkbox_with_options(
                group_spec,
                self.field_options[group_name],
                self.source.source_config.groups[group_name].enabled,
                lambda checked, group_name=group_name: self.source.source_config.set_group_enabled(
                    group_name, checked
                ),
            )
            self.opts_layout.addWidget(opts_widget)

        self.fields_changed.emit()

    def clear(self):
        while (child := self.opts_layout.takeAt(0)) is not None:
            if child.widget() is not None:
                child.widget().deleteLater()

    def _on_fields_changed(self):

        if (
            self.source is None
            or not self.source.path
            or self.source.source_config is None
        ):
            return

        report = check_fields_compatibility(
            self.source.path,
            self.source.source_config.get_fields_to_load(
                list(self.source.source_config.groups.keys())
            ),
        )

        loading_possible = True
        for field in report.fields:
            if not field.available and field.spec.required:
                loading_possible &= False
                status = "required_missing"
            elif not field.available and not field.spec.required:
                status = "optional_missing"
            else:
                status = "available"

            self.field_options[field.group].update_style(field.label, status)

        # also, finally color current session properly!!
        self.source.status["loading_possible"] = loading_possible

    def manipulate_fields(
        self, group_name: str, field_name: str, method="edit", **kwargs
    ):
        assert self.source is not None, "Source is not set."
        assert self.source.path is not None, "Path is not set."
        assert self.source.source_config is not None, "Load config is not initialized."

        field_path = None

        if method == "remove":
            self.source.source_config.remove_field(
                group_name,
                field_name,
            )
            self.field_options[group_name].refresh(field_name)

            self.fields_changed.emit()
            return

        if method == "rename":
            new_name, ok = QInputDialog.getText(
                self,
                "Change field title",
                "Enter new title:",
                text=field_name,
            )
            if not ok or not new_name:
                return
            try:
                self.source.source_config.rename_field(
                    group_name,
                    field_name,
                    new_name=new_name,
                )
            except KeyError as e:
                self.state.issue(
                    "warning",
                    "Renaming field is not possible",
                    str(e),
                )
                return
            if new_name != field_name:
                self.field_options[group_name].option[new_name] = self.field_options[
                    group_name
                ].option.pop(field_name)

                ## set name in data source to new one
                if (
                    data_source := getattr(self.source, group_name)
                ) and field_name in data_source:
                    data_source[new_name] = data_source.pop(field_name)

                field_name = new_name

        if method == "edit_path":

            field_path = kwargs.get("field_path")

            if field_path is not None:

                self.source.source_config.update_field(
                    group_name, field_name, path=field_path
                )

            else:

                spec = self.source.source_config.groups[group_name].fields[field_name]
                selection = FieldSelectDialog.get_field(
                    path=self.source.path, key=field_name, parent=self, spec=spec
                )

                if selection is None:
                    return

                field_path = selection.path
                self.source.source_config.update_field(
                    group_name,
                    field_name,
                    path=selection.path,
                    source=selection.source,
                    attribute=selection.attribute,
                    source_path=selection.source_path,
                )

        if method == "add":
            selection = FieldSelectDialog.get_field(
                path=self.source.path, key=field_name, parent=self
            )

            if selection is None:
                return

            field_path = selection.path

            ## check, if a field with the same path already exists in the group
            fields_to_load = self.source.source_config.get_fields_to_load([group_name])

            duplicate = any(
                field.path == selection.path
                and field.source_path == selection.source_path
                and field.source == selection.source
                and field.attribute == selection.attribute
                for field in fields_to_load[group_name].values()
            )

            if duplicate:

                self.state.issue(
                    "warning",
                    "Adding field is not possible",
                    (
                        f"Field {selection.path} from this source already "
                        f"exists in {group_name}."
                    ),
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

            try:
                self.source.source_config.add_field(
                    group_name,
                    field_name,
                    path=selection.path,
                    source=selection.source,
                    attribute=selection.attribute,
                    source_path=selection.source_path,
                )
            except KeyError as e:
                self.state.issue(
                    "warning",
                    "Adding field is not possible",
                    f"Field with name {field_name} already exists in load options for {group_name}.",
                )
                return
            self.field_options[group_name].add_chip(
                field_name,
                self.source.source_config.groups[group_name].fields[field_name],
                # enabled=not self.source.status.get(f"{group_name}_loaded", False)
            )

        field_path = (
            field_path
            or self.source.source_config.groups[group_name].fields[field_name].path
        )
        self.field_options[group_name].refresh(field_name, field_path)
        self.fields_changed.emit()


def checkbox_with_options(
    group_spec: FieldGroupSpec, opts_ref: QWidget, active: bool, callback: Callable
) -> QWidget:
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

    header_layout.addWidget(chk)
    header_layout.addStretch()

    layout.addWidget(header)

    # Indented child area
    options_container = QWidget()
    options_layout = QHBoxLayout(options_container)
    options_layout.setContentsMargins(14, 0, 0, 4)
    options_layout.addWidget(opts_ref)

    layout.addWidget(options_container)

    return container
