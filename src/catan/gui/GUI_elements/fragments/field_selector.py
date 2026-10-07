from typing import Callable, Optional
from pathlib import Path, PurePosixPath

from PySide6.QtCore import Qt, Signal, QTimer
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
    QComboBox,
    QSpinBox,
)

from catan.core.io.inspection import check_fields_compatibility
from catan.core.structures.load_config import FieldGroupSpec, FieldSpec
from catan.tracking.structures import Assignments
from catan.gui.structures import SessionData, data, state
from catan.gui.GUI_elements.utils.FlowLayout import FlowLayout, QSizePolicy

from .field_chip import FieldChip
from .dialog_load_field import FieldSelectDialog
from .file_inspection import FileInspection

COMPAT_COLORS = {
    "available": "#4F7F5A",  # muted green
    "optional_missing": "#8A7040",  # muted amber
    "required_missing": "#8A4F52",  # muted red
    "unknown": "#555B66",
}
# green  = "#3F6548"
# amber  = "#6F5B35"
# red    = "#6F4144"
COMPAT_BORDERS = {
    "available": "#6A9A74",
    "optional_missing": "#A88A52",
    "required_missing": "#A8676B",
    "unknown": "#89909C",
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

        visible_fields = (
            (name, spec) for name, spec in group_spec.fields.items() if spec.exposed
        )

        for row, (name, spec) in enumerate(visible_fields):
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
            if spec.exposed:
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

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        self._build_dimension_controls(layout)
        self._inspection = FileInspection(self)
        layout.addWidget(self._inspection)

        self.opts_layout = QVBoxLayout()
        self.opts_layout.setContentsMargins(6, 6, 6, 6)
        self.opts_layout.setSpacing(3)
        layout.addLayout(self.opts_layout)

        self._inspection.result.connect(self._apply_compatibility)
        self._inspection.failed.connect(self._set_compatibility_unknown)
        self._inspection.cancelled.connect(self._set_compatibility_unknown)
        self._inspection.busy_changed.connect(self._on_compatibility_busy)

        self._check_timer = QTimer(self)
        self._check_timer.setSingleShot(True)
        self._check_timer.setInterval(250)
        self._check_timer.timeout.connect(self._start_compatibility_check)

        self.fields_changed.connect(self._on_fields_changed)

        self._compatibility_dirty = True
        self.update_source(source)
        # self.rebuild()
        # self.state.data_changed.connect(self._on_data_changed)

    def update_source(self, source: SessionData | Assignments | None):
        self.source = source
        self.rebuild()

    def rebuild(self):

        self.clear()

        self._sync_dimension_controls()

        if self.source is None or self.source.source_config is None:
            return
        ## loading options
        for group_name, group_spec in self.source.source_config.groups.items():
            if not group_spec.exposed:
                continue

            # Avoid an empty static group when all its fields are hidden.
            # Dynamic groups can remain empty because users may add fields.
            if group_spec.type == "static" and not any(
                spec.exposed for spec in group_spec.fields.values()
            ):
                continue

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
                lambda checked, group_name=group_name: self._set_group_enabled(
                    group_name, checked
                ),
            )
            self.opts_layout.addWidget(opts_widget)

        # Refresh compatibility indicators without announcing a configuration edit.
        self._on_fields_changed()

    def clear(self):
        self._check_timer.stop()
        self._inspection.invalidate()

        while (child := self.opts_layout.takeAt(0)) is not None:
            if child.widget() is not None:
                child.widget().deleteLater()
        self.field_options.clear()

    def _build_dimension_controls(self, layout):
        self.dimension_box = QWidget(self)
        box = QVBoxLayout(self.dimension_box)
        box.setContentsMargins(6, 6, 6, 6)

        box.addWidget(QLabel("Image dimensions (height × width)"))

        self.dimension_mode = QComboBox()
        for title, mode in (
            ("Manual", "manual"),
            ("Read field values", "values"),
            ("Use image shape", "image"),
        ):
            self.dimension_mode.addItem(title, mode)

        box.addWidget(self.dimension_mode)

        row = QHBoxLayout()
        self.dimension_height = QSpinBox()
        self.dimension_width = QSpinBox()

        for label, spin in (
            ("Height", self.dimension_height),
            ("Width", self.dimension_width),
        ):
            spin.setRange(1, 1000000)
            row.addWidget(QLabel(label))
            row.addWidget(spin)

        box.addLayout(row)

        self.dimension_source = QLabel()
        self.dimension_source.setWordWrap(True)
        box.addWidget(self.dimension_source)

        self.dimension_browse = QToolButton()
        self.dimension_browse.setText("Browse dimension source…")
        box.addWidget(self.dimension_browse)

        self.dimension_status = QLabel()
        self.dimension_status.setWordWrap(True)
        box.addWidget(self.dimension_status)

        layout.addWidget(self.dimension_box)

        self._syncing_dimensions = False

        self.dimension_mode.currentIndexChanged.connect(self._dimensions_changed)
        self.dimension_height.valueChanged.connect(self._dimensions_changed)
        self.dimension_width.valueChanged.connect(self._dimensions_changed)
        self.dimension_browse.clicked.connect(self._browse_dimensions)

    def _sync_dimension_controls(self):
        source = self.source
        visible = isinstance(source, SessionData) and source.source_config is not None
        self.dimension_box.setVisible(visible)

        if not visible:
            return

        spec = source.source_config.dimensions
        self._syncing_dimensions = True

        try:
            mode = spec.get("mode", "manual")
            self.dimension_mode.setCurrentIndex(self.dimension_mode.findData(mode))
            self.dimension_height.setValue(spec.get("height", 512))
            self.dimension_width.setValue(spec.get("width", 512))

            manual = mode == "manual"
            self.dimension_height.setEnabled(manual)
            self.dimension_width.setEnabled(manual)
            self.dimension_browse.setEnabled(not manual)

            field = spec.get("field")
            self.dimension_source.setText(
                (f"{field.get('source_path') or source.path}: " f"{field['path']}")
                if field
                else (
                    "No source selected; image mode uses " "the configured background."
                )
            )
            self.dimension_source.setVisible(not manual)

        finally:
            self._syncing_dimensions = False

    def _dimensions_changed(self, *_):
        if self._syncing_dimensions or not isinstance(self.source, SessionData):
            return

        config = self.source.source_config
        if config is None:
            return

        config.dimensions.update(
            mode=self.dimension_mode.currentData(),
            height=self.dimension_height.value(),
            width=self.dimension_width.value(),
        )

        self._sync_dimension_controls()
        self.fields_changed.emit()

    def _browse_dimensions(self):
        source = self.source
        config = source.source_config
        raw = config.dimensions.get("field")

        selection = FieldSelectDialog.get_field(
            path=source.path,
            key="dimensions",
            parent=self,
            spec=FieldSpec.from_dict(raw) if raw else None,
        )

        if (
            selection is None
            or self.source is not source
            or source.source_config is not config
        ):
            return

        config.dimensions["field"] = FieldSpec(
            path=selection.path,
            source=selection.source,
            attribute=selection.attribute,
            source_path=selection.source_path,
            required=True,
        ).to_dict()

        self._sync_dimension_controls()
        self.fields_changed.emit()

    def _dimension_parameters(self):
        source = self.source

        orientation = getattr(source, "_recipe_options", {}).get(
            "background_orientation",
            getattr(self.data, "background_orientation", "auto"),
        )

        expected = next(
            (
                tuple(other.dims)
                for other in self.data.sessions
                if other is not source and other.status["spatial_loaded"]
            ),
            None,
        )

        return {
            "dimensions": source.source_config.dimensions,
            "orientation": orientation,
            "expected_dims": expected,
        }

    def _show_dimension_status(self, ok, message):
        key = "unknown" if ok is None else "available" if ok else "required_missing"

        self.dimension_status.setText(message)

        for widget in (
            self.dimension_height,
            self.dimension_width,
            self.dimension_source,
        ):
            widget.setStyleSheet(
                f"background-color: {COMPAT_COLORS[key]}; " f"color: {TEXT_COLOR};"
            )
            widget.setToolTip(message)

    def _on_fields_changed(self):
        self._check_timer.stop()
        self._inspection.invalidate()
        self._set_compatibility_unknown()

        if self._defer_compatibility_check():
            return

        if (
            self.source is None
            or not self.source.path
            or self.source.source_config is None
        ):
            return

        self._check_timer.start()

    def _on_compatibility_busy(self, busy):
        if busy:
            self._set_compatibility_unknown()

    def _set_compatibility_unknown(self):
        self._compatibility_dirty = True
        self._show_dimension_status(
            None,
            "Dimensions have not been verified.",
        )

        if self.source is not None:
            self.source.status["loading_possible"] = False

        for options in self.field_options.values():
            for name, widget in options.option.items():
                options.update_style(name, "unknown")
                widget.setToolTip("Source compatibility has not been verified.")

    def _start_compatibility_check(self):
        if (
            self.source is None
            or not self.source.path
            or self.source.source_config is None
        ):
            return

        config = self.source.source_config

        fields = config.get_fields_to_load(list(config.groups.keys()))
        spatial = isinstance(self.source, SessionData) and "footprints" in fields.get(
            "spatial", {}
        )

        self._inspection.start(
            "session_compatibility" if spatial else "compatibility",
            self.source.path,
            fields_to_load=fields,
            **(self._dimension_parameters() if spatial else {}),
        )

    def _apply_compatibility(self, report):
        if self.source is None:
            return

        geometry = report.get("geometry") if isinstance(report, dict) else None

        if geometry is not None:
            report = report["fields"]
            self._show_dimension_status(
                geometry["ok"],
                geometry["message"],
            )

        loading_possible = geometry is None or geometry["ok"]

        for field in report.fields:
            if field.available:
                status = "available"
            elif field.spec.required:
                status = "required_missing"
                loading_possible = False
            else:
                status = "optional_missing"

            options = self.field_options.get(field.group)
            if options is None or field.label not in options.option:
                continue

            options.update_style(field.label, status)
            options.option[field.label].setToolTip(field.reason or field.spec.path)

        self.source.status["loading_possible"] = loading_possible
        self._compatibility_dirty = False

    def _set_group_enabled(self, group_name: str, enabled: bool):
        if self.source is None or self.source.source_config is None:
            return

        config = self.source.source_config
        enabled = bool(enabled)

        if config.groups[group_name].enabled == enabled:
            return

        config.set_group_enabled(group_name, enabled)
        self.fields_changed.emit()

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
            if not ok or not new_name or new_name == field_name:
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
            spec = self.source.source_config.groups[group_name].fields[field_name]
            field_path = kwargs.get("field_path")

            if field_path is not None:
                if field_path == spec.path:
                    return

                self.source.source_config.update_field(
                    group_name,
                    field_name,
                    path=field_path,
                )

            else:
                selection = FieldSelectDialog.get_field(
                    path=self.source.path,
                    key=field_name,
                    parent=self,
                    spec=spec,
                )

                if selection is None:
                    return

                previous = (
                    spec.path,
                    spec.source,
                    spec.attribute,
                    spec.source_path,
                )
                selected = (
                    selection.path,
                    selection.source,
                    selection.attribute,
                    selection.source_path,
                )

                if selected == previous:
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

    def _defer_compatibility_check(self):
        return (
            isinstance(self.source, SessionData)
            and getattr(self.source, "_restored_from_catan", False)
            and not self.isVisible()
        )

    def showEvent(self, event):
        super().showEvent(event)

        if self._compatibility_dirty:
            self._on_fields_changed()


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
