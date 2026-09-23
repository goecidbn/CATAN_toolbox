from dataclasses import dataclass, replace
from itertools import combinations, islice
from typing import Literal
import numpy as np

from PySide6.QtCore import Qt, QEvent, Signal, QPoint, QTimer
from PySide6.QtGui import QColor, QBrush
from PySide6.QtWidgets import (
    QMenu,
    QWidget,
    QLabel,
    QDialog,
    QPushButton,
    QHBoxLayout,
    QVBoxLayout,
    QTableWidget,
    QTableWidgetItem,
    QAbstractItemView,
    QComboBox,
    QHeaderView,
)

from catan.core.structures import NeuronComponent
from catan.gui.structures import AppState, Data
from catan.gui.panels import BasePlot, styles, StatisticsData
from catan.gui.data.statistics import PickTable
from catan.gui.data.statistics.queries import (
    StatisticQuery,
    ReductionSpec,
    neuron_pair_bound_dims,
    component_pair_bound_dims,
)
from catan.gui.data.statistics.tabledata import (
    prepare_neuron_table_query,
    prepare_component_table_query,
    prepare_neuron_pair_table_query,
    prepare_component_pair_table_query,
)
from catan.gui.data.statistics.dimensions import (
    neuron_bound_dim,
    component_bound_dims,
)
from catan.gui.utils.popups import constrain_popup

MAX_PAIR_ROWS = 200
MAX_STATISTIC_HEADER_LENGTH = 24

EntityMode = Literal["footprint", "neuron"]
RowMode = Literal["single", "pair"]


TableEntity = NeuronComponent


@dataclass(frozen=True, slots=True)
class SelectionTableBinding:
    entity_mode: EntityMode
    row_mode: RowMode

    @property
    def arity(self) -> int:
        if self.row_mode == "single":
            return 1
        if self.row_mode == "pair":
            return 2
        raise ValueError(f"Unknown row mode: {self.row_mode!r}")


@dataclass(frozen=True, slots=True)
class SelectionTableRow:
    entities: tuple[NeuronComponent, ...]


@dataclass(frozen=True, slots=True)
class StatisticColumn:
    raw_query: StatisticQuery


@dataclass(slots=True)
class StatisticColumnResult:
    column: StatisticColumn
    query: StatisticQuery
    table: PickTable | None

    neuron_dim: str | None
    session_dim: str | None

    # semantic entity -> PickTable row
    row_lookup: dict
    # Direct values for SelectionDisplay rows.
    # Used for pair mode so we do not construct a global pair table.
    row_values: np.ndarray | None = None

    error: str | None = None
    fallback_title: str | None = None

    display_title: str | None = None

    @property
    def title(self) -> str:

        if self.display_title is not None:
            return self.display_title

        if self.fallback_title is not None:
            return self.fallback_title

        return self.query.statistic_key

    def value_for_component(self, component: NeuronComponent | None):
        if component is None:
            return np.nan
        if self.table is None:
            return np.nan

        if self.session_dim is None:
            key = int(component.neuron_id)
        else:
            if component.session_id is None:
                return np.nan
            key = (int(component.session_id), int(component.neuron_id))

        row = self.row_lookup.get(key)

        if row is None:
            return np.nan

        return self.table.values[row]

    def value_for_row(
        self,
        row_index: int,
        components: tuple[NeuronComponent, ...],
    ):
        if self.row_values is not None:
            if not 0 <= row_index < len(self.row_values):
                return np.nan

            return self.row_values[row_index]

        return self.value_for_components(components)

    def value_for_components(
        self,
        components: tuple[NeuronComponent, ...],
    ):

        if len(components) == 1:
            return self.value_for_component(components[0])

        if len(components) != 2:
            return None
        if self.table is None:
            return np.nan

        component_a, component_b = components

        dims = set(self.table.dims)

        # Shared session dimension:
        # both components must belong to that same session.
        if "session" in dims:

            if (
                component_a.session_id is None
                or component_b.session_id is None
                or component_a.session_id != component_b.session_id
            ):
                return None

            keys = (_canonical_component_pair(component_a, component_b),)

        else:
            keep_session_i = "session_i" in dims
            keep_session_j = "session_j" in dims

            def masked_component(component, keep_session):
                return NeuronComponent(
                    neuron_id=component.neuron_id,
                    session_id=(component.session_id if keep_session else None),
                )

            # Try both assignments of A/B to i/j.
            keys = (
                _canonical_component_pair(
                    masked_component(component_a, keep_session_i),
                    masked_component(component_b, keep_session_j),
                ),
                _canonical_component_pair(
                    masked_component(component_b, keep_session_i),
                    masked_component(component_a, keep_session_j),
                ),
            )

        for key in keys:
            row = self.row_lookup.get(key)

            if row is not None:
                return self.table.values[row]

        return None


def _canonical_component_pair(
    component_a: NeuronComponent,
    component_b: NeuronComponent,
):
    return frozenset((component_a.id, component_b.id))


class Display(QWidget):

    add_statistic_requested = Signal(object)

    statistic_header_context_requested = Signal(int, object)
    statistic_columns_reordered = Signal(object)

    def __init__(self, parent, controls, config=None):
        super().__init__(parent)

        self.state: AppState = parent.state
        self.data: Data = parent.data
        # self.tracking = parent.tracking

        self.controls = controls

        self.entity_mode: EntityMode = "footprint"
        self.row_mode: RowMode = "single"

        menu_layout = QVBoxLayout(self)
        menu_layout.setContentsMargins(0, 0, 0, 0)

        self.pair_warning_label = QLabel()
        self.pair_warning_label.setWordWrap(True)
        self.pair_warning_label.setVisible(False)

        self.pair_warning_label.setStyleSheet("""
            QLabel {
                color: #5f4b00;
                background-color: #fff3cd;
                border: 1px solid #e0c36e;
                border-radius: 4px;
                padding: 5px 8px;
            }
        """)

        menu_layout.addWidget(self.pair_warning_label)

        # Create statistics table
        self.stats_table = QTableWidget()

        self.styles = styles.Styles()

        self.rows: list[SelectionTableRow] = []
        self.statistic_columns: list[StatisticColumnResult] = []

        self._table_hover_components = None

        # We handle semantic states ourselves.
        self.stats_table.setSelectionMode(QAbstractItemView.SelectionMode.NoSelection)
        self.stats_table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)

        self.stats_table.setMouseTracking(True)
        self.stats_table.viewport().setMouseTracking(True)

        self.stats_table.cellEntered.connect(self._on_cell_entered)
        self.stats_table.cellClicked.connect(self._on_cell_clicked)
        self.stats_table.cellDoubleClicked.connect(self._on_cell_double_clicked)
        self.stats_table.viewport().installEventFilter(self)

        self.stats_table.setColumnCount(0)
        self.stats_table.setRowCount(0)

        self.stats_table.resizeColumnsToContents()
        menu_layout.addWidget(self.stats_table)

        self.state.data_changed.connect(self._on_data_changed)

        header = self.stats_table.horizontalHeader()
        header.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
        header.customContextMenuRequested.connect(self._on_header_context_menu)

        header.setSectionsMovable(True)
        header.sectionMoved.connect(self._on_header_section_moved)

        self._moving_header = False

        header.sectionClicked.connect(self._on_header_clicked)

    @property
    def add_column_index(self) -> int:
        return self.stats_table.columnCount() - 1

    def _on_data_changed(self):
        self.update_display()

    def _show_pair_warning(
        self,
        n_entities: int,
        n_pairs: int,
    ):
        self.pair_warning_label.setText(
            f"{n_entities} selected entities create {n_pairs} pairs. "
            f"Showing the first {MAX_PAIR_ROWS}."
        )
        self.pair_warning_label.setVisible(True)

    def _clear_pair_warning(self):
        self.pair_warning_label.clear()
        self.pair_warning_label.setVisible(False)

    # ----- configuration supplied by Controller -----

    def set_mode(
        self,
        *,
        entity_mode: EntityMode,
        row_mode: RowMode,
    ):

        if entity_mode not in ("footprint", "neuron"):
            raise ValueError(f"Unknown entity mode: {entity_mode!r}")

        if row_mode not in ("single", "pair"):
            raise ValueError(f"Unknown row mode: {row_mode!r}")

        changed = entity_mode != self.entity_mode or row_mode != self.row_mode

        self.entity_mode = entity_mode
        self.row_mode = row_mode

        if changed:
            self.update_display()

    # ----- row model -----

    def _selected_entities(self) -> list[NeuronComponent]:

        components = self.state.selected_components

        if not components:
            return []

        if self.entity_mode == "footprint":
            return list(dict.fromkeys(components))

        if self.entity_mode == "neuron":
            neuron_ids = dict.fromkeys(
                int(component.neuron_id) for component in components
            )

            return [
                NeuronComponent(neuron_id=neuron_id, session_id=None)
                for neuron_id in neuron_ids
            ]

        raise ValueError(self.entity_mode)

    def _build_rows(
        self,
    ) -> list[SelectionTableRow]:

        entities = self._selected_entities()

        if self.row_mode == "single":
            self._clear_pair_warning()

            return [SelectionTableRow((entity,)) for entity in entities]

        if self.row_mode == "pair":

            n = len(entities)
            n_pairs = n * (n - 1) // 2

            pairs = islice(
                combinations(entities, 2),
                MAX_PAIR_ROWS,
            )

            if n_pairs > MAX_PAIR_ROWS:
                self._show_pair_warning(
                    n_entities=n,
                    n_pairs=n_pairs,
                )
            else:
                self._clear_pair_warning()

            return [SelectionTableRow(pair) for pair in pairs]

    def row_data(
        self,
        row: int,
    ) -> SelectionTableRow | None:

        if not 0 <= row < len(self.rows):
            return None

        return self.rows[row]

    def component_for_entity(
        self,
        entity: NeuronComponent,
    ) -> NeuronComponent | None:

        # Already a concrete session-specific entity.
        if entity.session_id is not None:
            return entity

        # Session-independent neuron entity.
        if self.state.current_session_id is None:
            return None

        return NeuronComponent(
            neuron_id=entity.neuron_id,
            session_id=self.state.current_session_id,
        )

    def footprint_for_entity(
        self,
        entity: TableEntity,
    ) -> int | None:

        component = self.component_for_entity(entity)

        if component is None:
            return None

        return self.state.get_footprint_from_component(component)

    def entity_for_cell(
        self,
        row: int,
        column: int,
    ) -> TableEntity | None:

        row_data = self.row_data(row)

        if row_data is None:
            return None

        # '+' pseudo-column is not associated with an entity.
        if column == self.add_column_index:
            return None

        # In single mode, every cell belongs to the same entity,
        # including statistic cells.
        if self.row_mode == "single":
            return row_data.entities[0]

        # ---------------------------------------------------------
        # Pair mode
        # ---------------------------------------------------------

        if self.entity_mode == "neuron":

            if column == 0:
                return row_data.entities[0]

            if column == 1:
                return row_data.entities[1]

            return None

        if self.entity_mode == "footprint":

            # First two columns describe entity A.
            if column in (0, 1):
                return row_data.entities[0]

            # Next two columns describe entity B.
            if column in (2, 3):
                return row_data.entities[1]

            # Future pair-statistic columns refer to the pair,
            # not uniquely to A or B.
            return None

        raise ValueError(f"Unknown entity mode: {self.entity_mode!r}")

    # ----- rendering -----

    def update_display(self):

        self.rows = self._build_rows()

        headers = [
            *self._identity_headers(),
            *[
                self._statistic_header_title(column)
                for column in self.statistic_columns
            ],
            "+",
        ]

        self.stats_table.setColumnCount(len(headers))
        self.stats_table.setHorizontalHeaderLabels(headers)
        self.stats_table.clearContents()

        identity_columns = len(self._identity_headers())

        for i, column in enumerate(self.statistic_columns):
            header_item = self.stats_table.horizontalHeaderItem(identity_columns + i)

            if header_item is not None:
                tooltip = column.title

                if column.error is not None:
                    tooltip += (
                        "\n\nStatistic could not be calculated:\n" f"{column.error}"
                    )

                header_item.setToolTip(tooltip)

        self.stats_table.setRowCount(len(self.rows))

        for row_index, row_data in enumerate(self.rows):

            next_column = self._render_identity_cells(row_index, row_data)

            self._render_statistic_cells(
                row_index,
                row_data,
                start_column=next_column,
            )

        self.stats_table.resizeColumnsToContents()
        self._configure_add_column()
        self.update_styles()

    def _configure_add_column(self):
        """
        Configure the permanent '+' pseudo-column.

        This column is not data. It always stays at the far right and opens
        the statistic-column chooser when its header is clicked.
        """
        if self.stats_table.columnCount() == 0:
            return

        column = self.add_column_index

        header = self.stats_table.horizontalHeader()

        header.setSectionResizeMode(column, QHeaderView.ResizeMode.Fixed)

        self.stats_table.setColumnWidth(column, 28)

    def _render_identity_cells(
        self,
        row_index: int,
        row_data: SelectionTableRow,
    ) -> int:
        """
        Render the identity columns of one semantic table row.

        Returns the index of the first column after the identity columns.
        """

        column = 0

        if self.entity_mode == "footprint":

            for entity in row_data.entities:

                component = self.component_for_entity(entity)

                if component is None:
                    session_id = None
                    neuron_id = entity.neuron_id
                    footprint_id = None

                else:
                    session_id = component.session_id
                    neuron_id = component.neuron_id

                    footprint_id = self.state.get_footprint_from_component(component)

                component_text = (
                    f"{session_id},{neuron_id}"
                    if session_id is not None
                    else f"—,{neuron_id}"
                )

                self.stats_table.setItem(
                    row_index,
                    column,
                    QTableWidgetItem(component_text),
                )
                column += 1

                if self.row_mode == "single":

                    footprint_text = (
                        str(footprint_id)
                        if footprint_id is not None and footprint_id >= 0
                        else "--"
                    )

                    self.stats_table.setItem(
                        row_index,
                        column,
                        QTableWidgetItem(footprint_text),
                    )
                    column += 1

            return column

        if self.entity_mode == "neuron":

            for entity in row_data.entities:

                self.stats_table.setItem(
                    row_index,
                    column,
                    QTableWidgetItem(str(entity.neuron_id)),
                )
                column += 1

            return column

        raise ValueError(f"Unknown entity mode: {self.entity_mode!r}")

    def _identity_headers(self) -> list[str]:

        if self.entity_mode == "footprint":

            if self.row_mode == "single":
                return ["(s,n)", "footprint"]

            return [
                "(s,n) A",
                # "footprint A",
                "(s,n) B",
                # "footprint B",
            ]

        if self.entity_mode == "neuron":

            if self.row_mode == "single":
                return ["neuron"]

            return [
                "neuron A",
                "neuron B",
            ]

        raise ValueError(self.entity_mode)

    def _render_statistic_cells(
        self,
        row_index: int,
        row_data: SelectionTableRow,
        *,
        start_column: int,
    ):

        components = tuple(
            self.component_for_entity(entity) for entity in row_data.entities
        )

        for i, result in enumerate(self.statistic_columns):

            if result.row_values is not None:
                value = result.value_for_row(row_index, components)

            elif any(component is None for component in components):
                value = None

            else:
                value = result.value_for_components(components)

            text = (
                f"{float(value):.4g}"
                if value is not None and np.isfinite(value)
                else "--"
            )

            self.stats_table.setItem(
                row_index,
                start_column + i,
                QTableWidgetItem(text),
            )

    # ----- input interaction -----
    def components_for_row(
        self,
        row: int,
    ) -> list[NeuronComponent]:

        row_data = self.row_data(row)

        if row_data is None:
            return []

        return [
            component
            for entity in row_data.entities
            if (component := self.component_for_entity(entity)) is not None
        ]

    def _on_cell_entered(self, row: int, column: int):

        components = self.components_for_row(row)

        self._table_hover_components = components

        self.state.update_hovered_components(components or None)

    def _on_cell_clicked(self, row: int, column: int):

        components = self.components_for_row(row)
        self.state.update_highlighted_components(components or None)

    def _on_cell_double_clicked(self, row: int, column: int):
        entity = self.entity_for_cell(row, column)

        if entity is None:
            return
        component = self.component_for_entity(entity)
        self.state.focused_component = component

        # A Qt double-click also emits a normal click first.
        # Avoid ending with the same component both highlighted
        # and focused.
        if component in (self.state.highlighted_components or []):
            self.state.update_highlighted_components(None)

    def eventFilter(
        self,
        watched,
        event,
    ):

        if watched is self.stats_table.viewport() and event.type() == QEvent.Type.Leave:

            if self.state.hovered_components == self._table_hover_components:
                self.state.update_hovered_components(None)

            self._table_hover_components = None

        return super().eventFilter(
            watched,
            event,
        )

    def _row_matches_components(
        self,
        row_data: SelectionTableRow,
        components,
    ) -> bool:

        if not components:
            return False

        row_components = [
            self.component_for_entity(entity) for entity in row_data.entities
        ]

        row_components = [
            component for component in row_components if component is not None
        ]

        # Pair rows only react to a complete pair.
        if self.row_mode == "pair":

            if len(components) != 2:
                return False

            return all(
                any(
                    self.state._components_match_selection(
                        row_component,
                        component,
                    )
                    for component in components
                )
                for row_component in row_components
            )

        # Single rows require the one row entity to
        # occur somewhere in the interaction state.
        return any(
            self.state._components_match_selection(row_components[0], component)
            for component in components
        )

    # ----- Header interaction -----

    def _on_header_clicked(
        self,
        section: int,
    ):

        if section != self.add_column_index:
            return

        header = self.stats_table.horizontalHeader()

        x = header.sectionViewportPosition(section)
        y = header.height()

        global_pos = header.viewport().mapToGlobal(QPoint(x, y))

        self.add_statistic_requested.emit(global_pos)

    def set_statistic_columns(
        self,
        columns: list[StatisticColumnResult],
    ):
        self.statistic_columns = list(columns)
        self.update_display()

    def _on_header_context_menu(self, pos):

        header = self.stats_table.horizontalHeader()
        column = header.logicalIndexAt(pos)

        first_stat = len(self._identity_headers())
        statistic_index = column - first_stat

        if not (0 <= statistic_index < len(self.statistic_columns)):
            return

        self.statistic_header_context_requested.emit(
            statistic_index,
            header.mapToGlobal(pos),
        )

    def _on_header_section_moved(
        self,
        logical_index,
        old_visual_index,
        new_visual_index,
    ):

        if self._moving_header:
            return

        header = self.stats_table.horizontalHeader()

        first_stat = len(self._identity_headers())
        last_stat = first_stat + len(self.statistic_columns) - 1

        valid = (
            first_stat <= logical_index <= last_stat
            and first_stat <= new_visual_index <= last_stat
        )

        if not valid:
            self._moving_header = True
            try:
                header.moveSection(new_visual_index, old_visual_index)
            finally:
                self._moving_header = False
            return

        logical_columns = range(first_stat, last_stat + 1)

        ordered = sorted(logical_columns, key=header.visualIndex)

        order = [column - first_stat for column in ordered]

        self.statistic_columns_reordered.emit(order)

        # Controller rebuilds the table using the new logical order.
        # Reset QHeaderView's temporary visual ordering afterwards.
        QTimer.singleShot(0, self._reset_header_visual_order)

    def _reset_header_visual_order(self):

        header = self.stats_table.horizontalHeader()

        self._moving_header = True
        try:
            for logical in range(header.count()):
                visual = header.visualIndex(logical)

                if visual != logical:
                    header.moveSection(visual, logical)
        finally:
            self._moving_header = False

    def _statistic_header_title(
        self,
        column: StatisticColumnResult,
    ) -> str:

        full_title = column.title

        if len(full_title) <= MAX_STATISTIC_HEADER_LENGTH:
            title = full_title
        elif column.table is not None:
            title = f"{column.table.stat.title}*"
        else:
            title = f"{full_title}*"

        if column.error is not None:
            title += " ⚠"

        return title

    # ----- passive state display -----

    def update_styles(self):

        hovered = self.state.hovered_components
        focused = self.state.focused_component
        highlighted = self.state.highlighted_components

        for row, row_data in enumerate(self.rows):

            style = None

            # Focus is still singular.
            if self._row_contains_component(row_data, focused):
                style = "focused"

            if self._row_matches_components(row_data, highlighted):
                style = "highlighted"

            if self._row_matches_components(row_data, hovered):
                style = "hovered"

            brush = self._row_brush(style) if style is not None else QBrush()

            for column in range(self.stats_table.columnCount()):
                item = self.stats_table.item(row, column)

                if item is not None:
                    item.setBackground(brush)

    def _row_contains_component(
        self,
        row_data: SelectionTableRow,
        component: NeuronComponent | None,
    ) -> bool:

        if component is None:
            return False

        if self.entity_mode == "neuron":

            return any(
                entity.neuron_id == component.neuron_id for entity in row_data.entities
            )

        if self.entity_mode == "footprint":

            return any(
                self.component_for_entity(entity) == component
                for entity in row_data.entities
            )

        raise ValueError(f"Unknown entity mode: {self.entity_mode!r}")

    def _row_brush(
        self,
        style: str,
    ) -> QBrush:

        alpha = {
            "hovered": 0.20,
            "focused": 0.24,
            "highlighted": 0.24,
        }[style]

        rgba = self.styles.get_color_array(
            style,
            values=0.7,
            alpha=alpha,
        )

        rgba = np.asarray(
            rgba,
            dtype=float,
        ).reshape(-1)

        color = QColor.fromRgbF(
            float(rgba[0]),
            float(rgba[1]),
            float(rgba[2]),
            float(rgba[3]),
        )

        return QBrush(color)


class Controller(BasePlot.TableController):

    def __init__(self, display_section, config=None):
        super().__init__(display_section, config)

        self.entity_mode: EntityMode = "footprint"
        self.row_mode: RowMode = "single"

        self.statistic_config = self.data.statistic_display_config

        self.statistic_columns = [
            StatisticColumn(raw_query=query)
            for query in self.statistic_config.queries(
                self.entity_mode,
                self.row_mode,
            )
        ]
        self.statistic_config.changed.connect(self._on_statistic_display_config_changed)

        self.statistic_results: list[StatisticColumnResult] = []
        self._statistic_rebuild_generation = 0

        self.data.statistic_engine.registry_changed.connect(
            self._on_statistics_registry_changed
        )

    @property
    def table_binding(self) -> SelectionTableBinding:
        return SelectionTableBinding(
            entity_mode=self.entity_mode,
            row_mode=self.row_mode,
        )

    def _table_mode_key(self):
        return self.entity_mode, self.row_mode

    # ----- controls -----

    def build_controls(self):
        super().build_controls()

        self.entity_selector = QComboBox()
        self.entity_selector.addItem("Footprints", "footprint")
        self.entity_selector.addItem("Neurons", "neuron")

        self.row_selector = QComboBox()
        self.row_selector.addItem("Single", "single")
        self.row_selector.addItem("Pairs", "pair")

        self.section.x_options_layout.addWidget(QLabel("Entity:"))
        self.section.x_options_layout.addWidget(self.entity_selector)
        self.section.x_options_layout.addWidget(QLabel("Rows:"))
        self.section.x_options_layout.addWidget(self.row_selector)
        self.section.x_options_layout.addStretch()

        self.entity_selector.currentIndexChanged.connect(self._on_table_mode_changed)
        self.row_selector.currentIndexChanged.connect(self._on_table_mode_changed)

        # Synchronize initial controller -> display mode.
        self.entity_mode = self.entity_selector.currentData()
        self.row_mode = self.row_selector.currentData()

        self.table.set_mode(entity_mode=self.entity_mode, row_mode=self.row_mode)

        self.table.statistic_header_context_requested.connect(
            self._on_statistic_header_context_requested
        )

        self.table.statistic_columns_reordered.connect(
            self._on_statistic_columns_reordered
        )

    def configure_display(self):

        super().configure_display()

        self.table.add_statistic_requested.connect(self._open_add_statistic_dialog)

    def _open_add_statistic_dialog(self, _global_pos):

        popup = AddStatisticPopup(
            engine=self.data.statistic_engine,
            popup_bounds=self.table.window(),
            parent=self.table,
        )

        self._configure_statistic_selector(popup.selector)

        popup.statisticAccepted.connect(self._add_statistic_column)

        self._statistic_dialog = popup

        popup.adjustSize()

        parent_window = self.table.window()

        center = parent_window.frameGeometry().center()

        geometry = popup.frameGeometry()
        geometry.moveCenter(center)

        popup.move(_global_pos)

        constrain_popup(
            popup,
            bounds=self.table.window(),
        )

        popup.show()

    def _evaluate_statistic_column(
        self,
        column: StatisticColumn,
    ) -> StatisticColumnResult:

        query = self._prepare_column_query(column.raw_query)

        table = self.data.statistic_engine.evaluate_table(query)

        return self._build_statistic_column_result(
            column=column,
            query=query,
            table=table,
        )

    def _failed_statistic_column_result(
        self,
        *,
        column: StatisticColumn,
        query: StatisticQuery,
        error: Exception,
    ) -> StatisticColumnResult:

        stat_def = self.data.statistic_engine.registry.get(query.statistic_key)

        display_title = (
            StatisticsData.format_query_expression(
                query,
                stat_def,
            )
            if stat_def is not None
            else query.statistic_key
        )
        return StatisticColumnResult(
            column=column,
            query=query,
            table=None,
            neuron_dim=None,
            session_dim=None,
            row_lookup={},
            error=str(error),
            fallback_title=(
                stat_def.title if stat_def is not None else query.statistic_key
            ),
            display_title=display_title,
        )

    def _configure_statistic_selector(
        self,
        selector,
    ):

        if self.row_mode == "single":

            if self.entity_mode == "neuron":
                selector.set_query_mode("neuron_bound")
                selector.set_query_preparer(prepare_neuron_table_query)
                return

            if self.entity_mode == "footprint":
                selector.set_query_mode("component_bound")
                selector.set_query_preparer(prepare_component_table_query)
                return

        if self.row_mode == "pair":

            if self.entity_mode == "neuron":
                selector.set_query_mode("neuron_pair_bound")
                selector.set_query_preparer(prepare_neuron_pair_table_query)
                return

            if self.entity_mode == "footprint":
                selector.set_query_mode("component_pair_bound")
                selector.set_query_preparer(prepare_component_pair_table_query)
                return

        raise ValueError(
            f"Unsupported table mode: " f"{self.entity_mode}/{self.row_mode}"
        )

    def _build_statistic_column_result(
        self,
        *,
        column: StatisticColumn,
        query: StatisticQuery,
        table: PickTable | None,
    ) -> StatisticColumnResult:

        if table is None:

            stat_def = self.data.statistic_engine.registry.get(query.statistic_key)

            display_title = (
                StatisticsData.format_query_expression(query, stat_def)
                if stat_def is not None
                else query.statistic_key
            )

            return StatisticColumnResult(
                column=column,
                query=query,
                table=None,
                neuron_dim=None,
                session_dim=None,
                row_lookup={},
                error=None,
                fallback_title=(
                    stat_def.title if stat_def is not None else query.statistic_key
                ),
                display_title=display_title,
            )

        # ============================================================
        # Single rows
        # ============================================================

        if self.row_mode == "single":

            if self.entity_mode == "neuron":

                neuron_dim = neuron_bound_dim(table.dims)
                if neuron_dim is None:
                    raise ValueError(
                        "Neuron-bound statistic produced " "no neuron dimension."
                    )
                session_dim = None

            elif self.entity_mode == "footprint":

                bound = component_bound_dims(table.dims)
                if bound is None:
                    raise ValueError(
                        "Component-bound statistic produced " "no component dimensions."
                    )
                neuron_dim, session_dim = bound

            else:
                raise ValueError(self.entity_mode)

            lookup = {}
            neurons = np.asarray(table.refs[neuron_dim])

            sessions = (
                np.asarray(table.refs[session_dim]) if session_dim is not None else None
            )

            for row in range(table.n_rows):

                neuron_id = int(neurons[row])
                if sessions is None:
                    key = neuron_id
                else:
                    key = (
                        int(sessions[row]),
                        neuron_id,
                    )

                if key in lookup:
                    raise ValueError(
                        "Statistic result is not uniquely bound "
                        f"to table entities: duplicate {key!r}."
                    )

                lookup[key] = row

            stat_def = self.data.statistic_engine.registry[query.statistic_key]

            display_title = StatisticsData.format_query_expression(
                query,
                stat_def,
            )

            return StatisticColumnResult(
                column=column,
                query=query,
                table=table,
                neuron_dim=neuron_dim,
                session_dim=session_dim,
                row_lookup=lookup,
                display_title=display_title,
            )

        # ============================================================
        # Pair rows
        # ============================================================

        if self.row_mode == "pair":

            neuron_dims = neuron_pair_bound_dims(table.dims)

            if neuron_dims is None:
                raise ValueError(
                    "Pair-bound statistic produced " "no neuron-pair dimensions."
                )

            neuron_i_dim, neuron_j_dim = neuron_dims

            # Which session identity survived the reductions?
            #
            # A shared `session` means both components belong to it.
            if "session" in table.dims:
                session_i_dim = "session"
                session_j_dim = "session"

            else:
                session_i_dim = "session_i" if "session_i" in table.dims else None
                session_j_dim = "session_j" if "session_j" in table.dims else None

            neurons_i = np.asarray(table.refs[neuron_i_dim])
            neurons_j = np.asarray(table.refs[neuron_j_dim])

            sessions_i = (
                np.asarray(table.refs[session_i_dim])
                if session_i_dim is not None
                else None
            )

            sessions_j = (
                np.asarray(table.refs[session_j_dim])
                if session_j_dim is not None
                else None
            )

            lookup = {}
            for row in range(table.n_rows):

                component_i = NeuronComponent(
                    neuron_id=int(neurons_i[row]),
                    session_id=(
                        int(sessions_i[row]) if sessions_i is not None else None
                    ),
                )

                component_j = NeuronComponent(
                    neuron_id=int(neurons_j[row]),
                    session_id=(
                        int(sessions_j[row]) if sessions_j is not None else None
                    ),
                )

                key = _canonical_component_pair(component_i, component_j)

                # if key in lookup:
                #     raise ValueError(
                #         "Statistic result is not uniquely bound "
                #         f"to table entity pairs: duplicate {key!r}."
                #     )

                lookup.setdefault(key, row)

            return StatisticColumnResult(
                column=column,
                query=query,
                table=table,
                neuron_dim=None,
                session_dim=None,
                row_lookup=lookup,
            )

        raise ValueError(f"Unknown row mode: {self.row_mode!r}")

    def _on_statistics_registry_changed(self):
        """
        Re-evaluate the currently configured columns because
        their underlying data may have changed.
        """

        self.statistic_columns = [
            StatisticColumn(raw_query=query)
            for query in self.statistic_config.queries(self.entity_mode, self.row_mode)
        ]

        self._rebuild_statistic_columns()

    def _prepare_column_query(
        self,
        raw_query: StatisticQuery,
    ) -> StatisticQuery:

        if self.row_mode == "single":

            if self.entity_mode == "neuron":
                return prepare_neuron_table_query(
                    raw_query, self.data.statistic_engine.registry
                )

            if self.entity_mode == "footprint":
                return prepare_component_table_query(
                    raw_query, self.data.statistic_engine.registry
                )

        elif self.row_mode == "pair":

            if self.entity_mode == "neuron":
                return prepare_neuron_pair_table_query(
                    raw_query, self.data.statistic_engine.registry
                )

            if self.entity_mode == "footprint":
                return prepare_component_pair_table_query(
                    raw_query, self.data.statistic_engine.registry
                )

        raise ValueError(
            f"Unsupported table mode: " f"{self.entity_mode}/{self.row_mode}"
        )

    def _on_table_mode_changed(self):

        self.entity_mode = self.entity_selector.currentData()
        self.row_mode = self.row_selector.currentData()

        self.statistic_columns = [
            StatisticColumn(raw_query=query)
            for query in self.statistic_config.queries(
                self.entity_mode,
                self.row_mode,
            )
        ]

        self.table.set_mode(
            entity_mode=self.entity_mode,
            row_mode=self.row_mode,
        )

        self._rebuild_statistic_columns()

    def _on_statistic_header_context_requested(
        self,
        index,
        global_pos,
    ):

        menu = QMenu(self.table)

        edit_action = menu.addAction("Edit")
        remove_action = menu.addAction("Remove")

        action = menu.exec(global_pos)

        if action is edit_action:
            self._edit_statistic_column(index, global_pos)

        elif action is remove_action:
            self._remove_statistic_column(index)

    def _on_statistic_display_config_changed(
        self,
        entity_mode: str,
        row_mode: str,
    ):
        if (
            entity_mode,
            row_mode,
        ) != self._table_mode_key():
            return

        self.statistic_columns = [
            StatisticColumn(raw_query=query)
            for query in self.statistic_config.queries(
                entity_mode,
                row_mode,
            )
        ]

        self._rebuild_statistic_columns()

    def _add_statistic_column(
        self,
        raw_query: StatisticQuery,
    ):

        queries = [column.raw_query for column in self.statistic_columns]

        queries.append(raw_query)

        self.statistic_config.set_queries(
            self.entity_mode,
            self.row_mode,
            queries,
        )

    def _edit_statistic_column(
        self,
        index: int,
        global_pos,
    ):

        column = self.statistic_columns[index]

        popup = AddStatisticPopup(
            engine=self.data.statistic_engine,
            popup_bounds=self.table.window(),
            parent=self.table,
        )

        # Important: configure the selector for the current
        # table mode before loading the existing query.
        self._configure_statistic_selector(popup.selector)

        popup.set_query(
            column.raw_query,
            edit=True,
        )

        popup.statisticAccepted.connect(
            lambda query: self._replace_statistic_column(index, query)
        )

        popup.move(global_pos)

        constrain_popup(popup, bounds=self.table.window())

        self._statistic_popup = popup
        popup.show()

    def _remove_statistic_column(
        self,
        index: int,
    ):

        queries = [column.raw_query for column in self.statistic_columns]

        del queries[index]

        self.statistic_config.set_queries(
            self.entity_mode,
            self.row_mode,
            queries,
        )

    def _replace_statistic_column(
        self,
        index: int,
        raw_query: StatisticQuery,
    ):

        queries = [column.raw_query for column in self.statistic_columns]

        queries[index] = raw_query

        self.statistic_config.set_queries(
            self.entity_mode,
            self.row_mode,
            queries,
        )

    def _on_statistic_columns_reordered(
        self,
        order: list[int],
    ):

        queries = [self.statistic_columns[index].raw_query for index in order]

        self.statistic_config.set_queries(
            self.entity_mode,
            self.row_mode,
            queries,
        )

    def _pair_rows_for_evaluation(
        self,
        rows: tuple[SelectionTableRow, ...],
    ):
        """
        Resolve the currently displayed pair rows into the components
        needed for statistic evaluation.

        For neuron mode, session identity remains intentionally unbound.
        For footprint mode, session-independent entities are resolved
        against the current session.
        """

        current_session_id = self.state.current_session_id

        resolved_rows = []

        for row in rows:

            if self.entity_mode == "neuron":
                resolved_rows.append(tuple(row.entities))
                continue

            components = []

            for entity in row.entities:

                session_id = entity.session_id

                if session_id is None:
                    session_id = current_session_id

                if session_id is None:
                    components = None
                    break

                components.append(
                    NeuronComponent(
                        neuron_id=int(entity.neuron_id),
                        session_id=int(session_id),
                    )
                )

            resolved_rows.append(None if components is None else tuple(components))

        return tuple(resolved_rows)

    def _bind_pair_query_to_components(
        self,
        query: StatisticQuery,
        components: tuple[NeuronComponent, NeuronComponent],
        *,
        entity_mode: EntityMode,
    ) -> StatisticQuery | None:

        stat_def = self.data.statistic_engine.registry[query.statistic_key]

        neuron_dims = neuron_pair_bound_dims(stat_def.dims)

        if neuron_dims is None:
            raise ValueError(
                f"Statistic {query.statistic_key!r} has no neuron-pair dimensions."
            )

        component_i, component_j = components

        # ---------------------------------------------------------
        # Filters on dimensions that we are about to bind must be
        # checked before those dimensions disappear from PickTable.
        # ---------------------------------------------------------

        for pair_filter in query.filters:

            if pair_filter.target == "neuron":

                same = component_i.neuron_id == component_j.neuron_id

                if pair_filter.relation == "same" and not same:
                    return None

                if pair_filter.relation == "different" and same:
                    return None

                if pair_filter.relation == "with previous":
                    raise ValueError(
                        "Neuron 'with previous' filtering is not supported "
                        "for pointwise SelectionDisplay pair evaluation."
                    )

            elif pair_filter.target == "session" and entity_mode == "footprint":

                session_i = component_i.session_id
                session_j = component_j.session_id

                if session_i is None or session_j is None:
                    return None

                if pair_filter.relation == "same":
                    if session_i != session_j:
                        return None

                elif pair_filter.relation == "different":
                    if session_i == session_j:
                        return None

                elif pair_filter.relation == "with previous":

                    # Keep the statistic's i/j orientation meaningful.
                    if session_j == session_i - 1:
                        pass

                    elif session_i == session_j - 1:
                        component_i, component_j = (component_j, component_i)

                    else:
                        return None

        # ---------------------------------------------------------
        # Bind the pair dimensions to this table row.
        # ---------------------------------------------------------

        neuron_i_dim, neuron_j_dim = neuron_dims

        reductions = query.reduction_dict()

        reductions[neuron_i_dim] = ReductionSpec(
            "single",
            index=int(component_i.neuron_id),
        )
        reductions[neuron_j_dim] = ReductionSpec(
            "single",
            index=int(component_j.neuron_id),
        )

        # ---------------------------------------------------------
        # Footprint pair mode additionally binds session identity.
        #
        # Shared `session`:
        #     both footprints must come from the same session.
        #
        # session_i/session_j:
        #     each footprint gets its own session.
        # ---------------------------------------------------------

        if entity_mode == "footprint":

            if "session" in stat_def.dims:

                if component_i.session_id != component_j.session_id:
                    return None

                reductions["session"] = ReductionSpec(
                    "single",
                    index=int(component_i.session_id),
                )

            else:

                if "session_i" in stat_def.dims:
                    reductions["session_i"] = ReductionSpec(
                        "single",
                        index=int(component_i.session_id),
                    )

                if "session_j" in stat_def.dims:
                    reductions["session_j"] = ReductionSpec(
                        "single",
                        index=int(component_j.session_id),
                    )

        return replace(
            query,
            reductions=tuple(sorted(reductions.items())),
        )

    def _evaluate_pair_statistic_column(
        self,
        *,
        column: StatisticColumn,
        query: StatisticQuery,
        pair_rows,
        entity_mode: EntityMode,
    ) -> StatisticColumnResult:

        stat_def = self.data.statistic_engine.registry[query.statistic_key]

        values = np.full(
            len(pair_rows),
            np.nan,
            dtype=float,
        )

        for row_index, components in enumerate(pair_rows):

            if components is None:
                continue

            bound_query = self._bind_pair_query_to_components(
                query,
                components,
                entity_mode=entity_mode,
            )

            if bound_query is None:
                continue

            table = self.data.statistic_engine.evaluate_table(bound_query)

            if table is None or table.n_rows == 0:
                continue

            if table.n_rows != 1:
                raise ValueError(
                    f"Pair-bound query for {query.statistic_key!r} "
                    f"produced {table.n_rows} rows instead of one."
                )

            values[row_index] = table.values[0]

        display_title = StatisticsData.format_query_expression(query, stat_def)

        return StatisticColumnResult(
            column=column,
            query=query,
            table=None,
            neuron_dim=None,
            session_dim=None,
            row_lookup={},
            row_values=values,
            fallback_title=stat_def.title,
            display_title=display_title,
        )

    def _rebuild_statistic_columns(
        self,
        update_display=True,
    ):

        self._statistic_rebuild_generation += 1
        generation = self._statistic_rebuild_generation

        mode = self._table_mode_key()
        columns = tuple(self.statistic_columns)

        rows = tuple(self.table.rows)

        pair_rows = self._pair_rows_for_evaluation(rows) if mode[1] == "pair" else None

        # Nothing to calculate.
        if not columns:
            self.statistic_results = []

            if update_display:
                self.table.set_statistic_columns([])

            return

        # Prepare queries on the GUI thread while the current
        # entity/row mode is known and stable.
        prepared = [
            (column, self._prepare_column_query(column.raw_query)) for column in columns
        ]

        # Don't leave results from the previous mode/configuration
        # visible while the new ones are being calculated.
        self.statistic_results = []

        if update_display:
            self.table.set_statistic_columns([])

        def evaluate():

            evaluated = []

            for column, query in prepared:

                table = None
                pair_result = None

                try:

                    if pair_rows is not None:
                        pair_result = self._evaluate_pair_statistic_column(
                            column=column,
                            query=query,
                            pair_rows=pair_rows,
                            entity_mode=mode[0],
                        )

                    else:
                        table = self.data.statistic_engine.evaluate_table(query)

                    error = None

                except Exception as exc:
                    table = None
                    pair_result = None
                    error = exc

                evaluated.append(
                    (
                        column,
                        query,
                        table,
                        pair_result,
                        error,
                    )
                )

            return evaluated

        self.state.tasks.start(
            "calculating",
            "Calculate Selection Display statistics",
            evaluate,
            on_result=lambda evaluated: (
                self._on_statistic_columns_ready(
                    evaluated,
                    generation=generation,
                    mode=mode,
                    columns=columns,
                    rows=rows,
                    update_display=update_display,
                )
            ),
            unique=True,
        )

    def _on_statistic_columns_ready(
        self,
        evaluated,
        *,
        generation: int,
        mode,
        columns,
        rows,
        update_display: bool,
    ):

        # A newer calculation has already been requested.
        if generation != self._statistic_rebuild_generation:
            return

        # The user changed table mode while this was running.
        if mode != self._table_mode_key():
            return

        # The column configuration changed while this was running.
        if columns != tuple(self.statistic_columns):
            return

        # The actual displayed entity rows changed while calculation was running.
        if rows != tuple(self.table.rows):
            return

        results = []

        for column, query, table, pair_result, error in evaluated:

            if error is None:

                if pair_result is not None:
                    result = pair_result

                else:
                    result = self._build_statistic_column_result(
                        column=column,
                        query=query,
                        table=table,
                    )

            else:
                self.state.logger.error(
                    "Failed to evaluate Selection Display statistic "
                    f"{query.statistic_key!r}: {error}"
                )

                result = self._failed_statistic_column_result(
                    column=column, query=query, error=error
                )

            results.append(result)

        self.statistic_results = results

        if update_display:
            self.table.set_statistic_columns(results)

    # ----- controller → display state updates -----
    def update_neuron_selection(self):
        self.table.update_display()
        self._rebuild_statistic_columns()

    def update_styles(self):
        self.table.update_styles()

    # ----- saving / loading options


class AddStatisticPopup(QDialog):

    statisticAccepted = Signal(object)

    def __init__(
        self,
        *,
        engine,
        popup_bounds,
        parent=None,
    ):
        super().__init__(parent)

        self.setWindowFlags(Qt.WindowType.Popup)

        self.setWindowTitle("Add statistic to Selection Display")

        layout = QVBoxLayout(self)
        layout.setContentsMargins(8, 8, 8, 8)

        self.popup_bounds = popup_bounds
        self.selector = StatisticsData.StatisticQuerySelector(
            engine=engine,
            parent=self,
        )
        self.selector.set_popup_bounds(popup_bounds)

        layout.addWidget(self.selector)

        button_layout = QHBoxLayout()
        button_layout.addStretch()

        self.cancel_button = QPushButton("Cancel")

        self.add_button = QPushButton("Add")

        button_layout.addWidget(self.cancel_button)
        button_layout.addWidget(self.add_button)

        layout.addLayout(button_layout)

        self.cancel_button.clicked.connect(self.close)

        self.add_button.clicked.connect(self._accept_current_query)

        self.selector.queryChanged.connect(self._on_query_changed)

        self._on_query_changed(self.selector.effective_query())

    def _on_query_changed(self, query):
        self.add_button.setEnabled(self.selector.effective_query() is not None)

    def set_query(
        self,
        query: StatisticQuery,
        *,
        edit: bool = False,
    ):

        self.selector.set_query(
            query,
            emit=False,
        )

        if edit:
            self.setWindowTitle("Edit Selection Display statistic")
            self.add_button.setText("Apply")

        self._on_query_changed(self.selector.effective_query())

    def _accept_current_query(self):

        query = self.selector.raw_query()

        if query is None:
            return

        self.statisticAccepted.emit(query)

        self.close()
