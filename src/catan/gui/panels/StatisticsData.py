from typing import get_args

from PySide6.QtGui import QAction, QActionGroup
from PySide6.QtCore import Qt, Signal, QPoint

from PySide6.QtWidgets import (
    QDialog,
    QWidget,
    QHBoxLayout,
    QLabel,
    QComboBox,
    QMenu,
    QSpinBox,
    QToolButton,
    QVBoxLayout,
)

from catan.gui.data.statistics import (
    StatisticDefinition,
    StatisticEngine,
    CATEGORY_TITLES,
)
from catan.gui.data.statistics.dimensions import (
    NEURON_DIMS,
    SESSION_DIMS,
    DimensionInfo,
)
from catan.gui.data.statistics.queries import (
    Contexts,
    ReductionMethod,
    ReductionSpec,
    allowed_error_methods,
    allowed_reduction_methods,
    statistic_allowed_in_context,
    PairFilter,
    PairRelation,
    PairTarget,
    ReductionSpec,
    StatisticQuery,
    neuron_bound_dim,
    component_bound_dims,
    neuron_pair_bound_dims,
    component_pair_bound_dims,
    normalize_neuron_bound_reductions,
    normalize_session_series_reductions,
    normalize_generic_session_pair_reductions,
)
from catan.gui.utils.popups import constrain_popup


class ReductionRow(QWidget):
    changed = Signal(str, object)  # dim_name, ReductionSpec

    def __init__(
        self,
        dim_name: str,
        methods: tuple[str, ...],
        dim_info: DimensionInfo,
        spec: ReductionSpec,
        error_methods_for_method,
        parent=None,
    ):
        super().__init__(parent)

        self.dim_name = dim_name
        self.dim_info = dim_info
        self.error_methods_for_method = error_methods_for_method
        self._updating = False

        layout = QHBoxLayout(self)
        layout.setContentsMargins(8, 3, 8, 3)

        self.label = QLabel(dim_name)
        self.label.setMinimumWidth(90)

        self.method_combo = QComboBox()
        self.method_combo.addItems(methods)

        self.index_spin = QSpinBox()
        self.index_spin.setRange(0, max(0, dim_info.size - 1))

        self.index_label = QLabel("")
        self.index_label.setMinimumWidth(80)

        self.error_label = QLabel("err")

        self.error_combo = QComboBox()
        self.error_combo.setMinimumWidth(80)

        layout.addWidget(self.label)
        layout.addWidget(self.method_combo)
        layout.addWidget(self.index_spin)
        layout.addWidget(self.index_label)
        layout.addWidget(self.error_label)
        layout.addWidget(self.error_combo)

        self.method_combo.currentTextChanged.connect(self._on_method_changed)
        self.index_spin.valueChanged.connect(self._emit_changed)
        self.error_combo.currentTextChanged.connect(self._emit_changed)

        self.set_spec(spec, emit=False)

    def set_spec(self, spec: ReductionSpec, *, emit: bool = False):
        self._updating = True
        try:
            self.method_combo.blockSignals(True)
            self.index_spin.blockSignals(True)
            self.error_combo.blockSignals(True)

            if spec.method in [
                self.method_combo.itemText(i) for i in range(self.method_combo.count())
            ]:
                self.method_combo.setCurrentText(spec.method)
            #
            self.index_spin.setValue(spec.index or 0)

            self._update_index_visibility()
            self._update_error_methods(
                preferred_error_method=spec.error_method,
                emit=False,
            )

        finally:
            self.error_combo.blockSignals(False)
            self.index_spin.blockSignals(False)
            self.method_combo.blockSignals(False)
            self._updating = False

        if emit:
            self._emit_changed()

    def _current_spec(self) -> ReductionSpec:
        method = self.method_combo.currentText()

        assert method in get_args(
            ReductionMethod
        ), f"Invalid reduction method: {method}"

        if method == "single":
            return ReductionSpec(
                "single",
                index=self.index_spin.value(),
                error_method="none",
            )

        error_method = self.error_combo.currentText() or "none"

        return ReductionSpec(
            method,
            error_method=error_method,
        )

    def _on_method_changed(self, *args):
        if self._updating:
            return

        self._update_index_visibility()
        self._update_error_methods(emit=False)
        self._emit_changed()

    def _emit_changed(self, *args):
        if self._updating:
            return

        self._update_index_visibility()
        self.changed.emit(self.dim_name, self._current_spec())

    def _update_index_visibility(self):
        is_single = self.method_combo.currentText() == "single"

        self.index_spin.setVisible(is_single)
        self.index_label.setVisible(is_single)

        if is_single and self.dim_info.labels:
            idx = self.index_spin.value()
            if idx < len(self.dim_info.labels):
                self.index_label.setText(self.dim_info.labels[idx])
            else:
                self.index_label.setText(str(idx))

    def _update_error_methods(
        self,
        *,
        preferred_error_method: str | None = None,
        emit: bool = True,
    ):
        method = self.method_combo.currentText()
        methods = self.error_methods_for_method(method)

        old = preferred_error_method or self.error_combo.currentText() or "none"

        self.error_combo.blockSignals(True)
        self.error_combo.clear()

        if not methods:
            methods = ("none",)

        self.error_combo.addItems(methods)

        if old in methods:
            self.error_combo.setCurrentText(old)
        else:
            self.error_combo.setCurrentText(methods[0])

        self.error_combo.blockSignals(False)

        show_error = not (len(methods) == 1 and methods[0] == "none")

        self.error_label.setVisible(show_error)
        self.error_combo.setVisible(show_error)
        self.error_combo.setEnabled(show_error)

        if emit:
            self._emit_changed()


class ReductionPopup(QDialog):
    reductionChanged = Signal(str, object)

    def __init__(
        self,
        stat_def: StatisticDefinition,
        dim_info: dict[str, DimensionInfo],
        reductions: dict[str, ReductionSpec],
        parent: "StatisticQuerySelector",
    ):
        super().__init__(parent)
        self.selector = parent

        self.query_mode: Contexts = (
            parent.query_mode if parent is not None else "generic"
        )

        self.setWindowFlags(Qt.WindowType.Popup)
        self.setObjectName("ReductionPopup")

        self.rows: dict[str, ReductionRow] = {}

        layout = QVBoxLayout(self)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(4)

        self.stat_def = stat_def

        title = QLabel(f"<b>{stat_def.title}</b>")
        layout.addWidget(title)

        for dim in stat_def.dims:

            methods = self.selector.reduction_methods_for_dim(dim)

            # No user-adjustable reduction for this dimension.
            if not methods:
                continue

            spec = reductions.get(
                dim,
                ReductionSpec("keep"),
            )

            info = dim_info[dim]

            row = ReductionRow(
                dim_name=dim,
                methods=methods,
                dim_info=info,
                spec=spec,
                error_methods_for_method=(
                    lambda method, dim=dim: self.selector.error_methods_for_dim(
                        dim,
                        method,
                    )
                ),
                parent=self,
            )

            row.changed.connect(self.reductionChanged.emit)

            layout.addWidget(row)
            self.rows[dim] = row

    def set_reductions(
        self,
        reductions: dict[str, ReductionSpec],
        *,
        emit: bool = False,
    ):
        for dim, row in self.rows.items():
            row.set_spec(
                reductions.get(dim, ReductionSpec("keep")),
                emit=emit,
            )


class PairFilterSelector(QToolButton):
    filterChanged = Signal()

    def __init__(
        self,
        target: PairTarget,
        parent=None,
        *,
        collapse_same: bool = True,
    ):
        super().__init__(parent)

        self.target = target
        self.collapse_same = collapse_same

        self._current_relation: PairRelation = "all"
        self._filter_menu = QMenu(self)
        self._action_group = QActionGroup(self)
        self._action_group.setExclusive(True)

        self._relation_actions: dict[PairRelation, QAction] = {}

        self.setPopupMode(QToolButton.ToolButtonPopupMode.InstantPopup)
        self.setMenu(self._filter_menu)

        self._build_menu()
        self._update_text()

    def _build_menu(self):
        self._filter_menu.clear()

        relations = (
            ("all", "same", "different", "with previous")
            if self.target == "session"
            else ("all", "same", "different")
        )

        for relation in relations:
            action = QAction(relation, self._filter_menu)
            action.setCheckable(True)
            action.setData(relation)

            if relation == self._current_relation:
                action.setChecked(True)

            self._action_group.addAction(action)
            self._filter_menu.addAction(action)
            self._relation_actions[relation] = action

        self._action_group.triggered.connect(self._on_action_triggered)

    def _on_action_triggered(self, action: QAction):
        relation = action.data()

        if relation not in ("all", "same", "different", "with previous"):
            return

        self._current_relation = relation
        self._update_text()
        self.filterChanged.emit()

    def _update_text(self):
        if self._current_relation == "all":
            self.setText(f"{self.target}: all")
        elif self._current_relation == "same":
            self.setText(f"{self.target}: same")
        elif self._current_relation == "different":
            self.setText(f"{self.target}: different")
        else:
            self.setText(f"{self.target}: with previous")

    def current_relation(self) -> PairRelation:
        return self._current_relation

    def selected_filter(self) -> PairFilter | None:
        if self._current_relation == "all":
            return None

        return PairFilter(
            target=self.target,
            relation=self._current_relation,
            collapse_same=self.collapse_same,
        )

    def set_relation(self, relation: PairRelation, *, emit: bool = True):
        self._current_relation = relation

        action = self._relation_actions.get(relation)
        if action is not None:
            action.setChecked(True)

        self._update_text()

        if emit:
            self.filterChanged.emit()


_DIM_DISPLAY = {
    "neuron": "n",
    "neuron_i": "nᵢ",
    "neuron_j": "nⱼ",
    "session": "s",
    "session_i": "sᵢ",
    "session_j": "sⱼ",
}

_DIM_SUBSCRIPT = {
    "neuron": "ₙ",
    "neuron_i": "ₙᵢ",
    "neuron_j": "ₙⱼ",
    "session": "ₛ",
    "session_i": "ₛᵢ",
    "session_j": "ₛⱼ",
}


def format_dimension_short(dim: str) -> str:
    return _DIM_DISPLAY.get(dim, dim)


def format_reduction_short(
    dim: str,
    spec: ReductionSpec,
) -> str:

    dim_short = format_dimension_short(dim)

    if spec.method == "keep":
        return dim_short

    if spec.method == "single":
        return f"{dim_short}={spec.index}"

    subscript = _DIM_SUBSCRIPT.get(dim)

    if subscript is None:
        text = f"{spec.method}({dim_short})"
    else:
        text = f"{spec.method}{subscript}"

    if spec.error_method != "none":
        text += f"±{spec.error_method}"

    return text


def format_query_expression(
    query: StatisticQuery,
    stat_def: StatisticDefinition,
) -> str:

    reductions = query.reduction_dict()

    # -----------------------------------------
    # Dimensions that remain arguments
    # -----------------------------------------

    arguments = []

    for dim in stat_def.dims:

        spec = reductions.get(
            dim,
            ReductionSpec("keep"),
        )

        if spec.method == "keep":
            arguments.append(_DIM_DISPLAY.get(dim, dim))

        elif spec.method == "single":
            arguments.append(f"{_DIM_DISPLAY.get(dim, dim)}={spec.index}")

    # Use the short/internal statistic name here,
    # not the verbose menu title.
    base_name = stat_def.title

    if arguments:
        expression = f"{base_name}(" f"{', '.join(arguments)}" f")"
    else:
        expression = base_name

    # -----------------------------------------
    # Wrap actual reductions
    # -----------------------------------------

    reduction_dims = [
        dim
        for dim in query.reduction_order
        if reductions.get(
            dim,
            ReductionSpec("keep"),
        ).method
        not in ("keep", "single")
    ]

    # Generic queries currently often have no explicit
    # reduction_order. Fall back to statistic dimension order.
    if not reduction_dims:
        reduction_dims = [
            dim
            for dim in stat_def.dims
            if reductions.get(
                dim,
                ReductionSpec("keep"),
            ).method
            not in ("keep", "single")
        ]

    for dim in reduction_dims:

        spec = reductions[dim]

        subscript = _DIM_SUBSCRIPT.get(
            dim,
            f"_{dim}",
        )

        expression = f"{spec.method}{subscript}" f"({expression})"

    return expression


class StatisticQuerySelector(QWidget):
    queryChanged = Signal(object)  # StatisticQuery | None

    def __init__(self, engine: StatisticEngine, axis=None, parent=None):
        super().__init__(parent)

        self.engine = engine
        self.axis = axis

        self._popup_bounds = None

        self.query_mode: Contexts = "generic"
        self.query_preparer = None
        self._syncing_reductions = False
        self._emitting_query = False

        self.default_reduction_provider = None
        self.current_reductions: dict[str, ReductionSpec] = {}
        self._popup = None

        self._current_stat_key = next(iter(self.engine.registry.keys()))

        if axis == "y":
            layout = QVBoxLayout(self)
        else:
            layout = QHBoxLayout(self)

        layout.setContentsMargins(0, 0, 0, 0)
        layout.addStretch()

        self.stat_button = QToolButton()
        self.stat_button.setMaximumWidth(120)

        self.stat_button.setToolButtonStyle(Qt.ToolButtonStyle.ToolButtonTextOnly)
        self.stat_button.setPopupMode(QToolButton.ToolButtonPopupMode.InstantPopup)

        self.stat_menu = QMenu(self.stat_button)
        self.stat_button.setMenu(self.stat_menu)
        constrain_popup(
            self.stat_menu,
            anchor=self.stat_button,
            bounds=self._popup_bounds,
        )

        self._build_stat_menu()

        self.reduction_button = QToolButton()
        self.reduction_button.setMaximumWidth(160)
        self.reduction_button.setText("Reductions")
        self.reduction_button.clicked.connect(self._open_reduction_popup)

        self.neuron_filter_button = PairFilterSelector("neuron", parent=self)
        self.session_filter_button = PairFilterSelector("session", parent=self)

        self.neuron_filter_button.filterChanged.connect(self._on_filter_changed)
        self.session_filter_button.filterChanged.connect(self._on_filter_changed)

        self.engine.registry_changed.connect(self.refresh_statistics)

        layout.addWidget(self.stat_button, 1)
        layout.addWidget(self.reduction_button, 1)
        layout.addWidget(self.neuron_filter_button, 1)
        layout.addWidget(self.session_filter_button, 1)

        self._on_statistic_changed()
        self._update_filter_visibility()

    def set_popup_bounds(
        self,
        bounds: QWidget,
    ):
        self._popup_bounds = bounds

    def set_query(
        self,
        query: StatisticQuery,
        *,
        emit: bool = True,
    ):

        if query.statistic_key not in self.engine.registry:
            raise KeyError(query.statistic_key)

        self._current_stat_key = query.statistic_key
        self.current_reductions = dict(query.reductions)

        neuron_relation = "all"
        session_relation = "all"

        for filter_ in query.filters:
            if filter_.target == "neuron":
                neuron_relation = filter_.relation
            elif filter_.target == "session":
                session_relation = filter_.relation

        self.neuron_filter_button.set_relation(
            neuron_relation,
            emit=False,
        )
        self.session_filter_button.set_relation(
            session_relation,
            emit=False,
        )

        self._update_stat_menu_checks()
        self._update_stat_button()
        self._update_filter_visibility()

        # Reconcile the stored raw query with the current table context.
        self.sync_visible_reductions_from_effective_query()

        self._update_summary()

        if emit:
            self._emit_query_changed_once()

    def refresh_statistics(self):
        old_key = self._current_stat_key

        if old_key not in self.engine.registry:
            self._current_stat_key = "none"

        self._build_stat_menu()

        if self._current_stat_key != old_key:
            self._on_statistic_changed()
            return

        # Registry changed, but current statistic still exists.
        # Its underlying sources/data may nevertheless have changed.
        if self._current_stat_key != "none":
            self._update_summary()
            self._emit_query_changed_once()

    def _emit_query_changed_once(self):
        if self._syncing_reductions or self._emitting_query:
            return

        self._emitting_query = True
        try:
            self.queryChanged.emit(self.effective_query())
        finally:
            self._emitting_query = False

    def set_query_preparer(self, preparer):
        self.query_preparer = preparer
        self.sync_visible_reductions_from_effective_query()
        self._emit_query_changed_once()

    def _build_stat_menu(self):
        self.stat_menu.clear()
        self._stat_actions = {}

        registry = self.engine.registry

        # "None" stays directly at the top
        if "none" in registry:
            self._add_stat_action(
                self.stat_menu,
                "none",
                registry["none"],
            )
            self.stat_menu.addSeparator()

        for category, title in CATEGORY_TITLES.items():

            stats = [
                (key, stat_def)
                for key, stat_def in registry.items()
                if (
                    stat_def.category == category
                    and statistic_allowed_in_context(
                        stat_def.dims,
                        self.query_mode,
                    )
                )
            ]

            if not stats:
                continue

            submenu = self.stat_menu.addMenu(title)

            for key, stat_def in sorted(
                stats,
                key=lambda item: item[1].title.lower(),
            ):
                self._add_stat_action(
                    submenu,
                    key,
                    stat_def,
                )

        self._update_stat_menu_checks()

    def _add_stat_action(
        self,
        menu: QMenu,
        key: str,
        stat_def: StatisticDefinition,
    ):
        action = QAction(
            stat_def.title,
            menu,
        )

        action.setCheckable(True)
        action.setData(key)
        action.setToolTip(stat_def.description)

        action.triggered.connect(
            lambda checked=False, key=key: self._set_statistic(key)
        )

        menu.addAction(action)

        self._stat_actions[key] = action

    def _set_statistic(self, key: str):
        if key == self._current_stat_key:
            return

        self._current_stat_key = key
        self._update_stat_menu_checks()
        self._on_statistic_changed()

    def set_query_mode(self, mode: Contexts):
        if mode == self.query_mode:
            return

        self.query_mode = mode

        if self._current_stat_key != "none" and not statistic_allowed_in_context(
            self.current_stat_def().dims,
            self.query_mode,
        ):
            self._current_stat_key = "none"

        self._build_stat_menu()
        self._update_stat_button()

        self._apply_default_reductions(context=self.query_mode)
        self.sync_visible_reductions_from_effective_query()

        self._update_summary()
        self._update_filter_visibility()

        self._emit_query_changed_once()

    def set_default_reduction_provider(self, provider):
        self.default_reduction_provider = provider

    def _apply_default_reductions(
        self,
        context: Contexts = "generic",
    ):
        if self.current_stat_key() == "none":
            return

        stat_def = self.current_stat_def()

        if context == "session_series":
            self.current_reductions = normalize_session_series_reductions(
                stat_def,
                default_session_series_reductions(stat_def),
                self.current_filters(),
            )
            return

        if stat_def.default_reductions is not None:
            reductions = dict(stat_def.default_reductions)
        else:
            reductions = stat_def.get_default_reductions()

        # Consumer-specific defaults, e.g. Overview mode
        if self.default_reduction_provider is not None:
            overrides = self.default_reduction_provider(stat_def)
            if overrides:
                reductions.update(overrides)

        if context == "neuron_bound":
            reductions = normalize_neuron_bound_reductions(
                stat_def,
                reductions,
            )

        self.current_reductions = reductions

    def _update_stat_menu_checks(self):
        for key, action in self._stat_actions.items():
            action.setChecked(key == self._current_stat_key)

    def current_stat_key(self):
        return self._current_stat_key

    def current_stat_def(self):
        return self.engine.registry[self.current_stat_key()]

    def raw_query(self) -> StatisticQuery | None:
        if self.current_stat_key() == "none":
            return None

        return StatisticQuery(
            statistic_key=self.current_stat_key(),
            reductions=tuple(sorted(self.current_reductions.items())),
            filters=self.current_filters(),
            context=self.query_mode,
        )

    def effective_query(self) -> StatisticQuery | None:
        query = self.raw_query()

        if query is None:
            return None

        if self.query_preparer is None:
            return query

        return self.query_preparer(query, self.engine.registry)

    def current_query(self) -> StatisticQuery | None:
        # Important:
        # This must be pure. No sync. No widget updates. No repair.
        return self.raw_query()

    def _sync_reductions_from_effective_query(self):
        if self._syncing_reductions:
            return

        query = self.raw_query()

        if query is None or self.query_preparer is None:
            return

        effective_query = self.query_preparer(query, self.engine.registry)

        if effective_query is None:
            return

        effective_reductions = effective_query.reduction_dict()

        if effective_reductions == self.current_reductions:
            return

        self._syncing_reductions = True
        try:
            self.current_reductions = effective_reductions
            self._update_summary()

            if self._popup is not None:
                self._popup.set_reductions(self.current_reductions)

        finally:
            self._syncing_reductions = False

    def sync_visible_reductions_from_effective_query(self) -> bool:
        """
        Bring self.current_reductions and the popup UI into agreement with the
        effective plot-specific query.

        Returns True if reductions changed.
        """
        if self._syncing_reductions:
            return False

        query = self.raw_query()

        if query is None or self.query_preparer is None:
            return False

        effective_query = self.query_preparer(query, self.engine.registry)

        if effective_query is None:
            return False

        effective_reductions = effective_query.reduction_dict()

        if effective_reductions == self.current_reductions:
            return False

        self._syncing_reductions = True
        try:
            self.current_reductions = effective_reductions
            self._update_summary()

            if self._popup is not None:
                self._popup.set_reductions(
                    self.current_reductions,
                    emit=False,
                )

        finally:
            self._syncing_reductions = False

        return True

    def current_filters(self) -> tuple[PairFilter, ...]:
        filters: list[PairFilter] = []

        stat_def = self.current_stat_def()

        if stat_def.has_pair_dimension("neuron"):
            f = self.neuron_filter_button.selected_filter()
            if f is not None:
                filters.append(f)

        if stat_def.has_pair_dimension("session"):
            f = self.session_filter_button.selected_filter()
            if f is not None:
                filters.append(f)

        return tuple(filters)

    def _on_statistic_changed(self):
        self._apply_default_reductions(context=self.query_mode)
        self.sync_visible_reductions_from_effective_query()

        self._update_stat_button()
        self._update_summary()
        self._update_filter_visibility()

        self._emit_query_changed_once()

    def _update_stat_button(self):
        stat_def = self.current_stat_def()

        self.stat_button.setText(stat_def.title)
        self.stat_button.setToolTip(stat_def.description)

    def _update_filter_visibility(self):
        stat_def = self.current_stat_def()

        show_neuron = stat_def.has_pair_dimension("neuron")
        show_session = stat_def.has_pair_dimension("session")

        self.neuron_filter_button.setVisible(show_neuron)
        self.session_filter_button.setVisible(show_session)

        # Optional: reset hidden filters to all
        if not show_neuron:
            self.neuron_filter_button.set_relation("all", emit=False)

        if not show_session:
            self.session_filter_button.set_relation("all", emit=False)

    def _get_dimension_info(self, stat_def):
        return stat_def.get_dimension_info(self.engine.state)

    def _open_reduction_popup(self):
        stat_def = self.current_stat_def()

        if self.current_stat_key() == "none":
            return

        self._sync_reductions_from_effective_query()

        dim_info = self._get_dimension_info(stat_def)

        self._popup = ReductionPopup(
            stat_def=stat_def,
            dim_info=dim_info,
            reductions=self.current_reductions,
            parent=self,
        )

        self._popup.reductionChanged.connect(self._on_reduction_changed)

        constrain_popup(
            self._popup,
            anchor=self.reduction_button,
            bounds=self._popup_bounds,
            placements=("right", "left", "above", "below"),
        )

        self._popup.show()

    def _on_reduction_changed(
        self,
        dim_name,
        spec,
    ):
        if self._syncing_reductions:
            return

        self.current_reductions[dim_name] = spec

        if self.query_mode == "session_series":
            self.current_reductions = normalize_session_series_reductions(
                self.current_stat_def(),
                self.current_reductions,
                self.current_filters(),
            )
        else:
            self.current_reductions = normalize_generic_session_pair_reductions(
                self.current_stat_def(),
                self.current_reductions,
                self.current_filters(),
            )

        self.sync_visible_reductions_from_effective_query()
        self._update_summary()

        self._emit_query_changed_once()

    def _on_filter_changed(self):
        if self._syncing_reductions:
            return

        if self.query_mode == "session_series":
            self.current_reductions = normalize_session_series_reductions(
                self.current_stat_def(),
                self.current_reductions,
                self.current_filters(),
            )
        else:
            self.current_reductions = normalize_generic_session_pair_reductions(
                self.current_stat_def(),
                self.current_reductions,
                self.current_filters(),
            )

        self.sync_visible_reductions_from_effective_query()
        self._update_summary()

        self._emit_query_changed_once()

    def _update_summary(self):

        parts = []
        stat_def = self.current_stat_def()

        if self.current_stat_key() == "none":
            self.reduction_button.setText("-")
            return

        for dim in stat_def.dims:

            if not self.reduction_methods_for_dim(dim):
                continue

            spec = self.current_reductions.get(dim, ReductionSpec("keep"))

            parts.append(format_reduction_short(dim, spec))

        self.reduction_button.setText(", ".join(parts))

    def reduction_methods_for_dim(
        self,
        dim: str,
    ) -> tuple[str, ...]:

        stat_def = self.current_stat_def()

        general = allowed_reduction_methods(
            dim,
            context=self.query_mode,
        )

        specific = stat_def.get_allowed_reductions(dim)

        methods = tuple(m for m in specific if m in general)

        # ------------------------------------------------
        # Bound entity contexts
        # ------------------------------------------------

        if self.query_mode == "neuron_bound":

            output_dim = neuron_bound_dim(stat_def.dims)
            if dim == output_dim:
                # Implicitly kept by the bound context.
                return ()

            # Every other dimension must disappear.
            return tuple(method for method in methods if method != "keep")

        if self.query_mode == "component_bound":

            bound = component_bound_dims(stat_def.dims)

            if bound is None:
                return ()

            neuron_dim, session_dim = bound

            # Neuron identity is fixed by the table row.
            if dim == neuron_dim:
                return ()

            # Session may remain component-specific OR be reduced.
            if dim == session_dim:
                return methods

            # All other dimensions must disappear.
            return tuple(method for method in methods if method != "keep")

        if self.query_mode == "neuron_pair_bound":

            neuron_dims = neuron_pair_bound_dims(stat_def.dims)

            if neuron_dims is None:
                return ()

            if dim in neuron_dims:
                return ()

            return tuple(method for method in methods if method != "keep")

        if self.query_mode == "component_pair_bound":

            bound = component_pair_bound_dims(stat_def.dims)

            if bound is None:
                return ()

            neuron_dims, session_dims = bound

            if dim in neuron_dims:
                return ()

            if dim in session_dims:
                return methods

            return tuple(method for method in methods if method != "keep")

        session_dims = [d for d in stat_def.dims if d in SESSION_DIMS]

        if dim not in session_dims:
            return methods

        # ordinary, non-pair session statistic
        if session_dims == ["session"]:
            if self.query_mode == "session_series":
                return ()
            return methods

        # only special-case actual session-pair statistics
        if not ("session_i" in session_dims and "session_j" in session_dims):
            return methods

        relation = self.session_filter_button.current_relation()

        # ------------------------------------------------
        # Session-series plot
        # ------------------------------------------------
        if self.query_mode == "session_series":

            # session_i is the x-axis
            if dim == "session_i":
                return ()

            # session_j is implied by the relation
            if relation in ("same", "with previous"):
                return ()

            # all / different:
            # choose one reference session_j
            if dim == "session_j":
                return ("single",)

        # ------------------------------------------------
        # Generic / histogram / scatter
        # ------------------------------------------------
        else:

            # same / previous:
            # user chooses session_i only;
            # session_j is derived automatically
            if relation in ("same", "with previous"):
                if dim == "session_i":
                    return ("single",)
                return ()

            # different:
            # both sessions must be independently selectable
            if relation == "different":
                return ("single",)

            # all:
            # no relation constraint
            return methods

        return methods

    def error_methods_for_dim(
        self,
        dim: str,
        method: str,
    ) -> tuple[str, ...]:

        if self.query_mode in (
            "neuron_bound",
            "component_bound",
            "neuron_pair_bound",
            "component_pair_bound",
        ):
            return ("none",)

        methods = allowed_error_methods(
            dim,
            method,
            context=self.query_mode,
        )

        if self.query_mode != "session_series":
            return methods

        stat_def = self.current_stat_def()

        neuron_dims = [d for d in stat_def.dims if d in NEURON_DIMS]

        # With neuron-pair statistics only the LAST neuron reduction
        # should generate the error estimate.
        if dim in neuron_dims and len(neuron_dims) > 1:
            if dim != neuron_dims[-1]:
                return ("none",)

        return methods


def default_session_series_reductions(stat_def) -> dict[str, ReductionSpec]:
    session_dims = [dim for dim in stat_def.dims if dim in SESSION_DIMS]
    neuron_dims = [dim for dim in stat_def.dims if dim in NEURON_DIMS]

    if not session_dims:
        return {}

    keep_session_dim = "session" if "session" in session_dims else session_dims[0]

    # For now: only one error-producing neuron dimension.
    error_neuron_dim = neuron_dims[-1] if neuron_dims else None

    reductions = {}

    for dim in stat_def.dims:
        if dim == keep_session_dim:
            reductions[dim] = ReductionSpec("keep")

        elif dim in SESSION_DIMS:
            reductions[dim] = ReductionSpec("single", index=0)

        elif dim in NEURON_DIMS:
            reductions[dim] = ReductionSpec(
                "median",
                error_method="iqr" if dim == error_neuron_dim else "none",
            )
        else:
            reductions[dim] = ReductionSpec("single", index=0)

    return reductions
