"""Named places of the GUI the starter guide can point at.

An anchor resolves lazily (widgets are rebuilt, rows come and go) to a
widget, a list of widgets (their union), or ``(widget, QRect)`` for an
area inside a widget (widget coordinates). Unavailable or hidden anchors
resolve to None; the guide then shows its bubble without a cut-out.
"""

from PySide6.QtCore import QPoint, QRect

# All anchors the tour may reference (validated by tests).
KNOWN_ANCHORS = (
    "main_menu",
    "display_area",
    "panel.mode_selector",
    "panel.split_buttons",
    "panel.help_button",
    "tasks",
    "resources",
    "guide.button",
    "load.root_folder",
    "load.add_row",
    "load.save_button",
    "sessions.list",
    "sessions.first_row",
    "sessions.first_row.config",
    "sessions.first_row.process",
    # Main menu: model, registration, review counts.
    "menu.model_row",
    "menu.assignments_row",
    "menu.registration_parameters",
    "menu.review_counts",
    # Neuron navigation (below the panels).
    "navigation_bar",
    "navigation.stepper",
    "navigation.radius",
    # Panels (the largest visible panel showing that view) and their parts.
    "panel.alignment",
    "alignment.correlation_inset",
    "alignment.shift_controls",
    "alignment.settings",
    "alignment.apply",
    "alignment.update_stale",
    "alignment.self_view_bar",
    "alignment.shift_inset",
    "alignment.apply_reset",
    "panel.model",
    "model.options",
    "model.parameters",
    "model.distance_parameters",
    "model.correlation_parameters",
    "model.correlation_histogram",
    "model.distance_histogram",
    "model.joint",
    "model.joint_controls",
    "menu.model_selector",
    "menu.model_process",
    "panel.overview",
    "overview.options",
    "overview.statistic",
    "panel.footprints",
    "footprints.options",
    "footprints.review_tag",
    "panel.selection_table",
    "selection.header",
    "selection.mode",
    "panel.statistics",
    "statistics.selectors",
    "statistics.x_selector",
    "statistics.y_selector",
    "statistics.reductions",
    "statistics.pair_filters",
    "statistics.options",
    "panel.curation",
    "curation.select_button",
    "curation.inspection",
    "curation.preset_selector",
    "curation.review_filter",
    "curation.evaluate",
    "curation.results",
    "curation.actions",
    # The curator's filter: the tree, the root's "AND by …", a count, an
    # "applied to", a condition, the preset menu.
    "curation.filter_tree",
    "curation.group_header",
    "curation.count",
    "curation.applied_to",
    "curation.condition",
    "curation.preset_menu",
    # Comparing assignment sets.
    "menu.assignments_selector",
    "footprints.assignments_selector",
    "statistics.comparison",
    "plot.export_button",
)


class AnchorRegistry:
    def __init__(self):
        self._targets = {}

    def register(self, name, target):
        """``target``: a widget or a callable returning the above."""
        if name not in KNOWN_ANCHORS:
            raise ValueError(f"Unknown guide anchor: {name!r}")

        self._targets[name] = target

    def names(self):
        return tuple(self._targets)

    def widgets(self, name):
        """The anchor's widgets (resolved; areas give their widget)."""
        target = self._targets.get(name)
        if callable(target) and not hasattr(target, "isVisible"):
            try:
                target = target()
            except (AttributeError, IndexError, KeyError, RuntimeError):
                return []
        if target is None:
            return []
        if isinstance(target, tuple):
            return [target[0]]
        if isinstance(target, list):
            return [w for w in target if w is not None]
        return [target]

    def rect(self, name, window):
        """The anchor's visible area in ``window`` coordinates, or None."""
        rects = self.rects(name, window)
        if not rects:
            return None
        union = rects[0]
        for r in rects[1:]:
            union = union.united(r)
        return union

    def rects(self, name, window, join=30):
        """The anchor's visible parts (window coordinates); parts closer
        than ``join`` px are combined (e.g. a caption and its editor)."""
        target = self._targets.get(name)

        if target is None:
            return []

        if callable(target) and not hasattr(target, "isVisible"):
            try:
                target = target()
            except (AttributeError, IndexError, KeyError, RuntimeError):
                return []

        if target is None:
            return []

        if isinstance(target, tuple):
            widget, area = target
            rects = [_widget_rect(widget, window, area)]
        elif isinstance(target, list):
            rects = [_widget_rect(w, window) for w in target]
        else:
            rects = [_widget_rect(target, window)]

        return _join_rects([r for r in rects if r is not None], join)


def _join_rects(rects, join):
    rects = list(rects)
    merged = True
    while merged:
        merged = False
        for i in range(len(rects)):
            for j in range(i + 1, len(rects)):
                grown = rects[i].adjusted(-join, -join, join, join)
                if grown.intersects(rects[j]):
                    rects[i] = rects[i].united(rects.pop(j))
                    merged = True
                    break
            if merged:
                break
    return rects


def _widget_rect(widget, window, area=None):
    from shiboken6 import isValid

    if widget is None or not isValid(widget) or not widget.isVisible():
        return None

    if widget.window() is not window:
        return None

    area = widget.rect() if area is None else QRect(area)
    top_left = widget.mapTo(window, area.topLeft())
    rect = QRect(top_left, area.size())

    # Clip to the visible part of scrolled / partly hidden widgets.
    parent = widget.parentWidget()
    while parent is not None and parent is not window:
        visible = QRect(parent.mapTo(window, QPoint(0, 0)), parent.size())
        rect = rect.intersected(visible)
        parent = parent.parentWidget()

    return rect if rect.isValid() and not rect.isEmpty() else None


def register_guide_anchor(state, name, target):
    state.guide_anchors.register(name, target)


def panel_section(display_area, mode):
    """The largest visible panel showing ``mode`` (None if none)."""
    from catan.gui.GUI_elements.display_section import DisplaySection

    sections = [
        s
        for s in display_area.findChildren(DisplaySection)
        if s.isVisible() and s.display_mode == mode
    ]
    if not sections:
        return None
    return max(sections, key=lambda s: s.width() * s.height())


def panel_controller(display_area, mode):
    section = panel_section(display_area, mode)
    return None if section is None else section.active_controller


def alignment_editor(display_area):
    controller = panel_controller(display_area, "session_alignment")
    display = getattr(controller, "display", None)
    editor = getattr(display, "editor", None)
    if editor is None or editor.isHidden():
        return None
    return editor


def install_main_window_anchors(window):
    """Anchors of the main window's fixed elements (resolved lazily)."""
    from catan.gui.GUI_elements.display_section import DisplaySection

    state = window.state
    menu = window.gui_elements["main_menu"]
    display_area = window.gui_elements["display_area"]
    load_row = menu.session_list.load_row

    def first_section():
        sections = [s for s in display_area.findChildren(DisplaySection) if s.isVisible()]
        return sections[0] if sections else None

    def first_row():
        rows = menu.session_list._row_widgets
        if not rows:
            return None
        return rows[min(rows)]

    def help_button():
        # The first panel that has one (empty panels do not).
        for section in display_area.findChildren(DisplaySection):
            button = getattr(section, "btn_help", None)
            if section.isVisible() and button is not None and button.isVisible():
                return button
        return None

    def split_buttons():
        section = first_section()
        if section is None:
            return None
        return [section.btn_split_h, section.btn_split_v, section.btn_close]

    anchors = {
        "main_menu": menu,
        "display_area": display_area,
        "panel.mode_selector": lambda: getattr(first_section(), "display_mode_selector", None),
        "panel.split_buttons": split_buttons,
        "panel.help_button": help_button,
        "tasks": menu.task_overview,
        "resources": lambda: menu.resource_monitor,
        "guide.button": lambda: menu.guide_button,
        "load.root_folder": lambda: menu.root_selector,
        "load.add_row": lambda: [
            w
            for w in (
                load_row.load_button,
                load_row.selector_load_mode,
                load_row.edit_load_glob,
            )
            if w.isVisible()
        ],
        "load.save_button": load_row.button_save_sessions,
        "sessions.list": menu.session_list.list_widget,
        "sessions.first_row": first_row,
        "sessions.first_row.config": lambda: first_row().config_constructor.toggle_config_options,
        "sessions.first_row.process": lambda: first_row().load_fields_button,
    }

    loader = menu.loader

    def controller(mode):
        return panel_controller(display_area, mode)

    def open_options(control):
        """A ControlPanel's ⚙ button; while its options are open, the whole
        options panel (framed, not dimmed, kept clear of the bubble) — the
        same for every panel's options."""
        button = getattr(control, "toggle_button", None)
        if button is not None and button.isChecked() and control.isVisible():
            return control
        return button

    def options_button(mode, key):
        return open_options(getattr(controller(mode), "controls", {}).get(key))

    def footprint_controllers():
        sections = [
            s
            for s in display_area.findChildren(DisplaySection)
            if s.isVisible() and s.display_mode == "footprints" and s.active_controller is not None
        ]
        sections.sort(key=lambda s: s.mapTo(window, s.rect().topLeft()).x())
        return [s.active_controller for s in sections]

    def editor_part(name):
        return lambda: getattr(alignment_editor(display_area), name, None)

    def curator_part(name):
        return lambda: getattr(getattr(controller("match_curator"), "menu", None), name, None)

    def curator_groups():
        """The filter's group widgets, root first (tree order)."""
        from catan.gui.GUI_elements.fragments.curation_filter_group import (
            CurationFilterGroupWidget,
        )

        root = curator_part("root_widget")()
        if root is None:
            return []
        return [root, *root.findChildren(CurationFilterGroupWidget)]

    def curator_group_header():
        groups = curator_groups()
        if not groups:
            return None
        root = groups[0]
        return [root.operator_selector, root.match_level_selector]

    def curator_count():
        """The first group that counts (else the root's Count box)."""
        groups = curator_groups()
        for group in groups:
            if group.group.count is not None:
                return [
                    group.count_enabled,
                    group.count_per_label,
                    group.count_anchor if group.count_anchor.isVisible() else None,
                    group.count_target,
                    group.count_comparison,
                    group.count_value,
                    group.count_matches_label if group.count_matches_label.isVisible() else None,
                    group.count_unit_row if group.count_unit_row.isVisible() else None,
                ]
        return [groups[0].count_enabled] if groups else None

    def curator_applied_to():
        """The first subgroup applied to sources / targets / between."""
        groups = [g for g in curator_groups() if not g.is_root]
        for group in groups:
            if group.group.apply_to != "same":
                return group.binding_selector
        return groups[0].binding_selector if groups else None

    def curator_condition():
        """A condition with a threshold from the model, else one with a
        percentile, else the first."""
        from catan.gui.GUI_elements.fragments.curation_filter_chip import (
            CurationFilterChip,
        )

        groups = curator_groups()
        if not groups:
            return None
        chips = groups[0].findChildren(CurationFilterChip)
        conditions = {}

        def collect(node):
            for child in getattr(node, "children", ()):
                if getattr(child, "threshold", None) is not None:
                    conditions[child.id] = child
                collect(child)

        collect(groups[0].group)

        def kind(chip):
            reference = getattr(
                getattr(conditions.get(chip.condition_id), "threshold", None), "reference", None
            )
            return None if reference is None else type(reference).__name__

        for wanted in ("ModelDistanceReference", "PercentileReference"):
            for chip in chips:
                if kind(chip) == wanted:
                    return chip
        return chips[0] if chips else None

    def export_button():
        from catan.gui.GUI_elements.fragments.SaveDisplayButton import (
            SaveDisplayButton,
        )

        # The overview's (the panel the step shows), else any.
        overview = getattr(controller("tracked_overview"), "canvas", None)
        own = getattr(overview, "save_display_button", None)
        if own is not None and own.isVisible():
            return own
        buttons = [b for b in display_area.findChildren(SaveDisplayButton) if b.isVisible()]
        return buttons[0] if buttons else None

    def visible_or(primary, fallback):
        """``primary`` while it is shown in the window, else ``fallback``
        (both callables)."""

        def resolve():
            try:
                widget = primary()
            except (AttributeError, IndexError, KeyError, RuntimeError):
                widget = None
            if widget is not None and _widget_rect(widget, window) is not None:
                return widget
            return fallback()

        return resolve

    navigation = window.gui_elements["navigation_bar"]

    def model_display():
        return controller("model_results").display_widget

    def model_card(index):
        # Cards: correlation (top right), distance (bottom left), joint 2D,
        # joint 3D (bottom right); built once evidence is shown.
        return lambda: model_display().cards[index]

    def model_joint():
        cards = model_display().cards
        return next((c for c in cards[2:4] if c.isVisible()), None)

    def model_editors(keys, card):
        """Parameter editors with their captions, and a histogram."""

        def resolve():
            display = model_display()
            widgets = []
            for key in keys:
                editor = display.editors.get(key)
                if editor is None:
                    continue
                widgets.append(editor)
                grid = editor.parentWidget().layout()
                row, column, _, _ = grid.getItemPosition(grid.indexOf(editor))
                item = grid.itemAtPosition(row, column - 1)
                if item is not None and item.widget() is not None:
                    widgets.append(item.widget())
            if card is not None and len(display.cards) > card:
                widgets.append(display.cards[card])
            return widgets

        return resolve

    anchors.update(
        {
            "menu.model_row": lambda: [
                loader["model"]["selector"],
                loader["model"]["button_execute"],
            ],
            "menu.assignments_row": lambda: [
                loader["assignments"]["selector"],
                loader["assignments"]["button_execute"],
            ],
            "menu.registration_parameters": lambda: next(
                iter(menu.registration_spins.values())
            ).parentWidget(),
            "menu.review_counts": lambda: menu.review_counts_label,
            "navigation_bar": navigation,
            "navigation.stepper": lambda: [
                navigation.footprint_edit,
                navigation.footprint_id_prev,
                navigation.footprint_slider,
                navigation.footprint_id_next,
            ],
            "navigation.radius": lambda: navigation.adj_radius_spin,
            "panel.alignment": lambda: panel_section(display_area, "session_alignment"),
            "alignment.correlation_inset": editor_part("correlation_inset"),
            "alignment.shift_controls": editor_part("shift_controls"),
            "alignment.settings": lambda: open_options(
                getattr(alignment_editor(display_area), "options", None)
            ),
            "alignment.apply": editor_part("apply_button"),
            "alignment.update_stale": lambda: getattr(
                getattr(controller("session_alignment"), "display", None),
                "update_stale",
                None,
            ),
            "alignment.self_view_bar": editor_part("self_view_bar"),
            # Until the inset is switched: the correlation raster in its place.
            "alignment.shift_inset": visible_or(
                editor_part("shift_inset"), editor_part("correlation_inset")
            ),
            "alignment.apply_reset": lambda: [
                w
                for w in (
                    editor_part("apply_button")(),
                    editor_part("reset_button")(),
                )
                if w is not None
            ],
            "panel.model": lambda: panel_section(display_area, "model_results"),
            "model.options": lambda: controller("model_results").display_widget.options_button,
            "model.parameters": lambda: controller("model_results").display_widget.launcher,
            "model.distance_parameters": model_editors(
                ("p_same", "h", "sigma_h", "sigma_det", "sigma_move", "nu_move"), 1
            ),
            "model.correlation_parameters": model_editors(
                ("u_same_mean", "u_same_shape", "u_same_slope", "u_diff_mean", "u_diff_shape"),
                0,
            ),
            "model.correlation_histogram": model_card(0),
            "model.distance_histogram": model_card(1),
            "model.joint": model_joint,
            "model.joint_controls": lambda: model_display().joint_controls,
            "menu.model_selector": lambda: loader["model"]["selector"],
            "menu.model_process": lambda: loader["model"]["button_execute"],
            "panel.overview": lambda: panel_section(display_area, "tracked_overview"),
            "overview.options": lambda: options_button("tracked_overview", "panel"),
            "overview.statistic": lambda: [
                w
                for w in (
                    controller("tracked_overview").controls.get(key)
                    for key in ("statistics", "stat_cmap_label", "stat_cmap", "stat_reverse")
                )
                if w is not None
            ],
            "panel.footprints": lambda: panel_section(display_area, "footprints"),
            "footprints.options": lambda: options_button("footprints", "parameter"),
            # Shown once a neuron is focused.
            "footprints.review_tag": visible_or(
                lambda: controller("footprints").canvas.review_status_label,
                lambda: panel_section(display_area, "footprints"),
            ),
            "panel.selection_table": lambda: panel_section(display_area, "selection_display"),
            "selection.header": lambda: controller(
                "selection_display"
            ).table.stats_table.horizontalHeader(),
            "selection.mode": lambda: [
                controller("selection_display").entity_selector,
                controller("selection_display").row_selector,
            ],
            "panel.statistics": lambda: panel_section(display_area, "statistics"),
            "statistics.selectors": lambda: [
                controller("statistics").controls["x_selector"],
                controller("statistics").controls["y_selector"],
            ],
            "statistics.x_selector": lambda: controller("statistics").controls["x_selector"],
            "statistics.y_selector": lambda: controller("statistics").controls["y_selector"],
            "statistics.reductions": lambda: [
                controller("statistics").controls[f"{slot}_selector"].reduction_button
                for slot in ("x", "y")
            ],
            # Shown for pair statistics only; else the selectors.
            "statistics.pair_filters": lambda: [
                getattr(controller("statistics").controls[f"{slot}_selector"], button)
                for slot in ("x", "y")
                for button in ("neuron_filter_button", "session_filter_button")
                if getattr(
                    controller("statistics").controls[f"{slot}_selector"], button
                ).isVisible()
            ]
            or [
                controller("statistics").controls["x_selector"],
                controller("statistics").controls["y_selector"],
            ],
            "statistics.options": lambda: options_button("statistics", "panel"),
            "panel.curation": lambda: panel_section(display_area, "match_curator"),
            "curation.select_button": curator_part("select_button"),
            "curation.inspection": curator_part("inspection_row"),
            "curation.preset_selector": curator_part("preset_selector"),
            "curation.review_filter": curator_part("review_filter"),
            "curation.evaluate": curator_part("evaluate_button"),
            "curation.results": lambda: [
                w
                for w in (
                    curator_part("result_label")(),
                    curator_part("status_label")(),
                )
                if w is not None
            ],
            "curation.actions": lambda: getattr(
                controller("match_curator"), "actions", None
            ),
            "curation.filter_tree": curator_part("scroll"),
            "curation.group_header": curator_group_header,
            "curation.count": lambda: [w for w in curator_count() or () if w is not None],
            "curation.applied_to": curator_applied_to,
            "curation.condition": curator_condition,
            "curation.preset_menu": lambda: [
                curator_part("preset_selector")(),
                curator_part("preset_menu_button")(),
            ],
            "menu.assignments_selector": lambda: loader["assignments"]["selector"],
            "footprints.assignments_selector": lambda: [
                c.assignments_selector
                for c in footprint_controllers()
                if c.assignments_selector.isVisible()
            ][-1:],
            "statistics.comparison": lambda: open_options(
                controller("statistics").controls["panel"]
            ),
            "plot.export_button": export_button,
        }
    )

    for name, target in anchors.items():
        register_guide_anchor(state, name, target)
