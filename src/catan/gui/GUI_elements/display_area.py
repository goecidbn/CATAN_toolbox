import json, uuid
import importlib

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QWidget,
    QVBoxLayout,
    QSplitter,
)

from catan.gui.panels import BasePlot
from catan.gui.structures import Data, AppState

from .display_section import DisplaySection


def new_id():
    return uuid.uuid4().hex[:8]


def leaf_node(config=None):
    return {
        "type": "leaf",
        "id": new_id(),
        "config": config
        or {
            "display_mode": "empty",
        },
    }


def split_node(orientation, a, b):
    return {
        "type": "split",
        "id": new_id(),
        "orientation": orientation,  # "horizontal" or "vertical"
        "children": [a, b],
        "sizes": None,
    }


class DisplayArea(QWidget):
    MAX_SECTIONS = 10

    def __init__(self, parent):
        super().__init__(parent)
        self.state: AppState = parent.state
        self.data: Data = parent.data

        self.splitter_widgets = {}  # split_id -> QSplitter
        self.section_widgets = {}  # leaf_id -> DisplaySection

        self.root_layout = QVBoxLayout(self)
        self.root_layout.setContentsMargins(0, 0, 0, 0)

        self.tree = self._load_tree()
        self.rebuild()

    # ---------- public ----------
    def _save_settings(self):
        self._collect_live_leaf_configs(self.tree)
        self._collect_splitter_sizes(self.tree)

        self.state.settings.setValue("display/layout_tree", json.dumps(self.tree))
        self.state.settings.sync()

    def _collect_live_leaf_configs(self, node):
        if node["type"] == "leaf":
            w = self.section_widgets.get(node["id"])
            if w is not None:
                node["config"] = w.get_config()
            return

        for child in node["children"]:
            self._collect_live_leaf_configs(child)

    # ---------- build / rebuild ----------

    def rebuild(
        self,
        *,
        reload_alignment=False,
        reload_statistics=False,
    ):
        self._clear_layout()
        self.splitter_widgets.clear()
        self.section_widgets.clear()

        if reload_statistics:
            self._reload_statistics_logic()
        
        importlib.reload(BasePlot)

        if reload_alignment:
            from catan.core import image_correlation
            from catan.gui.GUI_elements.fragments import (
                alignment_shift_inset,
                manual_alignment_dialog,
            )

            importlib.reload(image_correlation)
            importlib.reload(alignment_shift_inset)
            importlib.reload(manual_alignment_dialog)

        widget = self._build_node(self.tree)
        self.root_layout.addWidget(widget)

        self._update_section_buttons()

    def _reload_statistics_logic(self):
        importlib.invalidate_caches()

        queries = importlib.import_module(
            "catan.gui.data.statistics.queries"
        )

        # Preserve identities used by existing queries, imported aliases,
        # and dictionary/cache keys.
        existing_classes = {
            name: getattr(queries, name)
            for name in (
                "ReductionSpec",
                "PairFilter",
                "StatisticQuery",
            )
        }

        importlib.reload(queries)

        for name, cls in existing_classes.items():
            setattr(queries, name, cls)

        # Reload consumers after their dependencies, refreshing
        # functions imported through `from ... import ...`.
        for module_name in (
            "catan.gui.data.statistics.tabledata",
            "catan.gui.data.statistics.plotdata_histogram",
            "catan.gui.data.statistics.plotdata_scatter",
            "catan.gui.data.statistics.plotdata_series",
            "catan.gui.panels.StatisticsData",
            "catan.gui.panels.helper.series_with_confidence",
        ):
            importlib.reload(importlib.import_module(module_name))

        self.data.statistic_engine.clear_cache()
        
    def _build_node(self, node):
        if node["type"] == "leaf":
            # print(f"Building leaf node {node['id']} with config {node.get('config')}")
            w = DisplaySection(
                self,
                section_id=node["id"],
                display_mode=node.get("config", {}).get("display_mode", "empty"),
            )

            w.apply_config(node.get("config", {}))

            w.requestSplit.connect(self.split_section)
            w.requestClose.connect(self.close_section)

            self.section_widgets[node["id"]] = w
            return w

        orientation = (
            Qt.Horizontal if node["orientation"] == "horizontal" else Qt.Vertical
        )

        splitter = QSplitter(orientation)
        splitter.setChildrenCollapsible(False)

        self.splitter_widgets[node["id"]] = splitter

        for child in node["children"]:
            splitter.addWidget(self._build_node(child))

        if node.get("sizes"):
            splitter.setSizes(node["sizes"])

        return splitter

    def _clear_layout(self):
        # Disconnect signals and dispose canvases before reloading modules.
        for section in tuple(self.section_widgets.values()):
            controller = section.active_controller
            if controller is not None:
                controller.deactivate()
                section.active_controller = None

        while self.root_layout.count():
            item = self.root_layout.takeAt(0)
            widget = item.widget()
            if widget is not None:
                widget.deleteLater()

    # ---------- split / close ----------

    def split_section(self, section_id: str, orientation: str):
        if self._count_leaves(self.tree) >= self.MAX_SECTIONS:
            return

        target = self._find_node(self.tree, section_id)
        if target is None:
            return

        # Capture all panels before changing the tree or destroying widgets.
        self._collect_live_leaf_configs(self.tree)
        self._collect_splitter_sizes(self.tree)

        # Get live config from the actual existing widget
        section_widget = self.section_widgets.get(section_id)

        if section_widget is not None:
            current_config = section_widget.get_config()
        else:
            current_config = target.get("config", {"display_mode": "empty"})

        # print(f"Splitting section {section_id} with orientation {orientation} and config {current_config}")
        old_leaf = leaf_node(config=current_config)
        old_leaf["id"] = section_id  # keep original id for left/top

        new_leaf = leaf_node()  # new right/bottom starts empty

        target.clear()
        target.update(split_node(orientation, old_leaf, new_leaf))

        self.rebuild()

    def close_section(self, section_id: str):
        if self._count_leaves(self.tree) <= 1:
            return

        self._collect_live_leaf_configs(self.tree)
        self._collect_splitter_sizes(self.tree)

        self.tree = self._remove_leaf(self.tree, section_id)
        self.rebuild()

    # ---------- tree utilities ----------

    def _count_leaves(self, node):
        if node["type"] == "leaf":
            return 1
        return sum(self._count_leaves(c) for c in node["children"])

    def _find_node(self, node, node_id):
        if node["id"] == node_id:
            return node

        if node["type"] == "split":
            for child in node["children"]:
                result = self._find_node(child, node_id)
                if result is not None:
                    return result

        return None

    def _remove_leaf(self, node, leaf_id):
        """
        Remove leaf. If a split loses one child, replace the split by its sibling.
        """
        if node["type"] == "leaf":
            return None if node["id"] == leaf_id else node

        new_children = []
        for child in node["children"]:
            result = self._remove_leaf(child, leaf_id)
            if result is not None:
                new_children.append(result)

        if len(new_children) == 1:
            return new_children[0]

        node["children"] = new_children
        return node

    # ---------- settings ----------

    def _load_tree(self):
        raw = self.state.settings.value("display/layout_tree", "", type=str)
        # raw = None
        # print("loading:", raw)
        if not raw:
            return leaf_node()

        try:
            tree = json.loads(raw)
            if self._count_leaves(tree) > self.MAX_SECTIONS:
                return leaf_node()
            return tree
        except Exception:
            return leaf_node()

    def _collect_splitter_sizes(self, node):
        if node["type"] != "split":
            return

        splitter = self.splitter_widgets.get(node["id"])
        if splitter is not None:
            node["sizes"] = splitter.sizes()

        for child in node["children"]:
            self._collect_splitter_sizes(child)

    def _update_section_buttons(self):
        n = self._count_leaves(self.tree)
        can_split = n < self.MAX_SECTIONS
        can_close = n > 1

        for section in self.section_widgets.values():
            section.set_split_options_enabled(
                can_split=can_split,
                can_close=can_close,
            )
