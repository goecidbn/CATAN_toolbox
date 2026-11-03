import json
import os
from importlib.resources import files

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest
# GUI package tests: need PySide6 (the headless CI job runs -m "not gui").
pytest.importorskip("PySide6")
pytestmark = pytest.mark.gui

from PySide6.QtWidgets import QApplication

import catan.gui.GUI_elements  # noqa: F401  (import order: avoids a cycle)
from catan.gui.data.curation_filter import CountSpec, curation_filter_from_dict
from catan.gui.GUI_elements.fragments.curation_filter_group import (
    CurationFilterGroupWidget,
)


@pytest.fixture(scope="module")
def app():
    return QApplication.instance() or QApplication([])


def _preset(name):
    text = (files("catan.resources.curation_filters") / f"{name}.json").read_text()
    root = curation_filter_from_dict(json.loads(text))
    return getattr(root, "root", root)


def _widget(root):
    return CurationFilterGroupWidget(
        root, format_condition=lambda c: (c.query.statistic_key, None), is_root=True
    )


def test_merge_preset_header_can_shrink(app):
    widget = _widget(_preset("footprint_manipulation_merge_candidates"))
    widget.show()
    app.processEvents()
    # Was 770 px (110 average characters) with all selectors on one header
    # line. Measured in characters: fonts differ per platform (CI "minimal"
    # has no font database, ~2.3x wider glyphs).
    char_width = widget.fontMetrics().averageCharWidth()
    assert widget.minimumSizeHint().width() <= 64 * char_width
    widget.deleteLater()


def test_count_sentence_emits_spec(app):
    root = _preset("footprint_manipulation_merge_candidates")
    widget = _widget(root)
    specs = []
    widget.count_changed.connect(lambda _, spec: specs.append(spec))

    widget.count_value.setValue(3)
    assert specs[-1] == CountSpec(
        target="footprint",
        comparison="eq",
        value=3,
        unit="footprint",
        anchor="source",
        separate_partner_sessions=True,
    )

    widget.count_enabled.setChecked(False)
    assert specs[-1] is None
    assert not widget.count_partner_sessions.isEnabled()
    widget.deleteLater()


def test_binding_and_summary_selectors(app):
    root = _preset("footprint_manipulation_merge_candidates")
    widget = _widget(root)
    between = next(
        w for w in widget._group_widgets.values()
        if w.group.apply_to == "between_targets"
    )
    assert between.binding_selector.currentText() == "between targets"
    assert not between.summary_selector.isHidden()
    assert not between.summary_label.isHidden()

    bindings = []
    widget.binding_changed.connect(lambda _, binding: bindings.append(binding))
    between.binding_selector.setCurrentIndex(between.binding_selector.findData("same"))
    assert bindings == ["same"]
    assert between.summary_selector.isHidden() and between.summary_label.isHidden()
    widget.deleteLater()
