"""Compact parameter display and help texts of statistics."""

import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest

# GUI package tests: need PySide6 (the headless CI job runs -m "not gui").
pytest.importorskip("PySide6")
pytestmark = pytest.mark.gui

from catan.gui.data.statistics.definitions import CALCULATED_STATISTICS


def _overlap():
    return CALCULATED_STATISTICS["footprint_overlap"]


def test_title_uses_short_names():
    stat_def = _overlap()

    assert stat_def.title_with_parameters() == (
        "Footprint overlap [norm=smaller, thr=0.1]"
    )


def test_changed_only_lists_non_defaults():
    stat_def = _overlap()

    assert stat_def.parameters_short(changed_only=True) == ""
    assert (
        stat_def.parameters_short({"normalize_by": "j"}, changed_only=True)
        == "norm=j"
    )


def test_old_overlap_switches_migrate(caplog):
    stat_def = _overlap()
    old = {"containment": True, "support_threshold": 0.2, "weighted": False}
    cases = [
        ({"containment": False, "directional": False}, "IoU"),
        ({"containment": True, "directional": False}, "smaller"),
        ({"directional": True, "normalize_by_target": False}, "i"),
        ({"containment": False, "directional": True, "normalize_by_target": True}, "j"),
        ({}, "smaller"),
    ]
    for switches, expected in cases:
        values = stat_def.resolve_parameters({**old, **switches})
        assert values == {"normalize_by": expected, "support_threshold": 0.2}

    union = CALCULATED_STATISTICS["footprint_union_overlap"]
    assert union.resolve_parameters({"directional": True})["normalize_by"] == "i"

    # Intensity-weighted overlaps are binary now: warned, not an error.
    with caplog.at_level("WARNING"):
        assert stat_def.resolve_parameters({"weighted": True}) == {
            "normalize_by": "smaller",
            "support_threshold": 0.1,
        }


def test_overlap_normalization_checked():
    import pytest

    from catan.gui.data.statistics.overlap_statistics import (
        check_normalization,
        overlap_denominator,
    )

    with pytest.raises(ValueError):
        check_normalization("target")

    import numpy as np

    inter, ma, mb = np.array([2.0]), np.array([4.0]), np.array([8.0])
    assert overlap_denominator("IoU", inter, ma, mb) == 10
    assert overlap_denominator("smaller", inter, ma, mb) == 4
    assert overlap_denominator("i", inter, ma, mb) == 4
    assert overlap_denominator("j", inter, ma, mb) == 8


def test_every_parameter_has_a_description():
    for stat_def in CALCULATED_STATISTICS.values():
        for spec in stat_def.parameters:
            assert spec.tooltip, (stat_def.key, spec.key)
            help_text = spec.help_text()
            assert spec.label in help_text and "default" in help_text


def test_loading_options_not_in_title():
    stat_def = next(
        stat_def
        for stat_def in CALCULATED_STATISTICS.values()
        if any(spec.key == "load_all" for spec in stat_def.parameters)
    )

    title = stat_def.title_with_parameters()
    assert "load" not in title.lower() and "save" not in title.lower()


def test_wrap_text():
    from catan.gui.panels.StatisticsData import wrap_text

    class Metrics:
        @staticmethod
        def horizontalAdvance(text):
            return len(text)

    assert wrap_text("Footprint to union similarity", Metrics, 14) == (
        "Footprint to\nunion\nsimilarity"
    )
    assert wrap_text("Supercalifragilistic word", Metrics, 5) == (
        "Supercalifragilistic\nword"
    )
