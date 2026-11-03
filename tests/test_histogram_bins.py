"""Integer-centred histogram bins for integer-valued statistics."""

import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np
import pytest

# GUI package tests: need PySide6 (the headless CI job runs -m "not gui").
pytest.importorskip("PySide6")
pytestmark = pytest.mark.gui

# Import in app order (known cycle via panels.helper.Threshold).
import catan.gui.GUI_elements.main_window  # noqa: F401,E402
from catan.gui.data.statistics.dimensions import Dimension
from catan.gui.data.statistics.plotdata_histogram import (
    histogram_edges,
    integer_bin_edges,
    integer_bin_label,
    possible_value_range,
)


def test_integer_bins_are_centred_on_integers():
    edges, step = integer_bin_edges(0, 11, max_bins=30)
    assert step == 1
    assert edges[0] == -0.5 and edges[-1] == 11.5
    assert np.allclose(np.diff(edges), 1.0)

    counts, _ = np.histogram([0, 3, 3, 11], bins=edges)
    assert counts[0] == 1 and counts[3] == 2 and counts[11] == 1


def test_wide_integer_ranges_use_nice_steps():
    edges, step = integer_bin_edges(17, 412, max_bins=30)
    assert step == 20
    assert edges[0] == -0.5  # bins start at multiples of the step
    assert edges[-1] >= 412
    assert len(edges) - 1 <= 30
    assert integer_bin_label(edges[1], edges[2]) == "values 20 – 39"
    assert integer_bin_label(2.5, 3.5) == "value 3"


def test_histogram_edges_detects_integers_and_applies_bounds():
    edges, step = histogram_edges(np.array([2.0, 5.0, np.nan]), 30, bounds=(0, 11))
    assert step == 1 and edges[0] == -0.5 and edges[-1] == 11.5

    edges, step = histogram_edges(np.array([0.2, 0.7]), 10)
    assert step is None and len(edges) == 11


class _Definition:
    value_bounds = (0, 1)


class _Stat:
    def __init__(self, method):
        self.dimensions = {
            "neuron": Dimension(name="neuron", coords=np.arange(5)),
            "session": Dimension(
                name="session",
                coords=np.arange(12),
                parameter={"method": method, "error_method": "none"},
                mode="reduced",
            ),
        }


def test_possible_range_follows_reductions():
    assert possible_value_range(_Definition, _Stat("sum"), n_sessions=11) == (0.0, 11.0)
    assert possible_value_range(_Definition, _Stat("mean"), n_sessions=11) == (0.0, 1.0)
    assert possible_value_range(_Definition, _Stat("std"), n_sessions=11) is None
    assert possible_value_range(object(), _Stat("sum"), n_sessions=11) is None
