import json
from importlib.resources import files

import numpy as np
from scipy import sparse

import catan.gui.GUI_elements  # noqa: F401  (import order, see CLAUDE.md)
from catan.gui.data.curation_filter import curation_filter_from_dict
from catan.gui.data.statistics.calculations import footprint_axes


def _ellipse(d1, d2, length, width, angle):
    yy, xx = np.mgrid[0:d1, 0:d2]
    t = np.deg2rad(angle)
    u = (xx - d2 / 2) * np.cos(t) + (yy - d1 / 2) * np.sin(t)
    v = -(xx - d2 / 2) * np.sin(t) + (yy - d1 / 2) * np.cos(t)
    return np.exp(-0.5 * ((u / (length / 2)) ** 2 + (v / (width / 2)) ** 2)).ravel()


def test_axis_ratio_of_synthetic_ellipses():
    d1, d2 = 60, 80
    cases = [(8, 8, 0, 1.0), (15, 3, 120, 5.0), (12, 4, 0, 3.0)]
    footprints = sparse.csc_matrix(
        np.array([_ellipse(d1, d2, a, b, angle) for a, b, angle, _ in cases]).T
    )

    major, minor = footprint_axes(footprints, (d1, d2), 0.2)
    ratio = np.sqrt(major / minor)

    np.testing.assert_allclose(ratio, [case[-1] for case in cases], rtol=0.05)


def test_axes_need_three_pixels():
    footprints = sparse.csc_matrix(np.eye(100, 2))  # one pixel each
    major, minor = footprint_axes(footprints, (10, 10), 0.2)

    assert np.isnan(major).all() and np.isnan(minor).all()


def test_shipped_presets_load():
    folder = files("catan.resources.curation_filters")
    stems = [r.name[:-5] for r in folder.iterdir() if r.name.endswith(".json")]

    assert "cleanup_edge_artifacts" in stems and "structure_orphans" in stems

    for stem in stems:
        document = json.loads(folder.joinpath(f"{stem}.json").read_text("utf-8"))
        root = curation_filter_from_dict(document)

        assert root.children, stem


def test_unavailable_optional_conditions_are_pruned():
    from catan.gui.data.curation_filter import (
        CurationFilterCondition,
        CurationFilterGroup,
        curation_filter_node_from_dict,
        curation_filter_node_to_dict,
        prune_unavailable_optional,
    )
    from catan.gui.data.statistics.queries import StatisticQuery
    from catan.gui.panels.helper.Threshold import ThresholdSpec

    def condition(key, optional):
        return CurationFilterCondition(
            StatisticQuery(key, ()),
            ThresholdSpec(value=1.0, direction="less", active=True),
            optional=optional,
        )

    only_optional = CurationFilterGroup(children=[condition("missing", True)])
    root = CurationFilterGroup(
        children=[
            condition("present", False),
            condition("missing", True),
            condition("missing_required", False),
            only_optional,
        ]
    )

    removed = prune_unavailable_optional(root, {"present": object()})

    assert [c.query.statistic_key for c in root.children] == ["present", "missing_required"]
    assert only_optional.id in removed and len(removed) == 3

    restored = curation_filter_node_from_dict(curation_filter_node_to_dict(condition("x", True)))
    assert restored.optional
    assert "optional" not in curation_filter_node_to_dict(condition("x", False))



def test_excluded_footprints_are_no_detections():
    from types import SimpleNamespace

    from catan.gui.data.statistics.session_relations import (
        DetectionReference,
        included_detections,
    )

    # One neuron in sessions 0..3; its footprint in session 1 is excluded.
    ids = np.array([[0, 1, 0, 2]])
    sessions = [
        SimpleNamespace(included=np.array([True])),
        SimpleNamespace(included=np.array([True, False])),
        None,
        SimpleNamespace(included=None),
    ]

    detected = included_detections(sessions, ids)
    np.testing.assert_array_equal(detected, [[0, -1, 0, 2]])
    np.testing.assert_array_equal(ids, [[0, 1, 0, 2]])  # input unchanged

    reference = DetectionReference(detected)
    np.testing.assert_array_equal(reference.reference("previous instance", [0, 0], [2, 3]), [0, 2])
    np.testing.assert_array_equal(reference.reference("next instance", [0], [0]), [2])


def test_strict_thresholds_round_trip():
    from catan.gui.data.curation_filter import (
        CurationFilterCondition,
        curation_filter_node_from_dict,
        curation_filter_node_to_dict,
    )
    from catan.gui.data.statistics.queries import StatisticQuery
    from catan.gui.panels.helper.Threshold import ThresholdSpec

    values = np.array([0.0, 0.5, 1.0])
    for direction, strict, expected in (
        ("greater", False, [False, True, True]),
        ("greater", True, [False, False, True]),
        ("less", False, [True, True, False]),
        ("less", True, [True, False, False]),
    ):
        spec = ThresholdSpec(value=0.5, direction=direction, active=True, strict=strict)
        np.testing.assert_array_equal(spec.mask(values), expected)

        condition = CurationFilterCondition(StatisticQuery("x", ()), spec)
        restored = curation_filter_node_from_dict(curation_filter_node_to_dict(condition))
        assert restored.threshold.strict == strict

    assert ThresholdSpec(value=0, direction="greater", active=True, strict=True).expression() == "value > 0"
