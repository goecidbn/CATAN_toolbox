from importlib.resources import files

import pytest

# GUI package tests: need PySide6 (the headless CI job runs -m "not gui").
pytest.importorskip("PySide6")
pytestmark = pytest.mark.gui

from catan.gui.guide.anchors import KNOWN_ANCHORS
from catan.gui.guide.tour import ACTIONS, CONDITIONS, PANEL_MODES, load_tour, parse_tour


def test_shipped_tour_is_valid():
    tour = load_tour()

    assert tour.version >= 2
    assert tour.demo_recipe.endswith(".json")
    ids = [chapter.id for chapter in tour.chapters]
    assert len(ids) == len(set(ids)) == 11
    assert tour.demo_alternative_assignments.endswith(".assignments.hdf5")

    for chapter in tour.chapters:
        # The outline shows a description and a duration per chapter.
        assert chapter.ready and chapter.description and chapter.minutes > 0, chapter.id
        assert all(name in CONDITIONS for name in chapter.requires), chapter.id
        # Chapters that need earlier results can set them up.
        assert not chapter.requires or chapter.setup in ACTIONS, chapter.id

        for step in chapter.steps:
            assert step.title and step.text
            assert step.anchor is None or step.anchor in KNOWN_ANCHORS
            assert all(
                name in CONDITIONS
                for name in (step.wait_for, step.advance_when, *step.skip_if)
                if name
            )
            assert step.do_it is None or step.do_it in ACTIONS
            assert set(step.required_panels(chapter)) <= set(PANEL_MODES)
            # Removed from the alignment editor.
            assert "back to comparison" not in step.text.lower()

    # The first chapters work without earlier results.
    assert not tour.chapters[0].requires and not tour.chapters[1].requires


def _texts(chapters):
    return " ".join(s.title + " " + s.text for c in chapters for s in c.steps).lower()


def test_navigation_bar_waits_for_tracked_neurons():
    tour = load_tour()
    order = [c.id for c in tour.chapters]
    results = order.index("results")

    # Introduced once neurons are tracked (with the linked views), not before.
    assert order.index("registration") < results
    assert "navigation" not in _texts(tour.chapters[:results])
    assert "navigation bar" in _texts([tour.chapters[results]])
    assert "neurons_registered" in tour.chapters[results].requires


def test_inspection_chapters_use_linked_views():
    tour = load_tour()

    for chapter_id in ("results", "footprints", "statistics", "curation"):
        chapter = tour.chapter(chapter_id)
        assert len(chapter.panels) >= 2, chapter_id
        steps = [s for s in chapter.steps if s.panel is not None]
        assert steps and all(len(s.required_panels(chapter)) >= 2 for s in steps)
        # Several panels: the whole display area takes clicks.
        assert all(s.open_area(chapter) == "display_area" for s in steps if s.anchor)

    # Curation shows its matches in the footprints and the overview.
    assert {"footprints", "tracked_overview"} <= set(tour.chapter("curation").panels)
    # The overview is not replaced to show the footprints.
    results = tour.chapter("results")
    footprint_step = next(s for s in results.steps if s.panel == "footprints")
    assert "tracked_overview" in footprint_step.required_panels(results)


def test_panel_menu_has_one_level_below_plots():
    from catan.gui.GUI_elements.display_section import display_modes
    from catan.gui.guide.controller import _panel_menu_path

    assert _panel_menu_path("tracked_overview") == ("Plots", "Overview")
    assert _panel_menu_path("statistics") == ("Plots", "Statistics")
    assert _panel_menu_path("selection_display") == ("Selection table",)
    assert _panel_menu_path("match_curator") == ("Curation", "Neuron tracking")
    plots = [o for o in display_modes.values() if o.get("menu") == "Plots"]
    assert all(len(o["options"]) == 1 for o in plots)
    assert "raster" not in {m for o in display_modes.values() for m in o["options"]}


def test_selection_table_defines_hover_tooltips():
    results = load_tour().chapter("results")
    steps = [s for s in results.steps if s.panel == "selection_display"]

    assert any("tooltip" in s.text.lower() and "hover" in s.text.lower() for s in steps)


@pytest.mark.parametrize(
    "step, problem",
    [
        ({"title": "t", "text": "x", "anchor": "nowhere"}, "anchor"),
        ({"title": "t", "text": "x", "wait_for": "never"}, "condition"),
        ({"title": "t", "text": "x", "skip_if": ["never"]}, "condition"),
        ({"title": "t", "text": "x", "do_it": "magic"}, "action"),
        ({"title": "t", "text": "x", "panel": "nowhere"}, "panel"),
        ({"title": "t", "text": "x", "panels": ["footprints", "nowhere"]}, "panel"),
        ({"title": "t", "text": "x", "advance_when": "never"}, "condition"),
    ],
)
def test_unknown_names_are_rejected(step, problem):
    document = {"version": 1, "chapters": [{"id": "c", "title": "C", "steps": [step]}]}

    with pytest.raises(ValueError, match=problem):
        parse_tour(document)


def test_tour_file_is_packaged():
    assert files("catan.resources.guide").joinpath("tour.json").is_file()


def test_unknown_chapter_requirements_are_rejected():
    document = {
        "version": 1,
        "chapters": [{"id": "c", "title": "C", "requires": ["never"], "steps": []}],
    }

    with pytest.raises(ValueError, match="condition"):
        parse_tour(document)


def test_panel_modes_exist():
    from catan.gui.GUI_elements.display_section import DISPLAY_CONTROLLERS

    assert set(PANEL_MODES) <= set(DISPLAY_CONTROLLERS)


def test_interactive_area_defaults_and_validation():
    document = {
        "version": 1,
        "chapters": [
            {
                "id": "c",
                "title": "C",
                "steps": [
                    {"title": "a", "text": "x", "anchor": "alignment.apply", "panel": "session_alignment"},
                    {"title": "b", "text": "x", "anchor": "alignment.apply", "panel": "session_alignment", "interactive": "none"},
                    {"title": "c", "text": "x", "anchor": "tasks", "interactive": "main_menu"},
                    {"title": "d", "text": "x", "anchor": "tasks"},
                ],
            }
        ],
    }
    a, b, c, d = parse_tour(document).chapters[0].steps

    # Steps of a panel leave the whole panel open for exploring.
    assert a.open_area() == "panel.alignment"
    assert b.open_area() == "alignment.apply"
    assert c.open_area() == "main_menu"
    assert d.open_area() == "tasks"

    document["chapters"][0]["steps"] = [{"title": "t", "text": "x", "interactive": "nowhere"}]
    with pytest.raises(ValueError, match="interactive"):
        parse_tour(document)
    document["chapters"][0]["steps"] = [{"title": "t", "text": "x", "on_leave": "magic"}]
    with pytest.raises(ValueError, match="action"):
        parse_tour(document)


def test_demo_path_relative_to_repository():
    tour = parse_tour({"version": 1, "chapters": [], "demo": {"default_path": "demo/x"}})
    from pathlib import Path

    assert Path(tour.demo_path).is_absolute()
    assert (Path(tour.demo_path).parent.parent / "src" / "catan").is_dir()


@pytest.fixture
def guide(tmp_path):
    """A GuideController on a bare window with a two-step test chapter."""
    import os
    from types import SimpleNamespace

    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    from PySide6.QtCore import QSettings
    from PySide6.QtWidgets import QApplication, QMainWindow, QWidget

    from catan.gui.guide.anchors import AnchorRegistry
    from catan.gui.guide.controller import GuideController

    app = QApplication.instance() or QApplication(["catan-test"])
    window = QMainWindow()
    window.setCentralWidget(QWidget())
    window.settings = QSettings(str(tmp_path / "guide.ini"), QSettings.Format.IniFormat)
    window.state = SimpleNamespace(guide_anchors=AnchorRegistry())
    window.data = SimpleNamespace(sessions=[])
    window.resize(800, 600)

    controller = GuideController(window)
    flag = {"done": False}
    controller.conditions["demo_available"] = lambda: flag["done"]
    controller.actions["choose_demo_folder"] = lambda: flag.update(done=True)
    controller.tour = parse_tour(
        {
            "version": 1,
            "chapters": [
                {
                    "id": "c",
                    "title": "C",
                    "steps": [
                        {"title": "a", "text": "x", "wait_for": "demo_available", "do_it": "choose_demo_folder"},
                        {"title": "b", "text": "y"},
                    ],
                }
            ],
        }
    )
    window.show()
    controller.start("c")
    yield SimpleNamespace(controller=controller, flag=flag, app=app)
    controller.stop()
    window.close()


def test_guide_never_moves_on_by_itself(guide):
    controller, bubble = guide.controller, guide.controller.bubble

    controller._tick()
    assert controller.index == 0 and not bubble.next_button.isEnabled()

    # "Do it for me" does it, but the user moves on.
    controller.do_it()
    for _ in range(3):
        controller._tick()
    assert controller.index == 0
    assert bubble.next_button.isEnabled() and "Next" in bubble.status.text()

    # Undoing it keeps the step done.
    guide.flag["done"] = False
    controller._tick()
    assert bubble.next_button.isEnabled()

    controller.next()
    assert controller.index == 1


def test_step_is_done_again_after_back(guide):
    controller = guide.controller
    guide.flag["done"] = True
    controller._tick()
    controller.next()
    guide.flag["done"] = False
    controller.back()
    controller._tick()

    assert controller.index == 0
    assert controller.bubble.next_button.isEnabled()


def test_advance_when_moves_on(guide):
    controller = guide.controller
    controller.tour.chapters[0].steps[0].advance_when = "demo_available"
    controller._tick()
    assert controller.index == 0

    guide.flag["done"] = True
    controller._tick()
    assert controller.index == 1


def test_quick_tour_and_panel_help_are_valid():
    tour = load_tour()
    quick = tour.quick

    assert quick is not None and quick.ready and not quick.requires
    # About five minutes, at most ten (~20 s per stop).
    assert 0 < quick.minutes <= 10 and len(quick.steps) * 20 <= 10 * 60
    linked = [s for s in quick.steps if s.more]
    assert len(linked) >= len(quick.steps) - 3
    for step in linked:
        assert step.more == "outline" or tour.section(step.more)[0] in tour.chapters
    # Every panel the quick tour shows has a ? section, and every view has one.
    assert {s.panel for s in quick.steps if s.panel} <= set(tour.help)
    assert set(tour.help) == set(PANEL_MODES)
    for spec in tour.help.values():
        chapter, first, last = tour.section(spec)
        assert 0 <= first <= last < len(chapter.steps)


@pytest.mark.parametrize(
    "extra, problem",
    [
        ({"help": {"footprints": "nowhere"}}, "unknown chapter"),
        ({"help": {"footprints": "c#nope"}}, "unknown step"),
        ({"help": {"nowhere": "c"}}, "unknown panels"),
        ({"quick": {"id": "q", "title": "Q", "steps": [{"title": "t", "text": "x", "more": "c#b:a"}]}}, "empty section"),
    ],
)
def test_sections_are_validated(extra, problem):
    document = {
        "version": 1,
        "chapters": [
            {
                "id": "c",
                "title": "C",
                "steps": [{"id": "a", "title": "a", "text": "x"}, {"id": "b", "title": "b", "text": "x"}],
            }
        ],
        **extra,
    }
    with pytest.raises(ValueError, match=problem):
        parse_tour(document)


SECTION_TOUR = {
    "version": 1,
    "help": {"footprints": "c#b2:b3"},
    "quick": {
        "id": "quick",
        "title": "Quick tour",
        "steps": [
            {"title": "q1", "text": "x", "more": "c#b2:b3"},
            {"title": "q2", "text": "x", "more": "outline"},
        ],
    },
    "chapters": [
        {
            "id": "c",
            "title": "C",
            "steps": [
                {"id": f"b{i}", "title": f"b{i}", "text": "x"} for i in range(1, 5)
            ],
        }
    ],
}


def test_learn_more_returns_to_the_quick_tour(guide):
    controller, bubble = guide.controller, guide.controller.bubble
    controller.tour = parse_tour(SECTION_TOUR)
    controller.start("quick")
    assert controller.chapter.id == "quick" and bubble.more_button.isVisibleTo(bubble)
    assert not bubble.skip_button.isVisibleTo(bubble)

    controller.learn_more()
    assert (controller.chapter.id, controller.index) == ("c", 1)
    assert bubble.skip_button.text() == "↩ Quick tour"
    controller.back()
    assert controller.index == 1  # not before the section
    controller.next()
    assert controller.index == 2
    assert bubble.next_button.text() == "↩ Quick tour"
    controller.next()
    assert (controller.chapter.id, controller.index) == ("quick", 0)

    # Leaving a section early also returns.
    controller.learn_more()
    controller.skip_chapter()
    assert (controller.chapter.id, controller.index) == ("quick", 0)

    controller.next()
    assert bubble.next_button.text() == "Done"
    controller.next()
    assert not controller.active and "quick" in controller.completed()


def test_panel_help_runs_its_section_and_closes(guide):
    controller, bubble = guide.controller, guide.controller.bubble
    controller.tour = parse_tour(SECTION_TOUR)
    assert controller.has_help("footprints") and not controller.has_help("statistics")

    controller.start_help("footprints")
    assert (controller.chapter.id, controller.index) == ("c", 1)
    assert bubble.counter.text().startswith("Guide: Chapter 1: C · step 1 / 2")
    assert not bubble.skip_button.isVisibleTo(bubble)
    controller.next()
    assert bubble.next_button.text() == "Done"
    controller.next()
    # Closed; a section does not mark the chapter completed.
    assert not controller.active and "c" not in controller.completed()


def test_outline_lists_the_quick_tour(guide):
    controller = guide.controller
    controller.tour = parse_tour(SECTION_TOUR)
    controller.start()
    outline = controller.outline
    assert outline.isVisible() and outline.quick_button.isVisibleTo(outline)
    assert "Quick tour" in outline.quick_label.text()
    outline.quick_button.click()
    assert controller.chapter.id == "quick"


def test_comparing_assignments_follows_curation():
    tour = load_tour()
    order = [c.id for c in tour.chapters]

    assert order.index("comparing") == order.index("curation") + 1
    comparing = tour.chapter("comparing")
    # The second set is loaded in the chapter itself ("Load it for me").
    first = comparing.steps[0]
    assert first.wait_for == "second_assignments_loaded"
    assert first.do_it == "load_alternative_assignments"
    assert "{demo_alternative}" in first.text


def test_curation_walks_through_three_presets():
    texts = _texts([load_tour().chapter("curation")])

    for words in ("and", "or", "by neuron", "count", "applied to", "percentile", "fitted model"):
        assert words in texts, words
    for preset in ("high-confidence", "merge neuron candidates", "split candidates"):
        assert preset in texts, preset
