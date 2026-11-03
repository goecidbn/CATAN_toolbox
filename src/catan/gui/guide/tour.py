"""The starter guide's chapters and steps (resources/guide/tour.json).

A chapter: ``id``, ``title``, ``summary``, ``description`` (one line for
the outline: what you will learn), ``minutes`` (estimate), ``requires``
(conditions that must hold before its first step; otherwise the guide
shows what is missing and offers ``setup``, an action that brings the
data into that state) and its steps.

A step: ``title``, ``text`` (rich text; placeholders {demo_path},
{demo_recipe}, {demo_alternative}, {first_session},
{project_folder}), optional ``anchor``
(anchors.KNOWN_ANCHORS), ``panel`` (a panel mode the step needs: while no
panel shows it, the guide first asks to open it), ``panels`` (several
views side by side, e.g. to show linked views; default: the chapter's
``panels`` if they include ``panel``), ``interactive`` (the anchor of the
area left open for clicks; default: the display area for several panels,
the step's panel, else the anchor itself; "none": only the anchor),
``also_open`` (further anchors left open and framed, e.g. the navigation
bar a panel step refers to), ``bright`` (anchors left open and undimmed
without a frame, e.g. the whole curator while its parts are framed; also
per chapter),
``wait_for`` (a condition: Next stays disabled until it held once during
the step; the guide never moves on by itself, except ``advance_when``: a
condition that moves on as soon as it holds, e.g. once loading started),
``skip_if`` (conditions; the step is skipped when one holds on the way
forward), ``do_it`` / ``do_it_label`` (an action the guide can perform
instead) and ``on_enter`` / ``on_leave`` (actions run when the step is
shown / left, e.g. closing an options popup), ``id`` (to start a section
there) and ``more`` (quick tour: the detailed part behind "Learn more").

Sections: ``"<chapter>"`` (the whole chapter) or ``"<chapter>#<from>"`` /
``"<chapter>#<from>:<to>"`` (step ids, inclusive); "outline" opens the
detailed guide's table of contents.

Two tracks: ``quick`` (a chapter of its own: what is where, about five
minutes; each stop links to its detailed section) and ``chapters`` (the
detailed guide). ``help`` maps panel modes to sections (the panels' ?
buttons).

A chapter's ``panels``: the views its steps use together (arranged side
by side by "Open them for me").

``demo.default_path``: absolute, ``~``-relative, or relative to the
CATAN repository (the folder holding ``src/catan``).
"""

import json
from dataclasses import dataclass, field
from importlib.resources import files
from pathlib import Path

from .anchors import KNOWN_ANCHORS

# Conditions and actions the controller implements (validated by tests).
CONDITIONS = (
    "demo_available",
    "root_is_demo",
    "demo_sessions_registered",
    "loading_finished",
    "project_saved",
    "sessions_loaded",
    "model_fitted",
    "neurons_registered",
    "neuron_focused",
    "alignment_self_view",
    "alignment_compare_on",
    "alignment_settings_open",
    "alignment_footprint_source",
    "model_options_open",
    "footprint_options_open",
    "selection_has_columns",
    "preset_loaded",
    "preset_evaluated",
    "neuron_reviewed",
    "later_session_shown",
    "model_evidence_available",
    "alignment_pair_view",
    "shift_inset_shown",
    "several_selected",
    "matches_selected",
    "histogram_shown",
    "scatter_shown",
    "series_shown",
    "footprint_highlighted",
    "merge_preset_loaded",
    "split_preset_loaded",
    "second_assignments_loaded",
    "footprints_other_set_shown",
    "difference_histogram_shown",
    "sets_compared",
)
ACTIONS = (
    "choose_demo_folder",
    "set_demo_root",
    "load_demo_recipe",
    "open_session_config",
    "save_project",
    "open_panel",
    "setup_chapter",
    "fit_model",
    "register_neurons",
    "focus_next_neuron",
    "alignment_self_view",
    "alignment_compare",
    "open_alignment_settings",
    "show_footprint_source",
    "open_model_options",
    "open_footprint_options",
    "load_preset",
    "evaluate_preset",
    "mark_focused_reviewed",
    "select_later_session",
    "close_alignment_settings",
    "close_model_options",
    "offer_demo_root_start",
    "clear_demo_root_start",
    "offer_demo_add_start",
    "clear_demo_add_start",
    "alignment_pair_view",
    "show_shift_inset",
    "show_correlation_inset",
    "select_some_neurons",
    "select_matches",
    "show_histogram",
    "show_scatter",
    "show_series",
    "highlight_footprint",
    "open_overview_options",
    "close_overview_options",
    "close_footprint_options",
    "open_statistics_options",
    "close_statistics_options",
    "quick_setup",
    "load_merge_preset",
    "load_split_preset",
    "evaluate_and_select",
    "evaluate_select_both",
    "clean_workspace",
    "open_condition",
    "reveal_anchor",
    "load_alternative_assignments",
    "show_other_set",
    "show_assignment_differences",
    "compare_occurrence",
)
# Panel modes a step may require (display_section.DISPLAY_CONTROLLERS).
PANEL_MODES = (
    "tracked_overview",
    "footprints",
    "traces",
    "statistics",
    "selection_display",
    "session_alignment",
    "model_results",
    "match_curator",
)
# The area left open on steps of a panel (``Step.interactive`` default).
PANEL_ANCHORS = {
    "tracked_overview": "panel.overview",
    "footprints": "panel.footprints",
    "statistics": "panel.statistics",
    "selection_display": "panel.selection_table",
    "session_alignment": "panel.alignment",
    "model_results": "panel.model",
    "match_curator": "panel.curation",
}
# The curation chapter's example preset.
EXAMPLE_PRESET = "builtin:high_confidence_tracked_neurons"
# Further presets the curation chapter walks through.
MERGE_PRESET = "builtin:merge_neuron_candidates"
SPLIT_PRESET = "builtin:footprint_manipulation_split_candidates"


@dataclass
class Step:
    title: str
    text: str
    anchor: str | None = None
    wait_for: str | None = None
    skip_if: tuple[str, ...] = ()
    do_it: str | None = None
    do_it_label: str = "Do it for me"
    panel: str | None = None
    panels: tuple[str, ...] = ()
    interactive: str | None = None
    also_open: tuple[str, ...] = ()
    bright: tuple[str, ...] = ()
    on_enter: str | None = None
    on_leave: str | None = None
    advance_when: str | None = None
    id: str | None = None
    more: str | None = None

    def required_panels(self, chapter=None):
        """The views shown while this step runs (in layout order)."""
        if self.panels:
            return self.panels
        if self.panel is None:
            return ()
        if chapter is not None and self.panel in chapter.panels:
            return chapter.panels
        return (self.panel,)

    def open_area(self, chapter=None):
        """Anchor of the area that takes clicks (None: no cut-out)."""
        if self.interactive == "none":
            return self.anchor
        if self.interactive is not None:
            return self.interactive
        if self.anchor is None:
            return None
        if len(self.required_panels(chapter)) > 1:
            return "display_area"
        if self.panel is not None:
            return PANEL_ANCHORS.get(self.panel, self.anchor)
        return self.anchor


@dataclass
class Chapter:
    id: str
    title: str
    summary: str
    steps: list[Step] = field(default_factory=list)
    # Planned chapters list their content in ``plan`` (no steps yet).
    plan: tuple[str, ...] = ()
    description: str = ""
    minutes: int = 0
    requires: tuple[str, ...] = ()
    setup: str | None = None
    panels: tuple[str, ...] = ()
    bright: tuple[str, ...] = ()

    @property
    def ready(self):
        return bool(self.steps)


@dataclass
class Tour:
    version: int
    demo_path: str
    demo_recipe: str
    chapters: list[Chapter]
    # A second assignment set of the demo (demo_assignments.py).
    demo_alternative_assignments: str = ""
    quick: Chapter | None = None
    help: dict = field(default_factory=dict)

    def chapter(self, chapter_id):
        if self.quick is not None and chapter_id == self.quick.id:
            return self.quick
        return next(c for c in self.chapters if c.id == chapter_id)

    def section(self, spec):
        """``"<chapter>[#<from>[:<to>]]"`` -> (chapter, first, last) step
        indices; raises ValueError for unknown chapters or step ids."""
        name, _, steps = spec.partition("#")
        try:
            chapter = self.chapter(name)
        except StopIteration:
            raise ValueError(f"unknown chapter {name!r}") from None
        first, last = 0, len(chapter.steps) - 1
        if steps:
            start, _, end = steps.partition(":")
            ids = [step.id for step in chapter.steps]
            for value in (start, end):
                if value and value not in ids:
                    raise ValueError(f"unknown step {value!r} in {name!r}")
            first = ids.index(start)
            if end:
                last = ids.index(end)
            if last < first:
                raise ValueError(f"empty section {spec!r}")
        return chapter, first, last


def parse_tour(document):
    """Tour document -> Tour; raises ValueError on unknown names."""
    try:
        chapters = [_parse_chapter(entry) for entry in document["chapters"]]
        quick = _parse_chapter(document["quick"]) if "quick" in document else None
        demo = document.get("demo", {})
        tour = Tour(
            version=int(document["version"]),
            demo_path=_demo_path(demo.get("default_path", "")),
            demo_recipe=demo.get("recipe", ""),
            chapters=chapters,
            demo_alternative_assignments=demo.get("alternative_assignments", ""),
            quick=quick,
            help=dict(document.get("help", {})),
        )
    except (KeyError, TypeError) as exc:
        raise ValueError(f"Invalid starter guide file: {exc}") from exc

    # Sections referenced by the quick tour and the panels' ? buttons.
    references = [(f"help {mode!r}", spec) for mode, spec in tour.help.items()]
    for chapter in (quick, *chapters):
        for step in chapter.steps if chapter is not None else ():
            if step.more is not None and step.more != "outline":
                references.append((f"step {step.title!r}", step.more))
    for where, spec in references:
        try:
            tour.section(spec)
        except ValueError as exc:
            raise ValueError(f"{where}: {exc}") from None
    unknown = [mode for mode in tour.help if mode not in PANEL_MODES]
    if unknown:
        raise ValueError(f"help for unknown panels {unknown}")
    return tour


def _parse_chapter(entry):
    try:
        steps = []

        for raw in entry.get("steps", ()):
            step = Step(
                title=raw["title"],
                text=raw["text"],
                anchor=raw.get("anchor"),
                wait_for=raw.get("wait_for"),
                skip_if=tuple(raw.get("skip_if", ())),
                do_it=raw.get("do_it"),
                do_it_label=raw.get("do_it_label", "Do it for me"),
                panel=raw.get("panel"),
                panels=tuple(raw.get("panels", ())),
                interactive=raw.get("interactive"),
                also_open=tuple(raw.get("also_open", ())),
                bright=tuple(raw.get("bright", ())),
                on_enter=raw.get("on_enter"),
                on_leave=raw.get("on_leave"),
                advance_when=raw.get("advance_when"),
                id=raw.get("id"),
                more=raw.get("more"),
            )
            problems = []
            if step.anchor is not None and step.anchor not in KNOWN_ANCHORS:
                problems.append(f"anchor {step.anchor!r}")
            problems += [
                f"condition {name!r}"
                for name in (step.wait_for, step.advance_when, *step.skip_if)
                if name is not None and name not in CONDITIONS
            ]
            problems += [
                f"action {name!r}"
                for name in (step.do_it, step.on_enter, step.on_leave)
                if name is not None and name not in ACTIONS
            ]
            if step.interactive not in (None, "none", *KNOWN_ANCHORS):
                problems.append(f"interactive anchor {step.interactive!r}")
            problems += [
                f"{kind} anchor {name!r}"
                for kind, names in (("also_open", step.also_open), ("bright", step.bright))
                for name in names
                if name not in KNOWN_ANCHORS
            ]
            problems += [
                f"panel {mode!r}"
                for mode in (step.panel, *step.panels)
                if mode is not None and mode not in PANEL_MODES
            ]
            if problems:
                raise ValueError(
                    f"Step {step.title!r} of {entry['id']!r}: unknown "
                    + ", ".join(problems)
                )
            steps.append(step)

        chapter = Chapter(
            id=entry["id"],
            title=entry["title"],
            summary=entry.get("summary", ""),
            steps=steps,
            plan=tuple(entry.get("plan", ())),
            description=entry.get("description", ""),
            minutes=int(entry.get("minutes", 0)),
            requires=tuple(entry.get("requires", ())),
            setup=entry.get("setup"),
            panels=tuple(entry.get("panels", ())),
            bright=tuple(entry.get("bright", ())),
        )
        problems = [
            f"bright anchor {name!r}" for name in chapter.bright if name not in KNOWN_ANCHORS
        ]
        problems += [
            f"condition {name!r}"
            for name in chapter.requires
            if name not in CONDITIONS
        ]
        problems += [
            f"panel {mode!r}" for mode in chapter.panels if mode not in PANEL_MODES
        ]
        if chapter.setup is not None and chapter.setup not in ACTIONS:
            problems.append(f"action {chapter.setup!r}")
        if problems:
            raise ValueError(
                f"Chapter {chapter.id!r}: unknown " + ", ".join(problems)
            )
        ids = [step.id for step in steps if step.id is not None]
        if len(ids) != len(set(ids)):
            raise ValueError(f"Chapter {chapter.id!r}: duplicate step ids")
        return chapter
    except (KeyError, TypeError) as exc:
        raise ValueError(f"Invalid starter guide file: {exc}") from exc


def _demo_path(value):
    if not value:
        return ""
    path = Path(value).expanduser()
    if not path.is_absolute():
        # src/catan/gui/guide/tour.py -> the repository
        path = Path(__file__).resolve().parents[4] / path
    return str(path)


def load_tour():
    text = files("catan.resources.guide").joinpath("tour.json").read_text("utf-8")
    return parse_tour(json.loads(text))


_HELP_MODES = None


def help_modes():
    """Panel modes with a guide section (their ? button is shown)."""
    global _HELP_MODES
    if _HELP_MODES is None:
        try:
            _HELP_MODES = frozenset(load_tour().help)
        except (OSError, ValueError):
            _HELP_MODES = frozenset()
    return _HELP_MODES
