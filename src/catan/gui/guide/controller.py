"""Runs the starter guide on the main window.

One timer drives everything while the guide is shown: the cut-out follows
its target and the step's conditions are checked. The guide never moves on
by itself: a step's ``wait_for`` only enables Next once it held. While a
modal dialog (file chooser, review dialogs) is open, the bubble stays
beside it and says that the guide continues when it closes.

Besides the steps, the bubble shows two kinds of pages: "Before you
start" while a chapter's ``requires`` do not hold (its ``setup`` action
brings the data into that state, one stage per idle tick), and "Open …"
while no panel shows the view a step needs (``Step.panel``). The outline
(table of contents) opens when the tour starts from the beginning.

Two tracks: the quick tour (``Tour.quick``) and the detailed chapters. A
quick-tour stop's "Learn more" runs its detailed section and then returns
to the stop; a panel's ? button runs the panel's section (``Tour.help``)
and closes the guide at its end.
"""

import os
from pathlib import Path

from PySide6.QtCore import QObject, QRect, QTimer
from PySide6.QtWidgets import QApplication, QCheckBox, QFileDialog, QMenu, QMessageBox

from catan.gui.utils.session_labels import session_label

from .anchors import alignment_editor, panel_controller, panel_section
from .overlay import GuideBubble, GuideFrame, GuideOutline, GuideOverlay
from .tour import EXAMPLE_PRESET, MERGE_PRESET, SPLIT_PRESET, load_tour

POLL_MS = 150


class GuideController(QObject):
    def __init__(self, window):
        super().__init__(window)
        self.window = window
        self.settings = window.settings
        self.state = window.state
        self.data = window.data
        self.tour = load_tour()

        self.overlay = GuideOverlay(window)
        self.frame = GuideFrame(window)
        self.bubble = GuideBubble(window)
        self.bubble.backRequested.connect(self.back)
        self.bubble.nextRequested.connect(self.next)
        self.bubble.doItRequested.connect(self.do_it)
        self.bubble.skipChapterRequested.connect(self.skip_chapter)
        self.bubble.closeRequested.connect(self.stop)
        self.bubble.moreRequested.connect(self.learn_more)

        self.outline = GuideOutline(window)
        self.outline.chapterRequested.connect(self._start_from_outline)
        self.outline.closeRequested.connect(self.stop)

        self.chapter = None
        self.index = 0
        # Steps whose wait_for held once (chapter id, index): Next stays
        # enabled when the user undoes it or comes back later.
        self._done_steps = set()
        # Geometry of the open modal dialog the bubble waits beside.
        self._modal_rect = None
        # "Before you start" page of the current chapter is shown.
        self._prerequisites = False
        # The step's panel is not shown: the "Open …" page is shown.
        self._panel_prompt = False
        # Requirements the setup action is bringing about (None: idle).
        self._setup_goal = None
        self._setup_launched = set()
        # A section of the current chapter (first, last step) when started
        # from "Learn more" or a panel's ? button; None: the whole chapter
        # and on to the next one.
        self._segment = None
        # The quick-tour stop (chapter, index) "Learn more" returns to.
        self._return_to = None
        # Started from a panel's ? button: the guide closes at its end.
        self._help = False
        self._start_index = 0

        self._timer = QTimer(self)
        self._timer.setInterval(POLL_MS)
        self._timer.timeout.connect(self._tick)

        self.conditions = {
            "demo_available": lambda: self.demo_recipe() is not None,
            "root_is_demo": self._root_is_demo,
            "demo_sessions_registered": self._demo_sessions_registered,
            "loading_finished": self._loading_finished,
            "project_saved": self._project_saved,
            "sessions_loaded": self._sessions_loaded,
            "model_fitted": self._model_fitted,
            "neurons_registered": self._neurons_registered,
            "neuron_focused": lambda: self.state.focused_component is not None,
            "alignment_self_view": lambda: self._editor()._self_view is not None,
            "alignment_compare_on": lambda: (
                self._editor()._self_view is not None
                and self._editor().self_view_compare.isChecked()
            ),
            "alignment_settings_open": lambda: self._editor().options.toggle_button.isChecked(),
            "alignment_footprint_source": lambda: (
                self._editor().moving_source.currentData() == "footprints"
            ),
            "model_options_open": lambda: self._controller(
                "model_results"
            ).display_widget.options_button.isChecked(),
            "footprint_options_open": lambda: self._controller(
                "footprints"
            ).controls["parameter"].toggle_button.isChecked(),
            "selection_has_columns": lambda: bool(
                self._controller("selection_display").table.statistic_columns
            ),
            "preset_loaded": lambda: (
                self._controller("match_curator").filter_store.working_preset_key
                == EXAMPLE_PRESET
            ),
            "preset_evaluated": lambda: (
                self._controller("match_curator").current_result is not None
            ),
            "neuron_reviewed": self._neuron_reviewed,
            "later_session_shown": lambda: (
                not self._editor().draft.external and self._editor().draft.session_id > 0
            ),
            "model_evidence_available": lambda: (
                self._controller("model_results").display_widget._counts is not None
            ),
            "alignment_pair_view": lambda: self._editor()._self_view is None,
            "shift_inset_shown": lambda: (
                self._editor().inset_mode.currentData() == "shifts"
            ),
            "several_selected": lambda: len(self.state.selected_components or ()) >= 2,
            "matches_selected": self._matches_selected,
            "histogram_shown": lambda: self._plot_type() == "histogram",
            "scatter_shown": lambda: self._plot_type() == "scatter",
            "series_shown": lambda: self._plot_type() == "session_series",
            "footprint_highlighted": lambda: bool(self.state.highlighted_components),
            "merge_preset_loaded": lambda: self._preset_is(MERGE_PRESET),
            "split_preset_loaded": lambda: self._preset_is(SPLIT_PRESET),
            "second_assignments_loaded": lambda: len(self._assignment_sets()) >= 2,
            "footprints_other_set_shown": self._footprints_other_set_shown,
            "difference_histogram_shown": lambda: (
                self._plot_type() == "histogram"
                and self._x_statistic() == "assignment_difference"
            ),
            "sets_compared": lambda: len(self._compared_sets()) >= 2,
        }
        self.actions = {
            "choose_demo_folder": self.choose_demo_folder,
            "set_demo_root": self._set_demo_root,
            "load_demo_recipe": self._load_demo_recipe,
            "open_session_config": self._open_session_config,
            "save_project": lambda: self._main_menu().save_project(),
            "open_panel": self._open_step_panel,
            "setup_chapter": self._setup_chapter,
            "fit_model": lambda: self.data.queue_process_model(mode="pending"),
            "register_neurons": lambda: self.data.queue_process_assignments(mode="pending"),
            "focus_next_neuron": lambda: self.window.gui_elements[
                "navigation_bar"
            ].on_next_footprint(),
            "alignment_self_view": self._alignment_self_view,
            "alignment_compare": self._alignment_compare,
            "open_alignment_settings": lambda: self._open_toggle(
                self._editor().options.toggle_button
            ),
            "show_footprint_source": self._show_footprint_source,
            "open_model_options": lambda: self._open_toggle(
                self._controller("model_results").display_widget.options_button
            ),
            "open_footprint_options": lambda: self._open_toggle(
                self._controller("footprints").controls["parameter"].toggle_button
            ),
            "load_preset": self._load_example_preset,
            "evaluate_preset": lambda: self._controller(
                "match_curator"
            ).menu.evaluate_button.click(),
            "mark_focused_reviewed": self._mark_focused_reviewed,
            "select_later_session": self._select_later_session,
            "close_alignment_settings": lambda: self._close_toggle(
                self._editor().options.toggle_button
            ),
            "close_model_options": lambda: self._close_toggle(
                self._controller("model_results").display_widget.options_button
            ),
            "offer_demo_root_start": lambda: setattr(
                self._main_menu(), "root_dialog_start", self.demo_path() or None
            ),
            "clear_demo_root_start": lambda: setattr(
                self._main_menu(), "root_dialog_start", None
            ),
            "offer_demo_add_start": lambda: setattr(
                self._main_menu().session_list.load_row, "dialog_start", self.demo_path() or None
            ),
            "clear_demo_add_start": lambda: setattr(
                self._main_menu().session_list.load_row, "dialog_start", None
            ),
            "alignment_pair_view": self._alignment_pair_view,
            "show_shift_inset": lambda: self._set_inset("shifts"),
            "show_correlation_inset": lambda: self._set_inset("correlations"),
            "select_some_neurons": self._select_some_neurons,
            "select_matches": lambda: self._controller(
                "match_curator"
            ).menu.select_button.click(),
            "show_histogram": lambda: self._show_statistics("footprint_size", "none"),
            "show_scatter": lambda: self._show_statistics(
                "footprint_size", "footprint_eccentricity"
            ),
            "show_series": lambda: self._show_statistics("none", "footprint_size"),
            "highlight_footprint": self._highlight_footprint,
            "open_overview_options": lambda: self._open_toggle(
                self._options_toggle("tracked_overview")
            ),
            "close_overview_options": lambda: self._close_toggle(
                self._options_toggle("tracked_overview")
            ),
            "close_footprint_options": lambda: self._close_toggle(
                self._controller("footprints").controls["parameter"].toggle_button
            ),
            "open_statistics_options": lambda: self._open_toggle(
                self._options_toggle("statistics")
            ),
            "close_statistics_options": lambda: self._close_toggle(
                self._options_toggle("statistics")
            ),
            "quick_setup": self._quick_setup,
            "load_merge_preset": lambda: self._load_preset(MERGE_PRESET),
            "load_split_preset": lambda: self._load_preset(SPLIT_PRESET),
            "evaluate_and_select": self._evaluate_and_select,
            "open_condition": self._open_condition,
            "reveal_anchor": self._reveal_anchor,
            "load_alternative_assignments": self._load_alternative_assignments,
            "show_other_set": self._show_other_set,
            "show_assignment_differences": lambda: self._show_statistics(
                "assignment_difference", "none"
            ),
            "compare_occurrence": self._compare_occurrence,
        }

    # ------------------------------------------------------------------
    # Settings, entry points
    # ------------------------------------------------------------------

    @property
    def active(self):
        return self.chapter is not None

    def demo_path(self):
        path = self.settings.value("guide/demo_path", "") or self.tour.demo_path
        return str(path) if path else ""

    def demo_recipe(self):
        """The demo's loading recipe, or None if the folder is missing."""
        folder = self.demo_path()
        if not folder:
            return None
        recipe = Path(folder) / self.tour.demo_recipe
        return recipe if recipe.is_file() else None

    def startup_enabled(self):
        return not _as_bool(self.settings.value("guide/disabled", False))

    def set_startup_enabled(self, enabled):
        self.settings.setValue("guide/disabled", not enabled)

    def offer_on_startup(self):
        """Ask once per launch until the tour was started (or switched off)."""
        seen = int(self.settings.value("guide/seen_version", 0) or 0)

        if not self.startup_enabled() or seen >= self.tour.version or self.active:
            return

        box = QMessageBox(self.window)
        box.setWindowTitle("Welcome to CATAN")
        box.setIcon(QMessageBox.Icon.Question)
        quick = self.tour.quick
        box.setText(
            "Take a tour? The <b>quick tour</b> shows what is where and what each "
            "panel is for"
            + (f" (about {quick.minutes} min)" if quick is not None and quick.minutes else "")
            + ". The <b>detailed guide</b> walks you through loading, aligning, "
            "tracking and curating on a small demo dataset, chapter by chapter."
        )
        box.setInformativeText(
            "You can start either later with the ? button; each panel's ? "
            "opens its part of the guide."
        )
        start_quick = (
            box.addButton("Quick tour", QMessageBox.ButtonRole.AcceptRole)
            if quick is not None
            else None
        )
        start = box.addButton("Detailed guide", QMessageBox.ButtonRole.AcceptRole)
        box.addButton("Not now", QMessageBox.ButtonRole.RejectRole)
        never = QCheckBox("Don't show this again")
        box.setCheckBox(never)
        box.exec()

        if never.isChecked():
            self.set_startup_enabled(False)

        if box.clickedButton() is start:
            self.start()
        elif start_quick is not None and box.clickedButton() is start_quick:
            self.start(quick.id)

    def build_menu(self, menu: QMenu):
        """Chapter chooser and guide settings (the ? button, Help menu)."""
        menu.clear()
        completed = self.completed()
        quick = self.tour.quick
        if quick is not None:
            text = f"Quick tour (~{quick.minutes} min)" if quick.minutes else quick.title
            action = menu.addAction(text + ("  ✓" if quick.id in completed else ""))
            action.setToolTip(quick.description or quick.summary)
            action.triggered.connect(lambda _=False: self.start(quick.id))
        menu.addAction("Detailed guide: outline…", lambda: self.start())
        menu.addSeparator()

        for number, chapter in enumerate(self.tour.chapters, start=1):
            text = f"{number}. {chapter.title}"
            if not chapter.ready:
                text += " — coming soon"
            elif chapter.id in completed:
                text += "  ✓"
            action = menu.addAction(text)
            action.setToolTip(chapter.description or chapter.summary)
            action.setEnabled(chapter.ready)
            # The first chapter starts with the outline.
            action.triggered.connect(
                lambda _=False, c=chapter.id, first=number == 1: self.start(
                    None if first else c
                )
            )

        menu.addSeparator()
        startup = menu.addAction("Offer the tour at startup")
        startup.setCheckable(True)
        startup.setChecked(self.startup_enabled())
        startup.toggled.connect(self._set_startup_offer)
        menu.addAction("Demo data folder…", self.choose_demo_folder)
        menu.setToolTipsVisible(True)

    def _set_startup_offer(self, enabled):
        self.set_startup_enabled(enabled)
        if enabled:
            self.settings.setValue("guide/seen_version", 0)

    def choose_demo_folder(self):
        folder = QFileDialog.getExistingDirectory(
            self.window, "Select the CATAN demo data folder", self.demo_path()
        )
        if folder:
            self.settings.setValue("guide/demo_path", folder)
            self._render()

    # ------------------------------------------------------------------
    # Navigation
    # ------------------------------------------------------------------

    def completed(self):
        value = self.settings.value("guide/completed", []) or []
        if isinstance(value, str):
            value = [value]
        return set(value)

    def _mark_completed(self, chapter):
        completed = self.completed() | {chapter.id}
        order = [c.id for c in self._all_chapters()]
        self.settings.setValue(
            "guide/completed", [c for c in order if c in completed]
        )

    def _all_chapters(self):
        quick = [self.tour.quick] if self.tour.quick is not None else []
        return [*quick, *self.tour.chapters]

    def start(self, chapter_id=None):
        """A chapter (or the quick tour), or (None) the detailed guide's
        outline."""
        self.settings.setValue("guide/seen_version", self.tour.version)
        self._reset_sections()

        if chapter_id is None:
            self.show_outline()
            return

        chapter = self.tour.chapter(chapter_id)
        if not chapter.ready:
            return

        self.outline.hide()
        self._start_chapter(chapter)
        self._timer.start()

    def has_help(self, mode):
        return mode in self.tour.help

    def start_help(self, mode, section=None):
        """The guide's part about a panel (its ? button); closes at its end."""
        spec = self.tour.help.get(mode)
        if spec is None:
            return
        chapter, first, last = self.tour.section(spec)
        self.settings.setValue("guide/seen_version", self.tour.version)
        self._reset_sections()
        self._help = True
        self._segment = (first, last)
        self.outline.hide()
        self._start_chapter(chapter, first)
        self._timer.start()

    def learn_more(self):
        """A quick-tour stop's detailed section; returns to the stop."""
        step = self.step
        if step is None or step.more is None:
            return
        if step.more == "outline":
            self.stop()
            self.start()
            return
        chapter, first, last = self.tour.section(step.more)
        self._return_to = (self.chapter, self.index)
        self._segment = (first, last)
        self._start_chapter(chapter, first)

    def _reset_sections(self):
        self._segment = None
        self._return_to = None
        self._help = False

    def _resume(self):
        """Back to the quick-tour stop "Learn more" was clicked on."""
        chapter, index = self._return_to
        self._leave_step()
        self._segment = None
        self._return_to = None
        self.chapter = chapter
        self._prerequisites = False
        self._enter(index, forward=False)

    def _last_index(self):
        if self._segment is not None:
            return self._segment[1]
        return len(self.chapter.steps) - 1

    def _first_index(self):
        return self._segment[0] if self._segment is not None else 0

    def _required_panels(self, step):
        """The views a step needs: in a panel's help only its own view (it
        explains the panel in place, without changing the layout)."""
        if self._help:
            return (step.panel,) if step.panel else ()
        return step.required_panels(self._panel_context())

    def _panel_context(self):
        """The chapter whose ``panels`` a step uses: none in a section (its
        steps show only their own view), unless it spans the chapter."""
        if self._segment is None or self._segment == (0, len(self.chapter.steps) - 1):
            return self.chapter
        return None

    def show_outline(self):
        """Table of contents: chapters, steps, duration, what you learn."""
        self._leave_step()
        self.chapter = None
        self.frame.hide()
        self._timer.stop()
        self.bubble.hide()
        self.overlay.set_hole(None)
        self.overlay.show()
        self.overlay.raise_()
        self.outline.populate(self.tour.chapters, self.completed(), self.tour.quick)
        central = self.window.centralWidget()
        self.outline.place(central.geometry() if central is not None else self.window.rect())
        self.outline.show()
        self.outline.raise_()

    def _start_from_outline(self, chapter_id):
        self.start(chapter_id)

    def stop(self):
        self._leave_step()
        if (self.active or self.outline.isVisible()) and not self._help:
            self._remove_demo_saves()
        self.chapter = None
        self._reset_sections()
        self._setup_goal = None
        self._modal_rect = None
        self._timer.stop()
        self.overlay.hide()
        self.frame.hide()
        self.bubble.hide()
        self.outline.hide()

    def next(self):
        if not self.active:
            return
        if self._prerequisites or self._panel_prompt:
            return
        if self.index < self._last_index():
            self._enter(self.index + 1, forward=True)
            return
        if self.index + 1 == len(self.chapter.steps):
            self._mark_completed(self.chapter)
        self._end_chapter()

    def back(self):
        if self.active and self.index > self._first_index() and not self._prerequisites:
            self._enter(self.index - 1, forward=False)

    def skip_chapter(self):
        if self.active:
            self._end_chapter()

    def _end_chapter(self):
        """End of the chapter or section: back to the quick tour, close (a
        panel's help, the quick tour) or on to the next chapter."""
        if self._return_to is not None:
            self._resume()
        elif self._help or self.chapter not in self.tour.chapters:
            self.stop()
        else:
            self._next_chapter()

    def do_it(self):
        """The step's action; the user still moves on with Next."""
        if not self.active:
            return
        if self._prerequisites:
            if self.chapter.setup is not None:
                self.actions[self.chapter.setup]()
            return
        if self._panel_prompt:
            self._open_step_panel()
            return
        step = self.step
        if step is None or step.do_it is None:
            return
        self.actions[step.do_it]()

    @property
    def step(self):
        if not self.active:
            return None
        return self.chapter.steps[self.index]

    def _next_chapter(self):
        chapters = self.tour.chapters
        position = chapters.index(self.chapter)
        following = next((c for c in chapters[position + 1 :] if c.ready), None)
        if following is None:
            self.stop()
            return
        self._start_chapter(following)

    def _start_chapter(self, chapter, start=0):
        # A running setup (e.g. the quick tour's demo in the background)
        # goes on: it ends by itself once everything holds.
        self._leave_step()
        self.chapter = chapter
        self.index = start
        self._start_index = start
        # "Before you start" while its requirements do not hold.
        self._prerequisites = not self._holds_all(chapter.requires)
        if self._prerequisites:
            self._render()
        else:
            self._enter(start, forward=True)

    def _enter(self, index, *, forward):
        steps = self.chapter.steps

        # Forward: skip steps that are already done.
        while forward and index < self._last_index() and self._holds(steps[index].skip_if):
            index += 1

        self._leave_step()
        self.index = index
        step = steps[index]
        self.bubble.release()
        self._entered = (self.chapter.id, index)
        if step.on_enter is not None:
            self._run_action(step.on_enter)
        # After on_enter: it may have opened the step's panel.
        self._panel_prompt = self._needs_panel(step)
        self._render()

    def _leave_step(self):
        """Run the shown step's on_leave (once)."""
        entered = getattr(self, "_entered", None)
        self._entered = None
        if entered is None or self.chapter is None or entered[0] != self.chapter.id:
            return
        step = self.chapter.steps[entered[1]]
        if step.on_leave is not None:
            self._run_action(step.on_leave)

    def _run_action(self, name):
        try:
            self.actions[name]()
        except (AttributeError, KeyError, IndexError, RuntimeError, TypeError):
            pass  # e.g. the panel it refers to is gone

    def _step_done(self):
        step = self.step
        if step.wait_for is None:
            return True
        key = (self.chapter.id, self.index)
        if key not in self._done_steps and self._check(step.wait_for):
            self._done_steps.add(key)
        return key in self._done_steps

    # ------------------------------------------------------------------
    # Display
    # ------------------------------------------------------------------

    def _render(self):
        step = self.step
        if step is None:
            return

        chapters = self.tour.chapters
        bubble = self.bubble
        # (Disabled while a modal dialog was open.)
        bubble.close_button.setEnabled(True)
        bubble.skip_button.setEnabled(True)
        bubble.more_button.setEnabled(True)
        self._render_skip_button()
        bubble.more_button.hide()

        if self._prerequisites:
            self._render_prerequisites()
            return

        first, last = self._first_index(), self._last_index()
        unit = "stop" if self.chapter is self.tour.quick else "step"
        bubble.counter.setText(
            f"{self._chapter_label()} · {unit} {self.index - first + 1} / {last - first + 1}"
        )

        if self._panel_prompt:
            self._render_panel_prompt(step)
            return

        bubble.title.setText(step.title)
        bubble.text.setText(self._fill(step.text))

        bubble.do_it.setVisible(step.do_it is not None)
        bubble.do_it.setEnabled(True)
        bubble.do_it.setText(step.do_it_label)
        bubble.back_button.setEnabled(self.index > first)

        if step.more is not None:
            if step.more == "outline":
                bubble.more_button.setText("Open the detailed guide →")
            else:
                target = self.tour.section(step.more)[0]
                bubble.more_button.setText(f"Learn more: {target.title} →")
            bubble.more_button.show()

        last_step = self.index == last
        if not last_step:
            next_text = "Next"
        elif self._return_to is not None:
            next_text = "↩ Quick tour"
        elif self._help or self.chapter not in chapters:
            next_text = "Done"
        elif any(c.ready for c in chapters[chapters.index(self.chapter) + 1 :]):
            next_text = "Next chapter"
        else:
            next_text = "Finish"
        if last_step and self._return_to is None:
            bubble.skip_button.hide()
        bubble.next_button.setText(next_text)
        self._update_status()
        self._update_geometry(force=True)

    def _chapter_label(self):
        chapter = self.chapter
        if chapter is self.tour.quick:
            return chapter.title
        number = self.tour.chapters.index(chapter) + 1
        prefix = "Guide: " if self._help else ""
        return f"{prefix}Chapter {number}: {chapter.title}"

    def _render_skip_button(self):
        """"Back to the quick tour" in a Learn-more section; "Skip chapter"
        in the detailed guide; none in the quick tour and a panel's help
        (Close ends them)."""
        button = self.bubble.skip_button
        if self._return_to is not None:
            button.setText("↩ Quick tour")
            button.setToolTip("Back to the quick tour")
            button.show()
        elif self._help or self.chapter not in self.tour.chapters:
            button.hide()
        else:
            button.setText("Skip chapter")
            button.setToolTip("")
            button.show()

    REQUIREMENT_TEXT = {
        "sessions_loaded": "sessions with loaded spatial data (the demo data, or your own)",
        "model_fitted": "a fitted match model",
        "neurons_registered": "neurons registered across the sessions",
    }

    def _render_prerequisites(self):
        bubble = self.bubble
        chapter = self.chapter
        missing = [r for r in chapter.requires if not self._check(r)]
        bubble.counter.setText(f"{self._chapter_label()} · before you start")
        bubble.title.setText("Before you start")
        items = "".join(
            f"<li>{self.REQUIREMENT_TEXT.get(name, name)}</li>" for name in missing
        )
        working = (
            "<br><br><i>Setting it up… the chapter starts by itself.</i>"
            if self._setup_goal is not None
            else ""
        )
        bubble.text.setText(
            "This chapter needs:"
            f"<ul>{items}</ul>"
            "<b>Set it up for me</b> does what is missing: it loads the demo "
            "data if no sessions are loaded, fits the model and registers the "
            "neurons — or work through the previous chapters."
            + working
        )
        bubble.do_it.setVisible(chapter.setup is not None)
        bubble.do_it.setText("Set it up for me")
        bubble.do_it.setEnabled(self._setup_goal is None)
        bubble.back_button.setEnabled(False)
        bubble.next_button.setText("Next")
        bubble.next_button.setEnabled(False)
        bubble.status.setText("Waiting until this is done…")
        bubble.status.show()
        # Written directly: the next step's status must replace it.
        self._status_text = None
        self._last_anchor = None
        self._update_geometry(force=True)

    def _render_panel_prompt(self, step):
        bubble = self.bubble
        modes = self._required_panels(step)
        missing = self._missing_panels(step)
        self._prompt_missing = tuple(missing)
        if len(modes) == 1:
            path = _panel_menu_path(modes[0])
            bubble.title.setText(f"Open {path[-1]}")
            bubble.text.setText(
                f"This step uses the <b>{path[-1]}</b> view. Click the menu button "
                "at the top left of any panel and choose <b>"
                + " ▸ ".join(path)
                + "</b>."
            )
            bubble.do_it.setText("Open it for me")
        else:
            names = [_panel_menu_path(mode)[-1] for mode in modes]
            items = "".join(
                f"<li><b>{' ▸ '.join(_panel_menu_path(mode))}</b>"
                + (" (missing)" if mode in missing else " ✓")
                + "</li>"
                for mode in modes
            )
            bubble.title.setText("Open " + " and ".join([", ".join(names[:-1]), names[-1]]))
            bubble.text.setText(
                "This part uses several views side by side: what you select, "
                "focus or hover in one of them is shown in all others."
                f"<ul>{items}</ul>"
                "Split a panel with <b>+ →</b> or <b>+ ↓</b> in its header, then "
                "choose each view with the menu button at the panel's top left."
            )
            bubble.do_it.setText("Open them for me")
        bubble.do_it.setVisible(True)
        bubble.do_it.setEnabled(True)
        bubble.back_button.setEnabled(self.index > self._first_index())
        bubble.next_button.setText("Next")
        bubble.next_button.setEnabled(False)
        bubble.status.setText("Waiting for the panel…")
        bubble.status.show()
        # Written directly: the next step's status must replace it.
        self._status_text = None
        self._update_geometry(force=True)

    def _update_status(self):
        if self._prerequisites or self._panel_prompt:
            return
        step = self.step
        done = self._step_done()
        self.bubble.next_button.setEnabled(done)

        if step.wait_for is None:
            text = (
                "Loading the demo in the background… you can go on."
                if self._setup_goal is not None
                else None
            )
        elif done:
            text = "✓ Done — click Next when you are ready."
        else:
            text = "Waiting for you to do this…"
        if text != getattr(self, "_status_text", None) or (
            self.bubble.status.isVisible() != (text is not None)
        ):
            self._status_text = text
            self.bubble.status.setText(text or "")
            self.bubble.status.setVisible(text is not None)
            # The bubble's height follows the status line.
            self._update_geometry(force=True)

    def _current_anchor(self):
        if self._prerequisites:
            return None
        if self._panel_prompt:
            return "panel.mode_selector"
        return self.step.anchor

    def _current_open_area(self):
        """Anchor of the area left open for clicks (the step's panel for
        steps that explore it; Step.open_area). Asking for panels: the
        whole display area (split buttons, every panel's menu)."""
        if self._prerequisites:
            return self._current_anchor()
        if self._panel_prompt:
            return "display_area"
        return self.step.open_area(self._panel_context())

    def _also_open_rects(self):
        """Further open, framed areas of the step (Step.also_open)."""
        if self._prerequisites or self._panel_prompt or self.step is None:
            return []
        rects = []
        for name in self.step.also_open:
            rects += self.state.guide_anchors.rects(name, self.window)
        return rects

    def _panel_header_rects(self):
        """Headers of the shown panels (split buttons, view menus): the
        bubble keeps clear of them while it asks for panels."""
        from catan.gui.GUI_elements.display_section import DisplaySection

        from .anchors import _widget_rect

        rects = []
        for section in self._display_area().findChildren(DisplaySection):
            rect = _widget_rect(getattr(section, "header", None), self.window)
            if rect is not None:
                rects.append(rect)
        return rects

    def _anchor_rect(self, anchor):
        if anchor is None:
            return None
        return self.state.guide_anchors.rect(anchor, self.window)

    def _update_geometry(self, force=False):
        window = self.window
        anchor = self._current_anchor()
        open_anchor = self._current_open_area()
        rect = self._anchor_rect(anchor)
        open_rect = rect if open_anchor == anchor else self._anchor_rect(open_anchor)
        if open_rect is None:
            open_rect = rect
        extra = self._also_open_rects()
        framed = (
            self.state.guide_anchors.rects(anchor, window) if rect is not None else []
        ) + extra
        keep_clear = framed + (self._panel_header_rects() if self._panel_prompt else [])
        key = (rect, open_rect, tuple(framed), tuple(keep_clear), self._modal_rect)

        if force or key != getattr(self, "_last_rect", None) or (
            self.overlay.size() != window.size()
        ):
            self._last_rect = key
            self.overlay.set_hole(open_rect, extra)
            # The anchor's parts framed; the rest of a larger open area dimmed.
            self.frame.frame(open_rect, framed or None)
            central = window.centralWidget()
            bounds = central.geometry() if central is not None else window.rect()
            self.bubble.place(
                open_rect, bounds, avoid=self._modal_rect, keep_clear=keep_clear
            )

        # Independently: the outline leaves the overlay shown without bubble.
        if not self.overlay.isVisible():
            self.overlay.show()
        if not self.bubble.isVisible():
            self.bubble.show()
        self.overlay.raise_()
        if self.frame.isVisible():
            self.frame.raise_()
        self.bubble.raise_()

    def _tick(self):
        if not self.active:
            return

        # Modal dialogs (file choosers, reviews) take over: the bubble
        # waits beside them.
        modal = QApplication.activeModalWidget()
        if modal is not None and modal is not self.window:
            self._wait_for_modal(modal)
            return
        if self._modal_rect is not None:
            self._modal_rect = None
            self._status_text = None
            self._render()

        if self._setup_goal is not None:
            self._advance_setup()

        if self._prerequisites:
            if self._holds_all(self.chapter.requires):
                self._prerequisites = False
                self._enter(self._start_index, forward=True)
            else:
                self._update_geometry()
            return

        step = self.step
        needs_panel = self._needs_panel(step)
        if needs_panel != self._panel_prompt:
            self._panel_prompt = needs_panel
            self._render()
            return
        if self._panel_prompt:
            # The list shows which views are open already.
            if tuple(self._missing_panels(step)) != getattr(self, "_prompt_missing", None):
                self._render()
                return
            self._update_geometry()
            return

        if step.advance_when is not None and self._check(step.advance_when):
            # The one exception to "never moves on by itself" (e.g. loading
            # started: its progress is the next step).
            if self.index < self._last_index():
                self._enter(self.index + 1, forward=True)
                return

        self._update_status()
        self._update_geometry()
        self.bubble.settle()

    def _wait_for_modal(self, modal):
        """Keep the guide shown (it would look closed otherwise), beside
        the dialog, with a note; its buttons wait until it closes."""
        geometry = modal.frameGeometry()
        rect = QRect(self.window.mapFromGlobal(geometry.topLeft()), geometry.size())
        first = self._modal_rect is None
        if first or rect != self._modal_rect:
            self._modal_rect = rect
            if first:
                bubble = self.bubble
                for button in (
                    bubble.do_it,
                    bubble.more_button,
                    bubble.back_button,
                    bubble.next_button,
                    bubble.skip_button,
                    bubble.close_button,
                ):
                    button.setEnabled(False)
                self._status_text = None
                bubble.status.setText(
                    "Continue in the open window — the guide resumes when it closes."
                )
                bubble.status.show()
            self._update_geometry(force=True)

    def _fill(self, text):
        def project_folder():
            try:
                return str(self.data.project_folder())
            except ValueError:  # no root folder and no sessions yet
                return "the root folder's catan subfolder"

        values = {
            "demo_path": lambda: self.demo_path() or "(not set)",
            "demo_recipe": lambda: self.tour.demo_recipe,
            "demo_alternative": lambda: self.tour.demo_alternative_assignments,
            "first_session": lambda: (
                session_label(self.data, 0) if self.data.sessions else "S0"
            ),
            "project_folder": project_folder,
        }
        for key, value in values.items():
            if "{" + key + "}" in text:
                text = text.replace("{" + key + "}", value())
        return text

    # ------------------------------------------------------------------
    # Conditions and actions
    # ------------------------------------------------------------------

    def _check(self, name):
        try:
            return bool(self.conditions[name]())
        except (AttributeError, OSError, IndexError, KeyError, TypeError):
            return False

    def _holds(self, names):
        return any(self._check(name) for name in names)

    def _holds_all(self, names):
        return all(self._check(name) for name in names)

    def _display_area(self):
        return self.window.gui_elements["display_area"]

    def _controller(self, mode):
        return panel_controller(self._display_area(), mode)

    def _editor(self):
        return alignment_editor(self._display_area())

    def _missing_panels(self, step):
        modes = self._required_panels(step)
        if not modes:
            return []
        area = self._display_area()
        return [mode for mode in modes if panel_section(area, mode) is None]

    def _needs_panel(self, step):
        return bool(self._missing_panels(step))

    def _open_step_panel(self):
        """Show the step's view(s): one view in the largest panel; several
        side by side (the first left, the others stacked right)."""
        from catan.gui.GUI_elements.display_section import DisplaySection

        step = self.step
        if step is None:
            return
        modes = self._required_panels(step)
        if not modes or not self._missing_panels(step):
            return
        area = self._display_area()
        sections = [s for s in area.findChildren(DisplaySection) if s.isVisible()]
        if not sections:
            return
        if len(modes) == 1:
            section = max(sections, key=lambda s: s.width() * s.height())
            section.set_display_mode(modes[0])
            return
        arrange_panels(area, modes)

    @staticmethod
    def _open_toggle(button):
        if not button.isChecked():
            button.click()

    @staticmethod
    def _close_toggle(button):
        if button.isChecked():
            button.click()

    # Prerequisites: one stage per idle tick ------------------------------

    def _setup_chapter(self):
        if self.chapter is None:
            return
        self._setup_goal = tuple(self.chapter.requires)
        self._setup_launched = set()
        self._render()

    def _quick_setup(self):
        """Quick tour: the demo loaded, the model fitted and the neurons
        tracked in the background while the tour goes on — only into an
        empty workspace (loaded data of the user's is left as it is)."""
        if self.data.sessions or self._setup_goal is not None:
            return
        self._setup_goal = ("neurons_registered",)
        self._setup_launched = set()
        self._update_status()

    def _advance_setup(self):
        tasks = self.state.tasks
        busy = bool(tasks.tasks) or tasks.processing_busy() or tasks.processing_requested
        if busy or QApplication.activeModalWidget() is not None:
            return

        goal = self._setup_goal
        stages = (
            ("sessions_loaded", self._setup_sessions),
            ("model_fitted", self.actions["fit_model"]),
            ("neurons_registered", self.actions["register_neurons"]),
        )
        needed = {
            "sessions_loaded": ("sessions_loaded",),
            "model_fitted": ("sessions_loaded", "model_fitted"),
            "neurons_registered": ("sessions_loaded", "model_fitted", "neurons_registered"),
        }
        wanted = {stage for name in goal for stage in needed.get(name, (name,))}

        for name, launch in stages:
            if name not in wanted or self._check(name):
                continue
            if name in self._setup_launched:
                # Launched, finished, still missing: give up (shown as is).
                self._setup_goal = None
                self._render()
                return
            self._setup_launched.add(name)
            launch()
            return

        self._setup_goal = None

    def _setup_sessions(self):
        if not self._root_is_demo() and not self.data.sessions:
            self._set_demo_root()
        if not self._demo_sessions_registered() and not self.data.sessions:
            self._load_demo_recipe()

    def _main_menu(self):
        return self.window.gui_elements["main_menu"]

    DEMO_SAVE_SUFFIXES = (
        ".catan.hdf5",
        ".sessions.hdf5",
        ".model.hdf5",
        ".assignments.hdf5",
    )

    def _remove_demo_saves(self):
        """The guide's saves in the demo folder (its catan/ subfolder:
        project, sessions, model, assignments files) are removed when the
        guide closes, so the next run starts clean. Only files with
        CATAN's save suffixes, only in the demo folder."""
        folder = self.demo_path()
        if not folder or self.demo_recipe() is None:
            return
        saves = Path(folder) / "catan"
        if not saves.is_dir():
            return
        for path in saves.iterdir():
            if path.is_file() and path.name.endswith(self.DEMO_SAVE_SUFFIXES):
                try:
                    path.unlink()
                except OSError:
                    pass
        try:
            saves.rmdir()  # only if empty
        except OSError:
            pass

    def _root_is_demo(self):
        folder = self.demo_path()
        if not folder or not self.data.root:
            return False
        return os.path.realpath(self.data.root) == os.path.realpath(folder)

    def _demo_session_count(self):
        recipe = self.demo_recipe()
        if recipe is None:
            return None
        folder = os.path.realpath(recipe.parent)
        return sum(
            1
            for session in self.data.sessions
            if session.path is not None
            and os.path.realpath(str(session.path)).startswith(folder + os.sep)
        )

    def _demo_sessions_registered(self):
        count = self._demo_session_count()
        return bool(count) and count >= len(self._recipe_sessions())

    def _recipe_sessions(self):
        import json

        recipe = self.demo_recipe()
        return json.loads(recipe.read_text("utf-8"))["sessions"] if recipe else []

    def _loading_finished(self):
        """Spatial data of all active sessions loaded, no tasks left."""
        sessions = [s for s in self.data.sessions if getattr(s, "active", True)]
        tasks = self.state.tasks
        return (
            bool(sessions)
            and all(s.status["spatial_loaded"] for s in sessions)
            and not tasks.tasks
            and not tasks.processing_busy()
            and not tasks.processing_requested
        )

    def _project_saved(self):
        path = self.data.project_path()
        return path is not None and Path(path).is_file()

    def _set_demo_root(self):
        folder = self.demo_path()
        if not folder:
            self.choose_demo_folder()
            folder = self.demo_path()
        if not folder:
            return
        menu = self._main_menu()
        menu.edit_root_path.setText(folder)
        menu.edit_root_path.editingFinished.emit()

    def _load_demo_recipe(self):
        from catan.gui.background_tasks.TaskManager import TaskBatch

        recipe = self.demo_recipe()
        if recipe is None:
            self.choose_demo_folder()
            recipe = self.demo_recipe()
        if recipe is None:
            return
        load_row = self._main_menu().session_list.load_row
        load_row._queue_registration_path(
            recipe,
            batch=TaskBatch(),
            actions=set(load_row.registration_action_selector.actions),
            force_first_session=not self.data.sessions,
        )

    def _sessions_loaded(self):
        return bool(self.data.sessions) and self._loading_finished()

    def _model_fitted(self):
        model = self.data.model
        return model is not None and model.fitted and not self.data.model_work_pending()

    def _neurons_registered(self):
        assignments = self.data.assignments
        return (
            assignments is not None
            and assignments.ids.shape[0] > 0
            and not self.data.assignment_work_pending()
        )

    def _neuron_reviewed(self):
        status = getattr(self.data.assignments, "review_status", None)
        return status is not None and bool((status != 0).any())

    def _alignment_self_view(self):
        editor = self._editor()
        if editor is None:
            return
        session_id = editor.draft.session_id
        if session_id is not None:
            editor.correlation_inset.chosen.emit(session_id, session_id)

    def _alignment_compare(self):
        """Single-session view of the shown session, compared with its
        footprints (entering the single-session view first if needed)."""
        editor = self._editor()
        if editor is None:
            return
        if editor._self_view is None:
            self._alignment_self_view()
        QTimer.singleShot(0, self._check_compare)

    def _check_compare(self):
        editor = self._editor()
        if editor is not None and editor._self_view is not None:
            editor.self_view_compare.setChecked(True)

    def _alignment_pair_view(self):
        """Leave the single-session view: compare with the reference."""
        editor = self._editor()
        if editor is None or editor._self_view is None:
            return
        session_id = editor.draft.session_id
        index = editor.reference.currentIndex()
        references = editor._reference_session_ids
        reference = references[index] if 0 <= index < len(references) else None
        if reference is None or reference == session_id:
            reference = session_id - 1 if session_id else session_id + 1
        if 0 <= reference < len(self.data.sessions):
            editor.correlation_inset.chosen.emit(
                max(session_id, reference), min(session_id, reference)
            )

    def _set_inset(self, mode):
        editor = self._editor()
        if editor is not None:
            index = editor.inset_mode.findData(mode)
            if index >= 0:
                editor.inset_mode.setCurrentIndex(index)

    def _options_toggle(self, mode):
        return self._controller(mode).controls["panel"].toggle_button

    def _plot_type(self):
        return getattr(self._controller("statistics"), "plot_type", None)

    def _show_statistics(self, x_key, y_key):
        controls = self._controller("statistics").controls
        # y first: an x of "none" with a y gives the session series.
        for slot, key in (("y", y_key), ("x", x_key)):
            controls[f"{slot}_selector"]._set_statistic(key)

    def _well_tracked_neurons(self, count):
        import numpy as np

        assignments = self.data.assignments
        detections = (assignments.ids >= 0).sum(axis=1)
        included = getattr(assignments.union, "included", None)
        if included is not None and len(included) == len(detections):
            detections = np.where(included, detections, -1)
        order = np.argsort(-detections, kind="stable")[:count]
        return sorted(int(i) for i in order if detections[i] > 0)

    def _select_some_neurons(self):
        from catan.core.structures import NeuronComponent

        neurons = self._well_tracked_neurons(3)
        if neurons:
            self.state.update_selected_components(
                [NeuronComponent(neuron_id=i, session_id=None) for i in neurons]
            )

    def _highlight_footprint(self):
        from catan.core.structures import NeuronComponent

        focused = self.state.focused_component
        if focused is None:
            self.actions["focus_next_neuron"]()
            focused = self.state.focused_component
        if focused is None:
            return
        row = self.data.assignments.ids[focused.neuron_id]
        sessions = [s for s in range(len(row)) if row[s] >= 0]
        if sessions:
            current = self.state.current_session_id
            session = current if current in sessions else sessions[0]
            self.state.update_highlighted_components(
                NeuronComponent(neuron_id=focused.neuron_id, session_id=session)
            )

    def _matches_selected(self):
        result = self._controller("match_curator").current_result
        selected = {c.neuron_id for c in self.state.selected_components or ()}
        return bool(result.neurons) and selected == {int(i) for i in result.neurons}

    def _select_later_session(self):
        if len(self.data.sessions) > 1:
            self.state.current_session_id = 1

    def _show_footprint_source(self):
        editor = self._editor()
        if editor is not None:
            editor.moving_source.setCurrentIndex(
                editor.moving_source.findData("footprints")
            )

    def _load_example_preset(self):
        self._load_preset(EXAMPLE_PRESET)

    def _load_preset(self, key):
        controller = self._controller("match_curator")
        if controller is not None:
            controller.menu.preset_selected.emit(key)

    def _preset_is(self, key):
        return self._controller("match_curator").filter_store.working_preset_key == key

    def _evaluate_and_select(self):
        """Evaluate the curator's preset, then select its matches (once
        the evaluation finished; a few ticks)."""
        controller = self._controller("match_curator")
        if controller is None:
            return
        controller.menu.evaluate_button.click()

        def select(attempts=40):
            current = self._controller("match_curator")
            if current is None or attempts <= 0:
                return
            if current.current_result is None or self.state.tasks.tasks:
                QTimer.singleShot(250, self, lambda: select(attempts - 1))
                return
            current.menu.select_button.click()

        QTimer.singleShot(250, self, select)

    def _open_condition(self):
        """The editor of the condition the step points at (curation.condition)."""
        controller = self._controller("match_curator")
        chips = self.state.guide_anchors.widgets("curation.condition")
        if controller is not None and chips:
            controller._edit_condition_requested(chips[0].condition_id)

    def _reveal_anchor(self):
        """Scroll the step's target into view (e.g. inside the curator's
        filter, which scrolls)."""
        from PySide6.QtWidgets import QScrollArea

        step = self.step
        if step is None or step.anchor is None:
            return
        for widget in self.state.guide_anchors.widgets(step.anchor):
            parent = widget.parentWidget()
            while parent is not None and not isinstance(parent, QScrollArea):
                parent = parent.parentWidget()
            if parent is not None:
                parent.ensureWidgetVisible(widget, 20, 20)

    # Comparing assignment sets ------------------------------------------

    def _assignment_sets(self):
        """Loaded assignment sets with neurons."""
        return [
            name
            for name, item in self.data._assignments.items()
            if item is not None and getattr(item, "ids", None) is not None and item.ids.shape[0] > 0
        ]

    def _other_set(self):
        others = [n for n in self._assignment_sets() if n != self.data.current_assignments]
        return others[0] if others else None

    def demo_alternative_assignments(self):
        folder = self.demo_path()
        name = self.tour.demo_alternative_assignments
        if not folder or not name:
            return None
        path = Path(folder) / name
        return path if path.is_file() else None

    def _load_alternative_assignments(self):
        if len(self._assignment_sets()) >= 2:
            return
        path = self.demo_alternative_assignments()
        if path is None:
            return
        menu = self._main_menu()
        self.data.queue_import_source(
            "assignments",
            str(path),
            "Alternative",
            on_loaded=lambda: menu._on_source_registered("assignments"),
        )

    def _footprints_other_set_shown(self):
        current = self.data.current_assignments
        for controller in self._controllers("footprints"):
            name = controller.config.get("assignments_name")
            if name is not None and name != current:
                return True
        return False

    def _controllers(self, mode):
        from catan.gui.GUI_elements.display_section import DisplaySection

        sections = [
            section
            for section in self._display_area().findChildren(DisplaySection)
            if section.isVisible() and section.display_mode == mode
        ]
        sections.sort(key=lambda s: s.mapTo(self.window, s.rect().topLeft()).x())
        return [s.active_controller for s in sections if s.active_controller is not None]

    def _show_other_set(self, attempts=5):
        """A second footprints panel (right of the first) showing the other
        assignment set."""
        other = self._other_set()
        if other is None or attempts <= 0:
            return
        from catan.gui.GUI_elements.display_section import DisplaySection

        controllers = self._controllers("footprints")
        if len(controllers) < 2:
            empty = [
                s
                for s in self._display_area().findChildren(DisplaySection)
                if s.isVisible() and s.display_mode == "empty"
            ]
            if empty:
                empty[0].set_display_mode("footprints")
            else:
                section = panel_section(self._display_area(), "footprints")
                if section is None:
                    return
                section.btn_split_h.click()
            QTimer.singleShot(300, self, lambda: self._show_other_set(attempts - 1))
            return
        target = controllers[-1]
        selector = target.assignments_selector
        index = selector.findData(other)
        if index >= 0:
            selector.setCurrentIndex(index)

    def _x_statistic(self):
        selector = self._controller("statistics").controls["x_selector"]
        query = selector.current_query()
        return None if query is None else query.statistic_key

    def _compared_sets(self):
        controller = self._controller("statistics")
        if controller is None:
            return ()
        return controller._histogram_assignment_names()

    def _compare_occurrence(self):
        controller = self._controller("statistics")
        if controller is None:
            return
        self._show_statistics("occurence", "none")
        controller._set_histogram_assignments(tuple(self._assignment_sets()))

    def _mark_focused_reviewed(self):
        from catan.gui.data.curation_actions import ReviewStatus

        component = self.state.focused_component
        if component is None:
            self.actions["focus_next_neuron"]()
            component = self.state.focused_component
        if component is not None:
            self.data.set_review_status([component.neuron_id], ReviewStatus.REVIEWED)

    def _open_session_config(self):
        rows = self._main_menu().session_list._row_widgets
        if rows:
            toggle = rows[min(rows)].config_constructor.toggle_config_options
            if not toggle.isChecked():
                toggle.click()


def _panel_menu_path(mode):
    """("Curation", "Session alignment") for a panel mode."""
    from catan.gui.GUI_elements.display_section import display_modes

    for opts in display_modes.values():
        if mode in opts["options"]:
            label = opts["options"][mode]
            parents = [opts["menu"]] if opts.get("menu") else []
            # Single options are actions of the parent menu (see
            # display_section): "Plots ▸ Overview", "Selection table".
            if len(opts["options"]) > 1:
                parents.append(opts["title"])
            return (*parents, label)
    return (mode,)


def arrange_panels(display_area, modes):
    """Replace the panel layout by ``modes`` side by side (the first left,
    the others stacked on the right), keeping the settings of panels that
    already show one of them."""
    from catan.gui.GUI_elements.display_area import leaf_node, split_node

    display_area._collect_live_leaf_configs(display_area.tree)
    configs = {}

    def collect(node):
        if node["type"] == "leaf":
            config = node.get("config") or {}
            configs.setdefault(config.get("display_mode"), config)
        else:
            for child in node["children"]:
                collect(child)

    collect(display_area.tree)
    leaves = [
        leaf_node(config=dict(configs.get(mode) or {"display_mode": mode}))
        for mode in modes
    ]
    right = leaves[-1]
    for leaf in reversed(leaves[1:-1]):
        right = split_node("vertical", leaf, right)
    tree = split_node("horizontal", leaves[0], right) if len(leaves) > 1 else leaves[0]
    display_area.tree = tree
    display_area.rebuild()
    display_area._save_settings()


def _as_bool(value):
    if isinstance(value, str):
        return value.lower() in ("1", "true", "yes")
    return bool(value)
