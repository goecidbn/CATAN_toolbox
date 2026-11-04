import importlib
import threading

from PySide6.QtCore import QCoreApplication, QSettings, QThreadPool, QTimer
from PySide6.QtGui import QAction, QFont, Qt, QShortcut, QKeySequence
from PySide6.QtWidgets import (
    QMenu,
    QApplication,
    QAbstractSpinBox,
    QLineEdit,
    QTextEdit,
    QPlainTextEdit,
    QMessageBox,
    QHBoxLayout,
    QVBoxLayout,
    QMainWindow,
    QSplitter,
    QWidget,
    QSizePolicy,
)

from catan.gui.resources import (
    combine_stylesheets,
    load_stylesheet,
)

from catan.gui.GUI_elements import NeuronNavigationBar
from catan.gui.structures import AppState, Data
from catan.gui.interaction import click_events
from catan.gui.data.curation_filter_presets import CurationFilterPresetStore

from catan.tracking.structures import ReviewStatus

from .display_area import DisplayArea
from .main_menu import MainMenu
from .utils.menu_creation import fit_message_box_buttons
from catan.gui.data.assignment_colors import assignment_label, load_assignment_colors


# The main menu does not grow beyond this (px); the panels take the rest.
MAIN_MENU_MAX_WIDTH = 620

class MainWindow(QMainWindow):

    def __init__(self, *, settings=None):
        super().__init__()

        self.setWindowTitle("CATAN - Curation and Analysis of Tracked Across-sessions Neurons")

        QCoreApplication.setOrganizationName("WolfLabs")
        QCoreApplication.setApplicationName("CATAN")

        self.settings = (
            settings
            if settings is not None
            else QSettings(
                QSettings.Format.NativeFormat,
                QSettings.Scope.UserScope,
                "WolfLabs",
                "CATAN",
            )
        )

        self.state = AppState(settings=self.settings)
        load_assignment_colors(self.settings)

        self.data: Data = Data(self.state)
        self.state.tasks.start_queue_timer()

        # self.settings = QSettings()
        self._restore_settings()

        # --- UI setup ---
        self._init_ui()

        """
        connect interaction (make reloadable)
        """

        self._setup_navigation_shortcuts()
        self._setup_review_shortcuts()
        self._setup_curation_shortcuts()

        if CurationFilterPresetStore.can_save_builtin():
            reload_action = QAction("Reload plotting logic", self)
            reload_action.setShortcut("Ctrl+R")
            reload_action.triggered.connect(self.reload_logic)
            self.menuBar().addAction(reload_action)

        # Project: the assignments file with its sessions and model.
        # Owned by the window (kept alive with its actions and shortcuts).
        self.file_menu = file_menu = QMenu("File", self)
        self.menuBar().insertMenu(
            self.menuBar().actions()[0] if self.menuBar().actions() else None, file_menu
        )
        menu = lambda: self.gui_elements["main_menu"]  # created later
        for label, shortcut, slot in (
            ("Open project…", "Ctrl+O", lambda: menu().open_project()),
            ("Save project", "Ctrl+S", lambda: menu().save_project()),
            ("Save project as…", "Ctrl+Shift+S", lambda: menu().save_project(save_as=True)),
        ):
            action = file_menu.addAction(label)
            action.setShortcut(QKeySequence(shortcut))
            action.triggered.connect(lambda checked=False, slot=slot: slot())
            if label == "Open project…":
                self.recent_menu = file_menu.addMenu("Open recent")
                self.recent_menu.aboutToShow.connect(self._fill_recent_projects)

        # Analysis output: detected footprints only (splits / merges undone).
        file_menu.addSeparator()
        file_menu.addAction("Export for activity analysis…").triggered.connect(
            lambda checked=False: menu().export_analysis()
        )
        self.analysis_output_menu = file_menu.addMenu("Analysis file with project")
        self.analysis_output_menu.aboutToShow.connect(self._fill_analysis_output_menu)

        # Starter guide: Help menu, ? button (main menu), startup offer.
        from catan.gui.guide.anchors import install_main_window_anchors
        from catan.gui.guide.controller import GuideController

        install_main_window_anchors(self)
        self.guide = GuideController(self)
        help_menu = self.menuBar().addMenu("Help")
        guide_menu = help_menu.addMenu("Starter guide")
        guide_menu.aboutToShow.connect(lambda: self.guide.build_menu(guide_menu))
        self._add_title_corner()
        QTimer.singleShot(800, self.guide.offer_on_startup)

        print_debug = QAction("Print debug info", self)
        print_debug.setShortcut("Ctrl+D")
        print_debug.triggered.connect(self.print_debug_info)
        self.menuBar().addAction(print_debug)

        app = QApplication.instance()
        if app is None:
            return
        self.style_sheet = app.styleSheet()

        self.reset_stylesheet()
        # app.aboutToQuit.connect(self.debug_shutdown)

    def _restore_settings(self):

        # window geometry
        geom = self.settings.value("window/geometry")
        if geom is not None:
            self.restoreGeometry(geom)

    def _save_settings(self):

        self.settings.setValue("window/geometry", self.saveGeometry())
        self.settings.sync()

    def reload_logic(self):
        if not CurationFilterPresetStore.can_save_builtin():
            return

        tasks = self.state.tasks
        # print(self.data.assignments.union.footprints)

        # Avoid reloading module globals while workers are using them.
        if (
            tasks.processing_busy()
            or tasks.processing_requested
            or any(task is not None for task in tasks.current.values())
        ):
            QMessageBox.information(
                self,
                "Reload plotting logic",
                "Wait for running tasks to finish, or cancel them, "
                "then press Ctrl+R again.",
            )
            return

        # A standalone editor must be closed before its module is reloaded.
        if QApplication.activeModalWidget() is not None:
            return

        self._reload_model_logic()
        self._reload_statistics_logic()

        display_area = self.gui_elements["display_area"]

        # Preserve the current panel configuration and splitter positions.
        display_area._collect_live_leaf_configs(display_area.tree)
        display_area._collect_splitter_sizes(display_area.tree)

        importlib.reload(click_events)

        display_area.rebuild(
            reload_alignment=True,
        )
        self.gui_elements["main_menu"].rebuild()
        self.reset_stylesheet()

    def _reload_model_logic(self):
        module = importlib.import_module("catan.tracking.structures.model")
        existing_class = module.Model

        importlib.invalidate_caches()
        importlib.reload(module)
        updated_class = module.Model

        # Preserve class identity: existing instances and imported aliases
        # continue to refer to this class, now with updated methods.
        protected = {
            "__dict__",
            "__weakref__",
            "__module__",
            "__qualname__",
        }

        for name in tuple(vars(existing_class)):
            if name not in protected and name not in vars(updated_class):
                delattr(existing_class, name)

        for name, value in vars(updated_class).items():
            if name not in protected:
                setattr(existing_class, name, value)

        # Methods such as _from_dict() resolve Model through this module.
        module.Model = existing_class

        # print(f"Reloaded model logic from {module.__file__}")

    def _reload_statistics_logic(self):
        engine = self.data.statistic_engine

        factory_module_name = engine.registry_factory.__module__
        factory_name = engine.registry_factory.__name__

        module_names = (
            "catan.core.image_correlation",
            "catan.gui.data.statistics.footprint_pairs",
            "catan.gui.data.statistics.overlap_statistics",
            "catan.gui.data.statistics.calculations",
            "catan.gui.data.statistics.definitions",
            factory_module_name,
        )

        importlib.invalidate_caches()

        # Preserve order and avoid reloading the same module twice.
        for name in dict.fromkeys(module_names):
            module = importlib.import_module(name)
            importlib.reload(module)

        factory_module = importlib.import_module(factory_module_name)
        engine.registry_factory = getattr(factory_module, factory_name)

        # Keep the existing engine and its signal connections.
        engine.refresh_registry(emit=False)

        # Invalidate even when the definition's metadata stayed identical:
        # a helper function's implementation may have changed.
        engine._invalidate(tuple(engine.registry))

        print("[statistics reload] functions refreshed; caches invalidated", flush=True)

    def reset_stylesheet(self):

        app = QApplication.instance()
        if app is None:
            return

        style = combine_stylesheets(
            self.style_sheet,
            load_stylesheet("main.qss"),
        )

        app.setStyleSheet(style)

    REPOSITORY_URL = "https://github.com/goecidbn/CATAN_toolbox"

    def _add_title_corner(self):
        """Right end of the menu bar: CATAN version and a GitHub button
        (repository, issues for bugs and requests)."""
        from PySide6.QtCore import QUrl
        from PySide6.QtGui import QDesktopServices
        from PySide6.QtWidgets import QHBoxLayout, QLabel, QToolButton, QWidget
        import qtawesome as qta

        from catan.gui.utils.app_version import catan_version

        version = catan_version()
        corner = QWidget(self.menuBar())
        layout = QHBoxLayout(corner)
        layout.setContentsMargins(0, 0, 6, 0)
        layout.setSpacing(4)

        label = QLabel(f"CATAN {version.split('+')[0]}")
        label.setStyleSheet("color: #9aa3ad;")
        label.setToolTip(f"CATAN version {version}")
        layout.addWidget(label)

        button = QToolButton()
        button.setAutoRaise(True)
        button.setIcon(qta.icon("fa5b.github", color="#c8ccd2"))
        button.setToolTip(
            "CATAN on GitHub — report bugs, request features or suggest "
            "improvements as issues there"
        )
        button.clicked.connect(
            lambda: QDesktopServices.openUrl(QUrl(self.REPOSITORY_URL))
        )
        layout.addWidget(button)

        self.version_label, self.github_button = label, button
        self.menuBar().setCornerWidget(corner, Qt.Corner.TopRightCorner)

    def _fill_analysis_output_menu(self):
        """File ▸ Analysis file with project: ask / always / never (only
        when the assignments have splits, merges or replaced shapes)."""
        menu = self.analysis_output_menu
        menu.clear()
        main_menu = self.gui_elements["main_menu"]
        current = main_menu.analysis_output_mode()
        for mode, label in main_menu.ANALYSIS_OUTPUT_CHOICES:
            action = menu.addAction(label)
            action.setCheckable(True)
            action.setChecked(mode == current)
            action.triggered.connect(
                lambda checked=False, mode=mode: main_menu.set_analysis_output_mode(mode)
            )

    def _fill_recent_projects(self):
        """File ▸ Open recent: project name and folder, newest first."""
        from pathlib import Path

        menu = self.recent_menu
        menu.clear()
        main_menu = self.gui_elements["main_menu"]
        suffix = Data.PROJECT_SUFFIX
        paths = main_menu.data.recent_projects()
        if not paths:
            menu.addAction("No recent projects").setEnabled(False)
            return
        for path in paths:
            name = Path(path).name
            name = name[: -len(suffix)] if name.endswith(suffix) else name
            # Full folder (projects of other root folders), home as ~.
            folder = str(Path(path).parent)
            home = str(Path.home())
            if folder.startswith(home):
                folder = "~" + folder[len(home):]
            if len(folder) > 60:
                folder = folder[:20] + "…" + folder[-39:]
            action = menu.addAction(f"{name}  —  {folder}".replace("&", "&&"))
            action.setToolTip(path)
            action.triggered.connect(
                lambda checked=False, path=path: main_menu.open_recent_project(path)
            )
        menu.setToolTipsVisible(True)
        menu.addSeparator()
        menu.addAction("Clear list").triggered.connect(
            lambda: main_menu.data.forget_recent_project()
        )

    def print_debug_info(self):
        click_events.print_debug(self.state, self.data)

        from .utils.memory_report import memory_report

        print("\n".join(memory_report(self.data)))

    def _confirm_unsaved_assignments(self) -> bool:
        """Offer saving changed assignments and models before closing.

        Returns True to close now. Saving runs in the background; the
        window closes again once it has finished.
        """
        if getattr(self, "_closing_after_save", False):
            return True

        def close_after_save():
            self._closing_after_save = True
            # After the save's own bookkeeping.
            QTimer.singleShot(0, self.close)

        return self._offer_saving("closing", close_after_save) != "cancel"

    def _offer_saving(self, purpose, after_save):
        """Ask whether to save changed assignments / models before
        ``purpose`` (e.g. "closing"). Returns "clean" (nothing changed),
        "discard", "cancel" or "saving": the project save runs in the
        background and calls ``after_save`` once it has finished (a failed
        or cancelled save does not)."""
        unsaved = self.data.unsaved_assignments()
        models = self.data.unsaved_models()

        if not unsaved and not models:
            return "clean"

        external = set(self.data.changed_external_sets("assignments")) | set(
            self.data.changed_external_models()
        )

        dialog = QMessageBox(self)
        dialog.setIcon(QMessageBox.Icon.Warning)
        dialog.setWindowTitle("Unsaved changes")
        dialog.setText("Assignments or models have changes that were not saved.")
        parts = []
        if unsaved:
            parts.append(
                "Changed assignments: "
                + ", ".join(assignment_label(name) for name in unsaved)
            )
        if models:
            parts.append(
                "Changed models: " + ", ".join(assignment_label(name) for name in models)
            )
        if external:
            parts.append(
                "Loaded from a file (saved only where you choose): "
                + ", ".join(assignment_label(name) for name in sorted(external))
            )
        parts.append(f"Save the project before {purpose}?")
        dialog.setInformativeText("\n\n".join(parts))

        save_button = dialog.addButton("Save…", QMessageBox.ButtonRole.AcceptRole)
        dialog.addButton("Discard changes", QMessageBox.ButtonRole.DestructiveRole)
        cancel_button = dialog.addButton(QMessageBox.StandardButton.Cancel)
        dialog.setDefaultButton(save_button)
        fit_message_box_buttons(dialog)
        dialog.exec()

        clicked = dialog.clickedButton()

        if clicked is cancel_button:
            return "cancel"

        if clicked is not save_button:
            return "discard"

        # The project save writes sessions and models first, the
        # assignments last; ``finished`` runs after all of them.
        self.gui_elements["main_menu"].save_project(finished=after_save)
        return "saving"

    def clean_workspace(self):
        """Remove all sessions, models and assignment sets from this window
        (before loading another dataset), offering to save changes first
        as when closing."""
        tasks = self.state.tasks
        if (
            tasks.processing_busy()
            or tasks.processing_requested
            or tasks.has_deferred_processing()
        ):
            answer = QMessageBox.question(
                self,
                "Clean all",
                "Processing is still running. Cancel it and remove all data "
                "from this window?",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                QMessageBox.StandardButton.No,
            )
            if answer != QMessageBox.StandardButton.Yes:
                return

        outcome = self._offer_saving(
            "removing all data", lambda: QTimer.singleShot(0, self._clean_when_idle)
        )
        if outcome in ("cancel", "saving"):
            return

        if outcome == "clean":
            answer = QMessageBox.question(
                self,
                "Clean all",
                "Remove all sessions, models and assignment sets from this "
                "window?\n\nFiles on disk are not deleted.",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                QMessageBox.StandardButton.No,
            )
            if answer != QMessageBox.StandardButton.Yes:
                return

        self._clean_when_idle()

    def _clean_when_idle(self, attempts=50):
        """Cancel remaining tasks (also display work of the old data) and
        clear once they have ended; they may still finish in their threads."""
        tasks = self.state.tasks
        if tasks.tasks or tasks.processing_requested:
            if attempts == 50:
                tasks.cancel_all()
            if attempts > 0:
                QTimer.singleShot(100, lambda: self._clean_when_idle(attempts - 1))
                return

        if self.data.clear_workspace():
            self.gui_elements["main_menu"].workspace_cleared()

    def closeEvent(self, event):
        # print("MainWindow closeEvent")
        if not self._confirm_unsaved_assignments():
            event.ignore()
            return

        self._save_settings()
        for gui in self.gui_elements.values():
            if hasattr(gui, "_save_settings"):
                gui._save_settings()

        # Flush after every panel has written its settings.
        self.settings.sync()

        status = self.settings.status()
        if status != QSettings.Status.NoError:
            QMessageBox.warning(
                self,
                "Could not save settings",
                "CATAN could not save its preferences.\n\n"
                f"Location: {self.settings.fileName()}\n"
                f"Status: {status.name}",
            )

        super().closeEvent(event)
        # print("MainWindow closeEvent finished")

        # app = QApplication.instance()
        # print("quitOnLastWindowClosed:", app.quitOnLastWindowClosed())

        # print("top-level widgets:")
        # for widget in app.topLevelWidgets():
        #     print(
        #         " ",
        #         type(widget).__name__,
        #         repr(widget.objectName()),
        #         "visible=",
        #         widget.isVisible(),
        #         "window=",
        #         widget.isWindow(),
        #         "parent=",
        #         type(widget.parent()).__name__ if widget.parent() is not None else None,
        #     )

    ### ------------------------------------------###
    ###            UI initialization              ###
    ### ------------------------------------------###
    def _init_ui(self):
        """
        This should actually be its own class in a separate file.
        In here, it should merely specify, which widgets/handles
        should be available to the rest of the program.
        """

        splitter = QSplitter(Qt.Horizontal)
        splitter.setChildrenCollapsible(False)

        # =========== LEFT: SIDE MENU ===========
        main_menu = MainMenu(self)
        splitter.addWidget(main_menu)

        # =========== RIGHT: DISPLAY AREA ===========
        display = QWidget(self)
        display_layout = QVBoxLayout(display)

        display_area = DisplayArea(self)
        display_area.setSizePolicy(
            QSizePolicy.Policy.Expanding,
            QSizePolicy.Policy.Expanding,
        )
        display_layout.addWidget(display_area, 1)

        navigation_bar = NeuronNavigationBar.NeuronNavigationBar(self)
        navigation_bar.setSizePolicy(
            QSizePolicy.Policy.Expanding,
            QSizePolicy.Policy.Fixed,
        )
        display_layout.addWidget(navigation_bar, 0)

        splitter.addWidget(display)
        # Extra width goes to the panels; the menu stops at a maximum.
        main_menu.setMaximumWidth(MAIN_MENU_MAX_WIDTH)
        splitter.setStretchFactor(0, 0)
        splitter.setStretchFactor(1, 1)

        ## --- Build the UI ---
        main_app = QWidget(self)  # MainApp(self)
        layout = QHBoxLayout(main_app)
        layout.setContentsMargins(4, 4, 4, 4)

        layout.addWidget(splitter)

        self.setCentralWidget(main_app)

        # Qt recreates a shown top-level window when its first QOpenGLWidget
        # (vispy canvas) is added, which makes the window vanish briefly on the
        # first plot panel. A hidden one present before show() avoids that.
        try:
            from PySide6.QtOpenGLWidgets import QOpenGLWidget

            self._gl_surface_placeholder = QOpenGLWidget(main_app)
            self._gl_surface_placeholder.setFixedSize(0, 0)
            self._gl_surface_placeholder.hide()
        except ImportError:
            self._gl_surface_placeholder = None

        self.gui_elements = {
            "main_menu": main_menu,
            "display_area": display_area,
            "navigation_bar": navigation_bar,
        }

    ### ================================================ ###
    ### ============== SHORTKEY METHODS ================ ###
    ### ================================================ ###
    def _setup_navigation_shortcuts(self):
        self._navigation_shortcuts = []
        # Add a shortcut for navigating to the next neuron
        next_shortcut = QShortcut(QKeySequence("Right"), self)
        next_shortcut.setContext(Qt.ShortcutContext.ApplicationShortcut)
        next_shortcut.activated.connect(self.on_next_neuron_shortcut)
        self._navigation_shortcuts.append(next_shortcut)

        # Add a shortcut for navigating to the previous neuron
        prev_shortcut = QShortcut(QKeySequence("Left"), self)
        prev_shortcut.setContext(Qt.ShortcutContext.ApplicationShortcut)
        prev_shortcut.activated.connect(self.on_previous_neuron_shortcut)
        self._navigation_shortcuts.append(prev_shortcut)

    def on_previous_neuron_shortcut(self):
        if self._global_navigation_shortcut_blocked():
            return

        self.gui_elements["navigation_bar"].on_prev_footprint()

    def on_next_neuron_shortcut(self):
        if self._global_navigation_shortcut_blocked():
            return

        self.gui_elements["navigation_bar"].on_next_footprint()

    def _global_navigation_shortcut_blocked(self) -> bool:

        widget = QApplication.focusWidget()
        return isinstance(
            widget, (QLineEdit, QAbstractSpinBox, QTextEdit, QPlainTextEdit)
        )

    def _setup_review_shortcuts(self):

        self._review_shortcuts = []
        for status in ReviewStatus:

            # Single focused neuron
            shortcut = QShortcut(QKeySequence(status.shortcut), self)
            shortcut.setContext(Qt.ShortcutContext.ApplicationShortcut)
            shortcut.activated.connect(
                lambda status=status: self._set_review_status_from_shortcut(
                    status, batch=False
                )
            )
            self._review_shortcuts.append(shortcut)

            # Selected neurons in bulk
            batch_shortcut = QShortcut(QKeySequence(f"Shift+{status.shortcut}"), self)
            batch_shortcut.setContext(Qt.ShortcutContext.ApplicationShortcut)
            batch_shortcut.activated.connect(
                lambda status=status: self._set_review_status_from_shortcut(
                    status, batch=True
                )
            )
            self._review_shortcuts.append(batch_shortcut)

    def _setup_curation_shortcuts(self):
        self._curation_shortcuts = []

        bindings = (
            ("X", False, False, False),
            ("Del", True, False, False),
            ("Shift+X", False, True, False),
            ("Shift+Del", True, True, False),
            ("Ctrl+Shift+X", False, True, True),
            ("Ctrl+Shift+Del", True, True, True),
        )

        for sequence, remove, batch, whole_neurons in bindings:
            shortcut = QShortcut(QKeySequence(sequence), self)
            shortcut.setContext(Qt.ShortcutContext.WindowShortcut)
            shortcut.setAutoRepeat(False)
            shortcut.activated.connect(
                lambda remove=remove, batch=batch, whole_neurons=whole_neurons: self._curate_from_shortcut(
                    remove=remove,
                    batch=batch,
                    whole_neurons=whole_neurons,
                )
            )
            self._curation_shortcuts.append(shortcut)

    def _curate_from_shortcut(self, *, remove, batch, whole_neurons):
        if (
            self._review_shortcut_blocked()
            or QApplication.activeModalWidget() is not None
        ):
            return

        tasks = self.state.tasks
        if tasks.processing_busy() or tasks.processing_requested:
            QMessageBox.information(
                self,
                "Curation unavailable",
                "Wait for processing to finish before changing assignments.",
            )
            return

        if batch:
            components = list(self.state.selected_components or [])
            target_label = "selected"
        else:
            components = list(self.state.highlighted_components or [])
            target_label = "highlighted"

            if not components:
                focused = self.state.focused_component
                components = [] if focused is None else [focused]
                target_label = "focused"
                # Focused neuron without a session (e.g. browsing with the
                # navigation bar): the whole neuron.
                if focused is not None and focused.session_id is None:
                    whole_neurons = True

        if not components:
            return

        focused_neuron = target_label == "focused" and whole_neurons and not batch

        source = self.data.assignments
        version = self.state.data_version

        try:
            targets = self.data.curation_targets(
                components,
                whole_neurons=whole_neurons,
                remove=remove,
            )
        except ValueError as error:
            QMessageBox.warning(self, "Cannot curate selection", str(error))
            return

        if not targets:
            QMessageBox.information(
                self,
                "Nothing to change",
                "No eligible assigned components are selected.",
            )
            return

        if focused_neuron:
            # Exclude (X) is reversible: no question. Removal is not
            # recorded as a manipulation: confirm (Enter = remove).
            if remove:
                neuron_id = int(components[0].neuron_id)
                answer = QMessageBox.question(
                    self,
                    "Remove neuron",
                    f"Remove neuron {neuron_id} ({len(targets)} footprint"
                    f"{'s' if len(targets) != 1 else ''})?\n\n"
                    "Its assignment entries will be removed (cannot be undone; "
                    "X excludes it instead). Source files are unchanged.",
                    QMessageBox.StandardButton.Ok | QMessageBox.StandardButton.Cancel,
                    QMessageBox.StandardButton.Ok,
                )
                if answer != QMessageBox.StandardButton.Ok:
                    return

        elif batch or len(targets) > 1:
            action = "Remove" if remove else "Exclude"
            n_neurons = len({target[0] for target in targets})
            n_sessions = len({target[1] for target in targets})

            if whole_neurons:
                scope = (
                    f"{n_neurons} whole neurons represented by the selection"
                    f"\n({len(targets)} components across {n_sessions} sessions)"
                )
            else:
                scope = (
                    f"{len(targets)} {target_label} components "
                    f"across {n_sessions} sessions"
                )

            consequence = (
                "Their assignment entries will be removed."
                if remove
                else "Their included flags will be set to false."
            )

            answer = QMessageBox.question(
                self,
                f"{action} selection",
                f"{action} {scope}?\n\n" f"{consequence}\nSource files are unchanged.",
                QMessageBox.StandardButton.Ok | QMessageBox.StandardButton.Cancel,
                QMessageBox.StandardButton.Cancel,
            )

            if answer != QMessageBox.StandardButton.Ok:
                return

        # A confirmation dialog runs an event loop: recheck before applying.
        if (
            self.data.assignments is not source
            or self.state.data_version != version
            or tasks.processing_busy()
            or tasks.processing_requested
        ):
            QMessageBox.information(
                self,
                "Selection changed",
                "Data or processing state changed. Please retry the operation.",
            )
            return

        try:
            self.data.apply_curation_targets(
                targets,
                remove=remove,
                whole_neurons=whole_neurons,
            )
        except ValueError as error:
            QMessageBox.warning(self, "Curation failed", str(error))

    def _set_review_status_from_shortcut(self, status: ReviewStatus, *, batch: bool):

        if self._review_shortcut_blocked():
            return

        if batch:
            self._set_selected_review_status(status)
            return

        component = self.state.focused_component

        if component is None:
            return

        self.data.set_review_status([component.neuron_id], status)

    def _review_shortcut_blocked(self) -> bool:

        widget = QApplication.focusWidget()
        return isinstance(
            widget, (QLineEdit, QTextEdit, QPlainTextEdit, QAbstractSpinBox)
        )

    def _set_selected_review_status(self, status: ReviewStatus):

        components = self.state.selected_components or []
        neuron_ids = sorted({int(component.neuron_id) for component in components})

        if not neuron_ids:
            return

        n_neurons = len(neuron_ids)

        answer = QMessageBox.question(
            self,
            "Change review status",
            (
                f"Set {n_neurons} selected "
                f"neuron"
                f"{'' if n_neurons == 1 else 's'} "
                f"to '{status.label}'?"
            ),
            (QMessageBox.StandardButton.Ok | QMessageBox.StandardButton.Cancel),
            QMessageBox.StandardButton.Cancel,
        )

        if answer != QMessageBox.StandardButton.Ok:
            return

        self.data.set_review_status(neuron_ids, status)

    def debug_shutdown(self):
        print("\n=== SHUTDOWN DEBUG ===")

        pool = QThreadPool.globalInstance()

        print(
            "QThreadPool active:",
            pool.activeThreadCount(),
            "/ max:",
            pool.maxThreadCount(),
        )

        print("Python threads:")
        for thread in threading.enumerate():
            print(
                " ",
                repr(thread.name),
                "alive=",
                thread.is_alive(),
                "daemon=",
                thread.daemon,
                "ident=",
                thread.ident,
            )

        print("TaskManager:")
        for group in self.state.tasks.GROUPS:
            print(
                " ",
                group,
                self.state.tasks.group_summary(group),
            )

        print("======================\n")
