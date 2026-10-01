import importlib
import threading

from PySide6.QtCore import QCoreApplication, QSettings, QThreadPool
from PySide6.QtGui import QAction, QFont, Qt, QShortcut, QKeySequence
from PySide6.QtWidgets import (
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

from catan.tracking.structures import ReviewStatus

from .display_area import DisplayArea
from .main_menu import MainMenu


class MainWindow(QMainWindow):

    def __init__(self, *, settings=None):
        super().__init__()

        self.setWindowTitle("CATAN - Curating and Tracking Neurons")

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

        # --- setting up reload logic ---
        reload_action = QAction("Reload plotting logic", self)
        reload_action.setShortcut("Ctrl+R")
        reload_action.triggered.connect(self.reload_logic)
        self.menuBar().addAction(reload_action)

        # print_debug = QAction("Print debug info", self)
        # print_debug.setShortcut("Ctrl+D")
        # print_debug.triggered.connect(self.print_debug_info)
        # self.menuBar().addAction(print_debug)

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

        display_area = self.gui_elements["display_area"]

        # Preserve the current panel configuration and splitter positions.
        display_area._collect_live_leaf_configs(display_area.tree)
        display_area._collect_splitter_sizes(display_area.tree)

        importlib.reload(click_events)

        display_area.rebuild(
            reload_alignment=True,
            reload_statistics=True,
        )
        self.gui_elements["main_menu"].rebuild()
        self.reset_stylesheet()

    def _reload_model_logic(self):
        module = importlib.import_module(
            "catan.tracking.structures.model"
        )
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

    def reset_stylesheet(self):

        app = QApplication.instance()
        if app is None:
            return

        style = combine_stylesheets(
            self.style_sheet,
            load_stylesheet("main.qss"),
        )

        app.setStyleSheet(style)

    def print_debug_info(self):
        click_events.print_debug(self.state, self.data)

    def closeEvent(self, event):
        # print("MainWindow closeEvent")
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

        ## --- Build the UI ---
        main_app = QWidget(self)  # MainApp(self)
        layout = QHBoxLayout(main_app)
        layout.setContentsMargins(4, 4, 4, 4)

        layout.addWidget(splitter)

        self.setCentralWidget(main_app)

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
                lambda remove=remove, batch=batch, whole_neurons=whole_neurons:
                self._curate_from_shortcut(
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

        if not components:
            return

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

        if batch or len(targets) > 1:
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
                f"{action} {scope}?\n\n"
                f"{consequence}\nSource files are unchanged.",
                QMessageBox.StandardButton.Ok
                | QMessageBox.StandardButton.Cancel,
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
