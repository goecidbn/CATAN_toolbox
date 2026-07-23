import importlib
from importlib.resources import files

from PySide6.QtCore import QCoreApplication, QSettings
from PySide6.QtGui import QAction, QFont, Qt
from PySide6.QtWidgets import (
    QApplication,
    QHBoxLayout,
    QMainWindow,
    QSplitter,
    QWidget,
)

from catan.gui.resources import (
    combine_stylesheets,
    load_stylesheet,
)
from catan.gui.structures import AppState, Data
from catan.gui.interaction import click_events

from .display_area import DisplayArea
from .main_menu import MainMenu

class MainWindow(QMainWindow):

    settings = QSettings()
    state = AppState()
    # data = Data()

    def __init__(self):
        super().__init__()

        self.data: Data = Data(self.state)

        # self.settings = QSettings()
        self._restore_settings()

        self.setWindowTitle("CATAN - Curating and Tracking Neurons")

        QCoreApplication.setOrganizationName("WolfLabs")
        QCoreApplication.setApplicationName("CATAN")

        # --- UI setup ---
        self._init_ui()

        """
        connect interaction (make reloadable)
        """
        # --- setting up reload logic ---
        reload_action = QAction("Reload plotting logic", self)
        reload_action.setShortcut("Ctrl+R")
        reload_action.triggered.connect(self.reload_logic)
        self.menuBar().addAction(reload_action)

        print_debug = QAction("Print debug info", self)
        print_debug.setShortcut("Ctrl+D")
        print_debug.triggered.connect(self.print_debug_info)
        # print_debug.triggered.connect(
        #     self.gui_elements["primary_display"].print_debug_info
        # )
        self.menuBar().addAction(print_debug)

        app = QApplication.instance()
        if app is None:
            return
        self.style_sheet = app.styleSheet()

        self.reset_stylesheet()

    def _restore_settings(self):

        # window geometry
        geom = self.settings.value("window/geometry")
        if geom is not None:
            self.restoreGeometry(geom)

    def _save_settings(self):

        self.settings.setValue("window/geometry", self.saveGeometry())
        self.settings.sync()

    def reload_logic(self):
        self.state.data_version += 1

        importlib.reload(click_events)

        self.gui_elements["display_area"].rebuild()
        self.gui_elements["main_menu"].rebuild()
        self.reset_stylesheet()

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
        self._save_settings()
        for gui in self.gui_elements.values():
            if hasattr(gui, "_save_settings"):
                gui._save_settings()
        # self.gui_elements._save_settings()
        super().closeEvent(event)

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
        display_area = DisplayArea(self)
        splitter.addWidget(display_area)

        ## --- Build the UI ---
        main_app = QWidget(self)  # MainApp(self)
        layout = QHBoxLayout(main_app)
        layout.setContentsMargins(4, 4, 4, 4)

        layout.addWidget(splitter)

        self.setCentralWidget(main_app)

        self.gui_elements = {
            "main_menu": main_menu,
            "display_area": display_area,
        }
