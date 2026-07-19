from PySide6.QtWidgets import QWidget, QHBoxLayout, QSplitter
from PySide6.QtCore import Qt


from .main_menu import MainMenu
from .display_area import DisplayArea


class MainApp(QWidget):

    def __init__(self, parent):
        super().__init__(parent)

        self.state = parent.state
        self.data = parent.data

        layout = QHBoxLayout(self)
        layout.setContentsMargins(4, 4, 4, 4)
        # central = QWidget(self)
        splitter = QSplitter(Qt.Horizontal)
        splitter.setChildrenCollapsible(False)
        # layout.setSpacing(6)

        # =========== LEFT: SIDE MENU ===========
        main_menu = MainMenu(self)
        splitter.addWidget(main_menu)

        # =========== RIGHT: DISPLAY AREA ===========
        display_area = DisplayArea(self)
        splitter.addWidget(display_area)
        layout.addWidget(splitter)

        self.exposes = {
            "main_menu": main_menu,
            **main_menu.exposes,
            "display_area": display_area,
            **display_area.exposes,
        }
