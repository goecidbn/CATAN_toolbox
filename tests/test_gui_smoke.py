import os, pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtWidgets import QApplication

from catan.gui import MainWindow


@pytest.mark.gui
def test_main_window_can_be_created():
    app = QApplication.instance() or QApplication([])

    window = MainWindow()

    assert window is not None

    window.close()