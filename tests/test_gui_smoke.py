import os, pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")


@pytest.mark.gui
def test_main_window_can_be_created():
    from PySide6.QtWidgets import QApplication
    from catan.gui.GUI_elements.main_window import MainWindow

    app = QApplication.instance() or QApplication([])

    window = MainWindow()

    assert window is not None

    window.close()