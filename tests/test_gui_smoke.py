import os, pytest

# os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")


# @pytest.mark.gui
# def test_main_window_can_be_created():
#     from PySide6.QtWidgets import QApplication
#     from catan.gui.GUI_elements.main_window import MainWindow

#     app = QApplication.instance() or QApplication([])

#     window = MainWindow()

#     assert window is not None

#     window.close()
@pytest.mark.gui
def test_gui_application_imports():
    from PySide6.QtWidgets import QApplication
    from catan.gui.app import main

    assert QApplication is not None
    assert callable(main)

@pytest.mark.gui
def test_basic_qt_widget_can_be_created():
    from PySide6.QtWidgets import QApplication, QWidget

    app = QApplication.instance()
    if app is None:
        app = QApplication(["catan-test"])

    widget = QWidget()
    assert widget.isWidgetType()

    widget.close()
    app.processEvents()