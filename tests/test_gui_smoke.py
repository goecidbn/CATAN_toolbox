import pytest


@pytest.fixture(scope="session")
def qapp():
    from PySide6.QtWidgets import QApplication

    app = QApplication.instance()

    if app is None:
        app = QApplication(["catan-test"])

    return app


@pytest.mark.gui
def test_gui_application_imports(qapp):
    from PySide6.QtWidgets import QApplication
    from catan.gui.app import main

    assert QApplication.instance() is qapp
    assert callable(main)


@pytest.mark.gui
def test_basic_qt_widget_can_be_created(qapp):
    from PySide6.QtWidgets import QWidget

    widget = QWidget()

    assert widget.isWidgetType()
    assert not widget.isVisible()

    widget.close()
    widget.deleteLater()
    qapp.processEvents()