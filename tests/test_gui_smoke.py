import os
import sys
import pytest


@pytest.fixture(scope="session")
def qapp():
    if sys.platform.startswith("linux"):
        os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

    print("Creating QApplication...")
    print("platform:", sys.platform)
    print("QT_QPA_PLATFORM =", os.environ.get("QT_QPA_PLATFORM"))

    from PySide6.QtWidgets import QApplication

    app = QApplication.instance()

    print("Existing app:", app)

    if app is None:
        print("Constructing QApplication...")
        app = QApplication(["catan-test"])
        print("Done.")
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