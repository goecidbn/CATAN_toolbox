import pytest

pytest.importorskip("PySide6")
pytestmark = pytest.mark.gui

from PySide6.QtCore import QEvent, Qt  # noqa: E402
from PySide6.QtGui import QKeyEvent  # noqa: E402
from PySide6.QtWidgets import (  # noqa: E402
    QApplication,
    QLabel,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from catan.gui.GUI_elements.utils.enter_key import EnterKeyButton  # noqa: E402


@pytest.fixture
def window():
    app = QApplication.instance() or QApplication([])
    widget = QWidget()
    layout = QVBoxLayout(widget)
    label = QLabel("view", widget)
    button = QPushButton("Apply", widget)
    layout.addWidget(label)
    layout.addWidget(button)
    widget.show()
    app.processEvents()
    yield app, label, button
    widget.close()


def _press_enter(app, target):
    event = QKeyEvent(QEvent.Type.KeyPress, Qt.Key.Key_Return, Qt.KeyboardModifier.NoModifier)
    app.sendEvent(target, event)


def test_enter_clicks_the_button(window):
    app, label, button = window
    clicks = []
    button.clicked.connect(lambda: clicks.append(1))
    EnterKeyButton(button)
    _press_enter(app, label)
    assert clicks == [1]


def test_blocked_leaves_enter_to_others(window):
    # The curator's "Apply & next" while a manual split / merge is open.
    app, label, button = window
    clicks = []
    button.clicked.connect(lambda: clicks.append(1))
    blocked = [True]
    EnterKeyButton(button, blocked=lambda: blocked[0])
    _press_enter(app, label)
    assert clicks == []
    blocked[0] = False
    _press_enter(app, label)
    assert clicks == [1]
