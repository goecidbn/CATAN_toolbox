from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QWidgetAction,
    QWidget,
    QHBoxLayout,
    QLabel,
)

SECTION_STYLE = "color: #8a929c; font-size: 8pt; letter-spacing: 0.5px;"
TITLE_STYLE = "font-weight: 600;"


def add_menu_section(menu, text, *, title=False):
    """Visible section header. The dark theme draws QMenu.addSection as a
    plain separator without its text. ``title``: the menu's heading (bold),
    otherwise a small dimmed group label, separated from the entries above."""
    if not title and not menu.isEmpty():
        menu.addSeparator()

    action = QWidgetAction(menu)
    label = QLabel(text.upper() if not title else text, menu)
    label.setAttribute(Qt.WA_TransparentForMouseEvents, True)
    label.setContentsMargins(12, 5 if title else 3, 12, 3 if title else 1)
    label.setStyleSheet(TITLE_STYLE if title else SECTION_STYLE)
    action.setDefaultWidget(label)
    action.setEnabled(False)
    menu.addAction(action)
    if title:
        menu.addSeparator()
    return action


def fit_message_box_buttons(box, padding=36):
    """Make every button of a message box wide enough for its label.

    The dark theme's padding clips long labels otherwise. Applied
    automatically when a message box is shown (message_boxes.py); calling it
    before showing is still fine.
    """
    for button in box.buttons():
        # Measure with the final style and font (stylesheet resolved).
        button.ensurePolished()
        text = button.text().replace("&&", "\0").replace("&", "").replace("\0", "&")
        width = button.fontMetrics().horizontalAdvance(text) + padding

        if not button.icon().isNull():
            width += button.iconSize().width() + 6

        width = max(width, button.sizeHint().width())
        button.setMinimumWidth(max(button.minimumWidth(), width))


def get_colored_label(text, value, cmap, additional_tags=[]):
    color = cmap[value].rgba[0]
    color_hex = "#{:02x}{:02x}{:02x}".format(
        int(color[0] * 255), int(color[1] * 255), int(color[2] * 255)
    )
    html_str = f"<span style='color:{color_hex}'>{text}</span>"
    for tag in additional_tags:
        html_str = f"<{tag}>{html_str}</{tag}>"

    return html_str


def add_label_to_menu(menu, html, *, enabled=False):
    action = QWidgetAction(menu)

    row = QWidget(menu)
    row.setAttribute(Qt.WA_TransparentForMouseEvents, not enabled)

    # left padding roughly matching QAction text indentation
    layout = QHBoxLayout(row)
    layout.setContentsMargins(15, 3, 12, 3)
    layout.setSpacing(0)

    label = QLabel(row)
    label.setTextFormat(Qt.RichText)
    label.setText(html)
    # label.setStyleSheet("color: black;")

    action.setEnabled(True)

    layout.addWidget(label)
    layout.addStretch()

    action.setDefaultWidget(row)
    menu.addAction(action)
    return action
