from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QWidgetAction,
    QWidget,
    QHBoxLayout,
    QLabel,
)


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
