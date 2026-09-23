from PySide6.QtCore import Qt, Signal, QEvent
from PySide6.QtWidgets import (
    QButtonGroup,
    QHBoxLayout,
    QWidget,
    QVBoxLayout,
    QToolButton,
    QFormLayout,
    QApplication,
)
from catan.gui.panels.helper import ReviewStatusFilter


class ControlPanel(QWidget):

    display_parameter_changed = Signal()
    overlay_layout_changed = Signal()

    def __init__(self, parent):
        super().__init__(parent)

        self.state = parent.state
        self.data = parent.data

        root_layout = QVBoxLayout(self)
        root_layout.setContentsMargins(4, 4, 4, 4)
        root_layout.setSpacing(3)

        self.toggle_button = QToolButton()
        self.toggle_button.setText("⚙")
        self.toggle_button.setToolTip("Footprint display settings")
        self.toggle_button.setCheckable(True)
        self.toggle_button.setChecked(False)
        self.toggle_button.toggled.connect(self._set_expanded)

        root_layout.addWidget(
            self.toggle_button,
            alignment=Qt.AlignmentFlag.AlignRight,
        )

        self.parameter_body = QWidget()

        form = QFormLayout(self.parameter_body)
        form.setContentsMargins(8, 5, 8, 8)
        form.setHorizontalSpacing(8)
        form.setVerticalSpacing(5)

        root_layout.addWidget(self.parameter_body)

        self.parameter_body.setVisible(False)
        self.form = form

        app = QApplication.instance()

        if app is not None:
            app.installEventFilter(self)

        self.setObjectName("footprintParameterOverlay")

        self.setStyleSheet("""
            QWidget#footprintParameterOverlay {
                background: rgba(35, 39, 45, 235);
                border: 1px solid #59616c;
                border-radius: 6px;
            }

            QWidget#footprintParameterOverlay QLabel,
            QWidget#footprintParameterOverlay QCheckBox {
                color: #e8eaed;
                border: none;
            }

            QWidget#footprintParameterOverlay QToolButton {
                color: #e8eaed;
                background: #343941;
                border: 1px solid #59616c;
                border-radius: 4px;
                padding: 3px 6px;
            }

            QWidget#footprintParameterOverlay QToolButton:hover {
                background: #414751;
            }

            QToolButton:checked {
                background: #59616c;
                border: 1px solid #8a96a6;
            }

            QWidget#footprintParameterOverlay QToolButton#scopeLeft,
            QWidget#footprintParameterOverlay QToolButton#scopeRight,
            QWidget#footprintParameterOverlay QToolButton#scopeCenter {
                color: #e8eaed;
                background: #444a53;

                border: 1px solid #69727f;

                padding: 5px 8px;
                min-height: 22px;
            }

            /* Only the outside edges are rounded */
            QWidget#footprintParameterOverlay QToolButton#scopeLeft {
                border-top-left-radius: 15px;
                border-bottom-left-radius: 15px;

                border-top-right-radius: 1px;
                border-bottom-right-radius: 1px;

                /* avoid doubled border in the middle */
                border-right-width: 0px;
            }

            QWidget#footprintParameterOverlay QToolButton#scopeCenter {
                border-top-left-radius: 1px;
                border-bottom-left-radius: 1px;

                border-top-right-radius: 1px;
                border-bottom-right-radius: 1px;

                /* avoid doubled border in the middle */
                border-right-width: 0px;
            }
            

            QWidget#footprintParameterOverlay QToolButton#scopeRight {
                border-top-left-radius: 1px;
                border-bottom-left-radius: 1px;

                border-top-right-radius: 15px;
                border-bottom-right-radius: 15px;
            }

            /* "popped out" */
            QWidget#footprintParameterOverlay QToolButton#scopeLeft:!checked,
            QWidget#footprintParameterOverlay QToolButton#scopeRight:!checked,
            QWidget#footprintParameterOverlay QToolButton#scopeCenter:!checked {
                background: #4a515b;
                border-style: outset;
            }

            /* "pushed in" */
            QWidget#footprintParameterOverlay QToolButton#scopeLeft:checked,
            QWidget#footprintParameterOverlay QToolButton#scopeRight:checked,
            QWidget#footprintParameterOverlay QToolButton#scopeCenter:checked {
                background: #2d3239;

                border-color: #363b42;
                border-style: inset;

                color: #ffffff;

                /* subtle physical displacement */
                padding-top: 6px;
                padding-bottom: 4px;
            }

            /* Optional hover only for the unselected half */
            QWidget#footprintParameterOverlay QToolButton#scopeLeft:!checked:hover,
            QWidget#footprintParameterOverlay QToolButton#scopeRight:!checked:hover,
            QWidget#footprintParameterOverlay QToolButton#scopeCenter:!checked:hover {
                background: #555d68;
            }
        """)

    def _build_display_scope_selection(
        self, options: dict[str, dict[str, str]], default_scope: str = "adjacent"
    ):

        self.display_scope_group = QButtonGroup(self)
        self.display_scope_group.setExclusive(True)

        scope_widget = QWidget()
        scope_layout = QHBoxLayout(scope_widget)
        scope_layout.setContentsMargins(0, 0, 0, 0)
        scope_layout.setSpacing(0)

        for key, option in options.items():
            button = QToolButton()
            button.setText(option["label"])
            button.setCheckable(True)
            button.setProperty("scope", key)
            button.setObjectName(f"scope{option['position'].capitalize()}")
            button.setMinimumWidth(70)

            self.display_scope_group.addButton(button)
            scope_layout.addWidget(button)

            if key == default_scope:
                button.setChecked(True)

        self.display_scope_group.buttonClicked.connect(self._on_display_scope_changed)
        return scope_widget

    @property
    def display_scope(self) -> str:

        button = self.display_scope_group.checkedButton()
        if button is None:
            return "adjacent"

        return button.property("scope")

    def _on_display_scope_changed(self, button):
        self.display_parameter_changed.emit()

    def _build_review_selector(self):
        self.review_filter = ReviewStatusFilter.ReviewStatusFilter(self)
        self.review_filter.changed.connect(self.display_parameter_changed.emit)
        return self.review_filter

    def _set_expanded(self, expanded: bool):
        self.parameter_body.setVisible(expanded)
        self.adjustSize()

        self.overlay_layout_changed.emit()

    def _on_session_only_changed(self):
        self.display_parameter_changed.emit()

    def eventFilter(self, obj, event):

        if (
            self.toggle_button.isChecked()
            and event.type() == QEvent.Type.MouseButtonPress
        ):
            global_pos = event.globalPosition().toPoint()

            # popup = self.review_filter.menu
            popup = getattr(getattr(self, "review_filter", None), "menu", None)

            if popup is not None and popup.isVisible():
                popup_rect = popup.geometry()
                if popup_rect.contains(global_pos):
                    return super().eventFilter(obj, event)

            local_pos = self.mapFromGlobal(global_pos)
            inside_panel = self.rect().contains(local_pos)

            if not inside_panel:
                self.toggle_button.setChecked(False)
        return super().eventFilter(obj, event)
