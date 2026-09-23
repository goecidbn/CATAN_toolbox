import numpy as np

from PySide6.QtCore import Signal, QPoint
from PySide6.QtGui import QAction

from PySide6.QtWidgets import (
    QMenu,
    QToolButton,
)

from catan.tracking.structures import ReviewStatus


class StayOpenMenu(QMenu):

    def mouseReleaseEvent(self, event):

        action = self.actionAt(event.position().toPoint())

        if action is not None and action.isEnabled() and action.isCheckable():
            action.setChecked(not action.isChecked())

            # Important: don't let QMenu handle
            # the release, otherwise it closes.
            return

        super().mouseReleaseEvent(event)


class ReviewStatusFilter(QToolButton):

    changed = Signal()

    def __init__(self, parent=None):
        super().__init__(parent)

        self.menu = StayOpenMenu(self)
        self._custom = False

        self.clicked.connect(self._show_review_menu)

        self.actions = {}

        for status in ReviewStatus:
            action = QAction(status.label, self.menu)
            action.setCheckable(True)
            action.setChecked(True)

            action.toggled.connect(self._on_changed)

            self.menu.addAction(action)
            self.actions[status] = action

        self._update_text()

    @property
    def visible_statuses(self):
        return {status for status, action in self.actions.items() if action.isChecked()}

    def _on_changed(self):
        # User interacted with the review-state selector,
        # so selection is review-state based again.
        self._custom = False

        self._update_text()
        self.changed.emit()

    def set_custom(self, custom: bool = True):

        if self._custom == custom:
            return

        self._custom = custom
        self._update_text()

    def _update_text(self):

        if self._custom:
            self.setText("Review: Custom")
            return

        visible = self.visible_statuses

        if len(visible) == len(ReviewStatus):
            text = "Review: All"

        elif not visible:
            text = "Review: None"

        elif len(visible) == 1:
            status = next(iter(visible))
            text = f"Review: {status.label}"

        else:
            text = f"Review: {len(visible)}/{len(ReviewStatus)}"

        self.setText(text)

    def _show_review_menu(self):

        menu = self.menu

        menu.ensurePolished()
        menu_size = menu.sizeHint()

        # Align left edges, directly above button
        pos = self.mapToGlobal(QPoint(0, -menu_size.height()))

        window = self.window()

        if window is not None:
            window_rect = window.frameGeometry()

            if pos.y() < window_rect.top():
                pos = self.mapToGlobal(QPoint(0, self.height()))

        menu.popup(pos)


def neuron_mask(
    neuron_ids,
    review_status,
    visible_statuses,
    *,
    focused_neuron_id=None,
    keep_focused=False,
):
    neuron_ids = np.asarray(neuron_ids, dtype=int)

    review_status = np.asarray(review_status)

    visible_values = {int(status) for status in visible_statuses}

    mask = np.isin(review_status[neuron_ids], list(visible_values))

    if keep_focused and focused_neuron_id is not None:
        mask |= neuron_ids == int(focused_neuron_id)

    return mask
