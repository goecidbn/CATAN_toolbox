from PySide6.QtCore import (
    QObject,
    QEvent,
    QPoint,
    QRect,
    QTimer,
)
from PySide6.QtWidgets import QWidget


class PopupBoundsFilter(QObject):
    """
    Keep a popup inside a given widget/window.

    If an anchor widget is given, prefer opening below it and,
    if there is insufficient room, open above it instead.
    """

    def __init__(
        self,
        popup: QWidget,
        *,
        anchor: QWidget | None = None,
        bounds: QWidget | None = None,
        margin: int = 4,
    ):
        super().__init__(popup)

        self.popup = popup
        self.anchor = anchor

        if bounds is None:
            if anchor is not None:
                bounds = anchor.window()
            else:
                bounds = popup.window()

        self.bounds = bounds
        self.margin = margin

        popup.installEventFilter(self)

    def eventFilter(self, watched, event):
        if watched is self.popup and event.type() == QEvent.Type.Show:

            # Let Qt finish calculating the popup's actual size/position
            # first, then correct it.
            QTimer.singleShot(
                0,
                self.reposition,
            )

        return super().eventFilter(
            watched,
            event,
        )

    def reposition(self):
        popup = self.popup
        bounds = self.bounds

        if popup is None or bounds is None:
            return

        popup.adjustSize()

        popup_size = popup.size()

        bounds_top_left = bounds.mapToGlobal(QPoint(0, 0))

        bounds_rect = QRect(
            bounds_top_left,
            bounds.size(),
        ).adjusted(
            self.margin,
            self.margin,
            -self.margin,
            -self.margin,
        )

        # ------------------------------------------------
        # Preferred position
        # ------------------------------------------------

        if self.anchor is not None:

            anchor_top_left = self.anchor.mapToGlobal(QPoint(0, 0))

            anchor_bottom_left = self.anchor.mapToGlobal(
                QPoint(
                    0,
                    self.anchor.height(),
                )
            )

            x = anchor_bottom_left.x()
            y = anchor_bottom_left.y()

            # Not enough space below -> open above.
            if y + popup_size.height() > bounds_rect.bottom():
                y = anchor_top_left.y() - popup_size.height()

        else:
            # Preserve Qt's chosen location as far as possible.
            current = popup.pos()
            x = current.x()
            y = current.y()

        # ------------------------------------------------
        # Clamp to CATAN window
        # ------------------------------------------------

        x = max(
            bounds_rect.left(),
            min(
                x,
                bounds_rect.right() - popup_size.width() + 1,
            ),
        )

        y = max(
            bounds_rect.top(),
            min(
                y,
                bounds_rect.bottom() - popup_size.height() + 1,
            ),
        )

        popup.move(x, y)


def constrain_popup(
    popup: QWidget,
    *,
    anchor: QWidget | None = None,
    bounds: QWidget | None = None,
    margin: int = 4,
):
    """
    Configure a popup to remain within a window.

    The returned object does not need to be stored manually:
    it is parented to `popup`.
    """
    return PopupBoundsFilter(
        popup,
        anchor=anchor,
        bounds=bounds,
        margin=margin,
    )
