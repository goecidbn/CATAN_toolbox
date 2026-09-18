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
        gap: int = 4,
        placements: tuple[str, ...] = ("below", "above"),
    ):
        super().__init__(popup)

        self.popup = popup
        self.anchor = anchor

        self.placements = placements
        self.gap = gap

        if bounds is None:
            if anchor is not None:
                bounds = anchor.window()
            else:
                bounds = popup.window()

        self.bounds = bounds
        self.margin = margin

        self._reposition_pending = False

        popup.installEventFilter(self)

    def eventFilter(self, watched, event):

        if watched is self.popup and event.type() in (
            QEvent.Type.Show,
            QEvent.Type.LayoutRequest,
        ):
            self.schedule_reposition()

        return super().eventFilter(
            watched,
            event,
        )

    def schedule_reposition(self):
        if self._reposition_pending:
            return

        self._reposition_pending = True
        QTimer.singleShot(
            0,
            self._do_scheduled_reposition,
        )

    def _do_scheduled_reposition(self):
        self._reposition_pending = False
        self.reposition()

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

            anchor_rect = QRect(anchor_top_left, self.anchor.size())

            def candidate(placement: str) -> QPoint:

                if placement == "below":
                    return QPoint(
                        anchor_rect.left(),
                        anchor_rect.bottom() + 1 + self.gap,
                    )

                if placement == "above":
                    return QPoint(
                        anchor_rect.left(),
                        anchor_rect.top() - popup_size.height() - self.gap,
                    )

                if placement == "right":
                    return QPoint(
                        anchor_rect.right() + 1 + self.gap,
                        anchor_rect.top(),
                    )

                if placement == "left":
                    return QPoint(
                        anchor_rect.left() - popup_size.width() - self.gap,
                        anchor_rect.top(),
                    )

                raise ValueError(f"Unknown popup placement: {placement!r}")

            # Use the first preferred placement which fits
            # completely into the requested bounds.
            x = y = None

            for placement in self.placements:

                pos = candidate(placement)

                candidate_rect = QRect(pos, popup_size)

                if bounds_rect.contains(candidate_rect):
                    x = pos.x()
                    y = pos.y()
                    break

            # Nothing fits perfectly:
            # start from the first preference and clamp below.
            if x is None:
                pos = candidate(self.placements[0])
                x = pos.x()
                y = pos.y()

        else:
            # Preserve Qt's / caller's chosen location as far as possible.
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
    gap: int = 4,
    placements: tuple[str, ...] = ("below", "above"),
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
        gap=gap,
        placements=placements,
    )
