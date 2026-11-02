"""Axes linked to a vispy ViewBox that stay correct after layout changes."""


def link_axis(axis, view):
    """``axis.link_view(view)``, also refreshing the tick range when the axis
    or the view is resized or moved.

    vispy updates a linked axis' range only when the view's camera
    transform changes; the range depends on the axis' own geometry too, so
    after the grid laid out the widgets (e.g. when an inset is first shown)
    the ticks stayed wrong until the camera was moved.
    """
    from PySide6.QtCore import QTimer

    axis.link_view(view)
    pending = {"queued": False}

    def update():
        pending["queued"] = False
        if axis._linked_view is view:
            axis._view_changed()

    def refresh(event=None):
        update()
        # The grid places its widgets one after another; a refresh during
        # that pass can see a half-updated layout (first show of a grid
        # with an axis column: the bottom axis kept an offset range).
        # Repeat it once the layout pass is over.
        if not pending["queued"]:
            pending["queued"] = True
            QTimer.singleShot(0, update)

    for node in (axis, view):
        node.events.resize.connect(refresh)
        node.events.transform_change.connect(refresh)

    refresh()
