"""Axes linked to a vispy ViewBox that stay correct after layout changes."""


def link_axis(axis, view):
    """``axis.link_view(view)``, also refreshing the tick range when the axis
    or the view is resized or moved.

    vispy updates a linked axis' range only when the view's camera
    transform changes; the range depends on the axis' own geometry too, so
    after the grid laid out the widgets (e.g. when an inset is first shown)
    the ticks stayed wrong until the camera was moved.
    """
    axis.link_view(view)

    def refresh(event=None):
        if axis._linked_view is view:
            axis._view_changed()

    for node in (axis, view):
        node.events.resize.connect(refresh)
        node.events.transform_change.connect(refresh)

    refresh()
