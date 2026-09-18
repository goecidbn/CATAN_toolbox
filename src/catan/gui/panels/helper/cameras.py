from vispy import scene
from vispy.geometry import Rect


class XOnlyLockedPanZoomCamera(scene.PanZoomCamera):

    def __init__(self, y0=0.0, h0=1.0, on_changed=None, **kwargs):
        super().__init__(**kwargs)
        self._y0 = float(y0)
        self._h0 = float(h0)
        self._on_changed = on_changed

    def _notify_axis(self):
        self.view_changed()
        if self._on_changed is not None:
            self._on_changed()

    def set_y_lock_from_current(self):
        # call this once after you set the initial range
        r = self.rect
        if r is None:
            return
        self._y0 = float(r.bottom)
        self._h0 = float(r.height)

    def set_x_locks(self, xmin, xmax):

        self._xmin = float(xmin)
        self._xmax = float(xmax)

    def set_rect(self):
        r = self.rect
        if r is None:
            return

        self.rect = (
            max(r.left, self._xmin),
            self._y0,
            min(r.width, self._xmax - r.left),
            self._h0,
        )
        self._notify_axis()

    def zoom(self, factor, center=None):
        # x-only zoom
        if hasattr(factor, "__len__"):
            fx = float(factor[0])
        else:
            fx = float(factor)
        super().zoom((fx, 1.0), center=center)

        # lock y back
        self.set_rect()

    def pan(self, *args, **kwargs):
        super().pan(*args, **kwargs)
        # lock y back (prevents vertical panning)
        self.set_rect()


class FixedPanZoomCamera(scene.PanZoomCamera):
    def zoom(self, *args, **kwargs):
        return

    def pan(self, *args, **kwargs):
        return

    def set_exact_range(self, x, y):
        xmin, xmax = map(float, x)
        ymin, ymax = map(float, y)

        if xmax <= xmin:
            xmax = xmin + 1.0

        if ymax <= ymin:
            ymax = ymin + 1.0

        self.rect = Rect(
            xmin,
            ymin,
            xmax - xmin,
            ymax - ymin,
        )

        self.view_changed()

        # if "x" in self.axes:
        #     self.axes["x"]._view_changed()
        # if "y" in self.axes:
        #     self.axes["y"]._view_changed()
