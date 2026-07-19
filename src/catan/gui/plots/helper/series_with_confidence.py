import numpy as np
from vispy.scene import visuals


class SeriesVisual:
    def __init__(
        self,
        parent,
        *,
        line_color=(1.0, 1.0, 1.0, 1.0),
        band_color=(1.0, 1.0, 1.0, 0.18),
        marker_size=7,
        order=0,
    ):
        self.parent = parent
        self.line_color = line_color
        self.band_color = band_color
        self.marker_size = marker_size

        self.band = visuals.Mesh(parent=parent)
        self.line = visuals.Line(parent=parent)
        self.markers = visuals.Markers(parent=parent)

        self.band.order = order
        self.line.order = order + 10
        self.markers.order = order + 20

        for visual in (self.band, self.line, self.markers):
            visual.set_gl_state(
                "translucent",
                depth_test=False,
                blend=True,
            )

    def set_data(self, session_ids, values, errors_low=None, errors_high=None):
        x = np.asarray(session_ids, dtype=np.float32)
        y = np.asarray(values, dtype=np.float32)

        # print(
        #     f"Setting data for SeriesVisual: x={x}, y={y}, errors_low={errors_low}, errors_high={errors_high}"
        # )
        finite = np.isfinite(x) & np.isfinite(y)

        if errors_low is not None and errors_high is not None:
            err_low = np.asarray(errors_low, dtype=np.float32)
            err_high = np.asarray(errors_high, dtype=np.float32)

            finite &= np.isfinite(err_low) & np.isfinite(err_high)

        x = x[finite]
        y = y[finite]

        if x.size == 0:
            self.clear()
            return

        # Sort by x/session order.
        order = np.argsort(x)
        x = x[order]
        y = y[order]

        pos = np.column_stack([x, y]).astype(np.float32)

        self.line.set_data(
            pos=pos,
            color=self.line_color,
            width=2.0,
        )

        self.markers.set_data(
            pos=pos,
            size=self.marker_size,
            face_color=self.line_color,
            edge_color=self.line_color,
            edge_width=1.0,
        )

        if errors_low is None or errors_high is None:
            self.band.visible = False
            return

        err_low = err_low[finite][order]
        err_high = err_high[finite][order]

        lower = y - err_low
        upper = y + err_high

        self._set_error_band(x, lower, upper)

    def _set_error_band(self, x, lower, upper):
        n = x.size

        if n < 2:
            self.band.visible = False
            return

        vertices = np.empty((2 * n, 3), dtype=np.float32)

        # First half: upper curve
        vertices[:n, 0] = x
        vertices[:n, 1] = upper
        vertices[:n, 2] = 0.0

        # Second half: lower curve
        vertices[n:, 0] = x
        vertices[n:, 1] = lower
        vertices[n:, 2] = 0.0

        faces = []

        for i in range(n - 1):
            u0 = i
            u1 = i + 1
            l0 = n + i
            l1 = n + i + 1

            faces.append((u0, l0, u1))
            faces.append((u1, l0, l1))

        faces = np.asarray(faces, dtype=np.uint32)

        colors = np.tile(
            np.asarray(self.band_color, dtype=np.float32),
            (vertices.shape[0], 1),
        )

        self.band.set_data(
            vertices=vertices,
            faces=faces,
            vertex_colors=colors,
        )

        self.band.visible = True

    def clear(self):
        self.band.visible = False
        self.line.set_data(pos=np.zeros((0, 2), dtype=np.float32))
        self.markers.set_data(
            pos=np.zeros((0, 2), dtype=np.float32),
            size=self.marker_size,
            face_color=self.line_color,
            edge_color=self.line_color,
        )

    def destroy(self):
        print("destroy errorplot")
        for visual in (self.band, self.line, self.markers):
            # try:
            visual.visible = False
            visual.parent = None
            visual.destroy()
            # except Exception as e:
            # pass
