import numpy as np
from vispy.scene import visuals


class HistogramMeshLayer:
    def __init__(
        self,
        parent,
        *,
        order=0.0,
        edge_order=None,
        edge_color=(0, 0, 0, 0.8),
        edge_width=1.0,
        draw_edges=True,
    ):
        self.mesh = visuals.Mesh(parent=parent)
        self.mesh.set_gl_state(
            blend=True,
            depth_test=False,
            blend_func=("src_alpha", "one_minus_src_alpha"),
        )
        self.mesh.order = order

        self.draw_edges = draw_edges
        self.edge_color = edge_color
        self.edge_width = edge_width

        if edge_order is None:
            edge_order = order + 1.0

        if draw_edges:
            self.edge_line = visuals.Line(
                parent=parent,
                color=edge_color,
                width=edge_width,
                connect="segments",
            )
            self.edge_line.set_gl_state(
                blend=True,
                depth_test=False,
                blend_func=("src_alpha", "one_minus_src_alpha"),
            )
            self.edge_line.order = edge_order
        else:
            self.edge_line = None

        self.n_bins = 0
        self.bin_edges = None
        self.vertices = None
        self.faces = None
        self.colors = None
        self.edge_vertices = None

    def build_from_counts(self, bin_edges, counts, bin_colors):
        bin_edges = np.asarray(bin_edges, dtype=np.float32)
        counts = np.asarray(counts, dtype=np.float32)

        self.bin_edges = bin_edges
        self.n_bins = len(counts)

        x0 = bin_edges[:-1]
        x1 = bin_edges[1:]
        y0 = np.zeros_like(counts)
        y1 = counts

        vertices = np.zeros((self.n_bins, 4, 3), dtype=np.float32)

        # bottom left
        vertices[:, 0, 0] = x0
        vertices[:, 0, 1] = y0

        # bottom right
        vertices[:, 1, 0] = x1
        vertices[:, 1, 1] = y0

        # top right
        vertices[:, 2, 0] = x1
        vertices[:, 2, 1] = y1

        # top left
        vertices[:, 3, 0] = x0
        vertices[:, 3, 1] = y1

        faces_template = np.asarray(
            [[0, 1, 2], [0, 2, 3]],
            dtype=np.uint32,
        )
        offsets = 4 * np.arange(self.n_bins, dtype=np.uint32)
        faces = faces_template[None, :, :] + offsets[:, None, None]

        bin_colors = np.asarray(bin_colors, dtype=np.float32)

        if bin_colors.ndim == 1:
            bin_colors = np.tile(bin_colors, (self.n_bins, 1))

        self.vertices = vertices.reshape(-1, 3)
        self.faces = faces.reshape(-1, 3)
        self.colors = np.repeat(bin_colors, 4, axis=0)

        self._upload_mesh()

        if self.draw_edges:
            self._build_edge_vertices(counts)
            self._upload_edges()

    def _build_edge_vertices(self, counts):
        counts = np.asarray(counts, dtype=np.float32)

        x0 = self.bin_edges[:-1]
        x1 = self.bin_edges[1:]
        y0 = np.zeros_like(counts)
        y1 = counts

        # 4 line segments per bin:
        # bottom, right, top, left
        edges = np.zeros((self.n_bins, 8, 3), dtype=np.float32)

        # bottom: (x0,0) -> (x1,0)
        edges[:, 0, 0] = x0
        edges[:, 0, 1] = y0
        edges[:, 1, 0] = x1
        edges[:, 1, 1] = y0

        # right: (x1,0) -> (x1,y1)
        edges[:, 2, 0] = x1
        edges[:, 2, 1] = y0
        edges[:, 3, 0] = x1
        edges[:, 3, 1] = y1

        # top: (x1,y1) -> (x0,y1)
        edges[:, 4, 0] = x1
        edges[:, 4, 1] = y1
        edges[:, 5, 0] = x0
        edges[:, 5, 1] = y1

        # left: (x0,y1) -> (x0,0)
        edges[:, 6, 0] = x0
        edges[:, 6, 1] = y1
        edges[:, 7, 0] = x0
        edges[:, 7, 1] = y0

        self.edge_vertices = edges.reshape(-1, 3)

    def set_counts(self, counts):
        counts = np.asarray(counts, dtype=np.float32)

        if len(counts) != self.n_bins:
            raise ValueError(f"Expected {self.n_bins} counts, got {len(counts)}")

        # update mesh top vertices
        self.vertices[2::4, 1] = counts
        self.vertices[3::4, 1] = counts
        self._upload_mesh()

        if self.draw_edges:
            self._build_edge_vertices(counts)
            self._upload_edges()

    def set_bin_colors(self, bin_colors):
        bin_colors = np.asarray(bin_colors, dtype=np.float32)

        if bin_colors.shape != (self.n_bins, 4):
            raise ValueError(
                f"Expected bin_colors shape {(self.n_bins, 4)}, got {bin_colors.shape}"
            )

        self.colors[:] = np.repeat(bin_colors, 4, axis=0)
        self._upload_mesh()

    def _upload_mesh(self):
        self.mesh.set_data(
            vertices=self.vertices,
            faces=self.faces,
            vertex_colors=self.colors,
        )

    def _upload_edges(self):
        if self.edge_line is None:
            return

        self.edge_line.set_data(
            pos=self.edge_vertices,
            color=self.edge_color,
            width=self.edge_width,
            connect="segments",
        )

    def set_visible(self, visible: bool):
        self.mesh.visible = bool(visible)
        if self.edge_line is not None:
            self.edge_line.visible = bool(visible)

    def destroy(self):
        self.mesh.parent = None

        if self.edge_line is not None:
            self.edge_line.parent = None
            self.edge_line = None

        self.vertices = None
        self.faces = None
        self.colors = None
        self.edge_vertices = None
        self.bin_edges = None
        self.n_bins = 0
