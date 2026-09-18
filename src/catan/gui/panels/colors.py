import numpy as np
from vispy.color import get_colormap


class CyclicColorMap:
    def __init__(self, n_colors=10, cmap_name="viridis"):
        self.n_colors = n_colors
        self.cmap = get_colormap(cmap_name)

        # sample evenly from the colormap
        vals = np.linspace(0, 1, n_colors, endpoint=False)

        # RGBA colors
        self.colors = self.cmap.map(vals).astype(np.float32)

        self.idx = 0

    def next(self):
        color = self.colors[self.idx]
        self.idx = (self.idx + 1) % self.n_colors
        return color

    def get(self, i):
        """Deterministic color by index."""
        return self.colors[i % self.n_colors]

    def reset(self):
        self.idx = 0
