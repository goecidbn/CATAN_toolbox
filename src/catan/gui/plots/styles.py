import numpy as np
from vispy import color
from typing import Optional


class Styles:

    opts = {
        "background": {
            "cmap": "Greys",
            "alpha": 0.2,
            "width": 1.0,
            "size": 2.0,
            "edge_width": 0.0,
        },
        "default": {
            "cmap": "Greys",
            "alpha": 0.3,
            "width": 2.0,
            "size": 2.0,
            "edge_width": 0.0,
        },
        "hovered": {
            "cmap": "Blues",
            "alpha": 0.8,
            "width": 2.0,
            "size": 2.0,
            "edge_width": 1.5,
        },
        "selected": {
            "cmap": "Greens",
            "alpha": 0.8,
            "width": 2.0,
            "size": 3.0,
            "edge_width": 1.5,
        },
        "focused": {
            "cmap": "Greens",
            "alpha": 1.0,
            "width": 3.0,
            "size": 3.0,
            "edge_width": 1.5,
        },
        "highlighted": {
            "cmap": "Oranges",
            "alpha": 1.0,
            "width": 3.0,
            "size": 3.0,
            "edge_width": 1.5,
        },
    }

    ## all opts aside from color necessaary to specify the according style
    necessary_opts = {
        "line": ["width"],
        "marker": ["size", "edge_width"],
        "bar": ["border_color"],
        "mesh": [],
    }

    color_names = {
        "line": "color",
        "marker": "face_color",
        "bar": "color",
        "mesh": "vertex_colors",
    }

    def __init__(self):

        self.bg_color = "#f7f8fa"
        # pass

    def SESSION_COLORS(self, session_ids):
        pass

    def get_color_array(
        self,
        style: str,
        values: np.ndarray | float = 0.7,
        **kwargs,
    ):
        """
        need cmap & alpha
        * for footprint highlights (need only that)
        """
        if not isinstance(values, np.ndarray):
            values = np.array(values, dtype=np.float32)

        if (color_array := kwargs.get("colors", None)) is None:

            cmap_name = kwargs.get("cmap_name", self.opts[style]["cmap"])
            cmap = color.get_colormap(cmap_name)
            color_array = cmap.map(values)

        alpha = kwargs.get("alpha", self.opts[style]["alpha"])

        # print(f"building color array with cmap {cmap_name} and alpha {alpha}")
        color_array[..., 3] = alpha

        return color_array.astype(np.float32)

    def get_plot_options(
        self, style: str, plot_type: str, values: np.ndarray | float, **kwargs
    ):
        """
        return dict of options for a given style and plot type
        """

        plot_options = {}
        for n_opt in self.necessary_opts[plot_type]:
            plot_options[n_opt] = kwargs.get(n_opt, self.opts[style].get(n_opt))

        colors = self.get_color_array(style, values, **kwargs)

        plot_options[self.color_names[plot_type]] = colors

        return plot_options


# traces
display_style = {
    "background": {"cmap": "Greys", "alpha": 0.1},
    "default": {"cmap": "Greys", "alpha": 0.6, "width": 2.0},
    "hovered": {"cmap": "Blues", "alpha": 1.0, "width": 2.0},
    "focused": {"cmap": "Greens", "alpha": 1.0, "width": 3.0},
    "highlighted": {"cmap": "Oranges", "alpha": 1.0, "width": 3.0},
}

# stats_scatter
marker_styles = {
    "default": {"face_color": (0.5, 0.5, 0.5, 0.6), "size": 5.0, "edge_width": 0.0},
    "default_unselected": {
        "face_color": (0.0, 0.0, 0.0, 0.6),
        "size": 5.0,
        "edge_width": 0.0,
    },
    "selected": {
        "face_color": (1.0, 0.0, 0.0, 1.0),
        "size": 8.0,
        "edge_width": 1.5,
    },
    "hovered": {
        "face_color": (0.0, 1.0, 0.0, 1.0),
        "size": 10.0,
        "edge_width": 1.5,
    },
    "focused": {
        "face_color": (1.0, 0.0, 0.0, 1.0),
        "size": 8.0,
        "edge_width": 1.5,
    },
}

# stats_histo
bar_styles = {
    "default": {"cmap": "Greys", "alpha": 0.3, "border_color": "black"},
    "default_unselected": {"cmap": "Greys", "alpha": 0.6, "border_color": "black"},
    "hovered": {"cmap": "Blues", "alpha": 0.6, "border_color": "black"},
    "selected": {"cmap": "Greens", "alpha": 1.0, "border_color": "black"},
    "focused": {"cmap": "Greens", "alpha": 1.0, "border_color": "black"},
    "highlighted": {"cmap": "Oranges", "alpha": 1.0, "border_color": "black"},
}

# overview
footprint_style = {
    "background": {"alpha": 0.4, "size": 1},
    "default": {"alpha": 0.2, "size": 2.0},
    "hovered": {"color": "Blues", "alpha": 0.5, "size": 3.0},
    "selected": {"color": "Greens", "alpha": 0.8, "size": 3.0},
    "focused": {
        "color": "Greens",
        "alpha": 1.0,
        "size": 3.0,
    },
    "highlighted": {"color": "Oranges", "alpha": 1.0, "size": 3.0},
}


def colormap(
    values: np.ndarray,
    base_color: list | tuple | str = "viridis",
    alpha_scale: float = 1.0,
    offset=0.6,
):
    """
    values: array-like, assumed normalized to [0,1]
    returns: (N,4) RGBA array
    """
    # print(f"cmap which: {which}")
    val_min, val_max = np.percentile(values, [5, 95])
    v = (values - val_min) / (val_max - val_min + 1e-8)
    v = np.clip(v, 0.0, 1.0)

    # optional offset (keeps low values visible)
    if offset > 0:
        v = offset + (1.0 - offset) * v

    if isinstance(base_color, str):
        # map to RGBA using cmap
        cmap = color.get_colormap(base_color)
        rgba = cmap.map(v).astype(np.float32)

        # control alpha separately (very useful for overlap)
        rgba[:, 3] *= alpha_scale

        return rgba
    else:
        rgba = np.tile(base_color, (len(v), 1))
        rgba[:, 3] = v * alpha_scale
        return rgba.astype(np.float32)
