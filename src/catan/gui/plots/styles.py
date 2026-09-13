import numpy as np
from vispy import color
from typing import Optional


class Styles:

    opts = {
        "background": {
            "cmap": "Greys",
            "alpha": 0.2,
            "width": 1.0,
            "size": 1.1,
            "edge_width": 0.0,
        },
        "default": {
            "cmap": "Greys",
            "alpha": 0.3,
            "width": 2.0,
            "size": 1.5,
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

    overview = {
        "tracked_current_alpha": 0.85,
        "tracked_other_color": "#808080",
        "tracked_other_alpha": 0.18,
        "statistic_nan_color": "#686868",
        "statistic_nan_alpha": 0.65,
        "tracked_stat_other_alpha_scale": 0.30,
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
        if not isinstance(values, np.ndarray):
            values = np.asarray(values, dtype=np.float32)

        colors = kwargs.get("colors", None)
        alpha = kwargs.get("alpha", None)

        if colors is None:
            cmap_name = kwargs.get(
                "cmap_name",
                self.opts[style]["cmap"],
            )
            cmap = color.get_colormap(cmap_name)

            color_array = np.asarray(
                cmap.map(values),
                dtype=np.float32,
            ).copy()

            color_array[..., 3] = self.opts[style]["alpha"] if alpha is None else alpha

        else:
            color_array = np.asarray(
                colors,
                dtype=np.float32,
            ).copy()

            # Preserve alpha only for per-item RGBA.
            per_item_rgba = color_array.ndim == 2 and color_array.shape[-1] == 4

            if not per_item_rgba:
                # Single color -> stylesheet still controls alpha.
                color_array[..., 3] = (
                    self.opts[style]["alpha"] if alpha is None else alpha
                )

            elif alpha is not None:
                # Explicit alpha always wins.
                color_array[..., 3] = alpha

        return color_array.astype(np.float32)

    def get_plot_options(
        self,
        style: str,
        plot_type: str,
        values: np.ndarray | float,
        **kwargs,
    ):
        plot_options = {}

        # Geometry/style options such as size, width, edge_width
        for name in self.necessary_opts[plot_type]:
            plot_options[name] = kwargs.get(
                name,
                self.opts[style].get(name),
            )

        # Only pass actual colour-related arguments onward
        color_kwargs = {
            name: kwargs[name]
            for name in (
                "colors",
                "cmap_name",
                "alpha",
                "alpha_scale",
            )
            if name in kwargs
        }

        colors = self.get_color_array(
            style,
            values,
            **color_kwargs,
        )

        plot_options[self.color_names[plot_type]] = colors

        return plot_options


# def colormap(
#     values: np.ndarray,
#     base_color: list | tuple | str = "viridis",
#     alpha_scale: float = 1.0,
#     offset=0.6,
# ):
#     """
#     values: array-like, assumed normalized to [0,1]
#     returns: (N,4) RGBA array
#     """
#     # print(f"cmap which: {which}")
#     val_min, val_max = np.percentile(values, [5, 95])
#     v = (values - val_min) / (val_max - val_min + 1e-8)
#     v = np.clip(v, 0.0, 1.0)

#     # optional offset (keeps low values visible)
#     if offset > 0:
#         v = offset + (1.0 - offset) * v

#     if isinstance(base_color, str):
#         # map to RGBA using cmap
#         cmap = color.get_colormap(base_color)
#         rgba = cmap.map(v).astype(np.float32)

#         # control alpha separately (very useful for overlap)
#         rgba[:, 3] *= alpha_scale

#         return rgba
#     else:
#         rgba = np.tile(base_color, (len(v), 1))
#         rgba[:, 3] = v * alpha_scale
#         return rgba.astype(np.float32)
