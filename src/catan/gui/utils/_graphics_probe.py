"""Minimal VisPy rendering probe used by CATAN's startup logic."""

from __future__ import annotations

import traceback


def main() -> int:
    """Render a small VisPy line and return zero if it succeeds."""

    import numpy as np
    from vispy import app

    # Select the same backend CATAN uses.
    app.use_app("pyside6")

    from vispy import scene

    canvas = None

    try:
        canvas = scene.SceneCanvas(
            show=False,
            size=(64, 64),
            bgcolor="black",
        )

        view = canvas.central_widget.add_view(camera="panzoom")

        scene.visuals.Line(
            pos=np.array(
                [
                    [0.0, 0.0],
                    [1.0, 1.0],
                ],
                dtype=np.float32,
            ),
            color="white",
            method="gl",
            parent=view.scene,
        )

        view.camera.set_range()

        # This forces creation of the OpenGL context, shader compilation,
        # drawing, and framebuffer readback.
        image = canvas.render(size=(64, 64))

        if image.shape[:2] != (64, 64):
            raise RuntimeError(f"Unexpected probe image shape: {image.shape}")

        return 0

    finally:
        if canvas is not None:
            canvas.close()


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception:
        traceback.print_exc()
        raise SystemExit(1)
