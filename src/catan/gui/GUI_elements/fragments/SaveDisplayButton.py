from pathlib import Path

import numpy as np

from PySide6.QtCore import QEvent, QByteArray, QBuffer, QIODevice
from PySide6.QtGui import QImage, QImageWriter
from PySide6.QtWidgets import (
    QDialog,
    QDialogButtonBox,
    QDoubleSpinBox,
    QFormLayout,
    QLabel,
    QSpinBox,
    QStyle,
    QToolButton,
    QCheckBox,
)

from catan.gui.background_tasks.file_writes import save_bytes_task
from .path_selector import choose_path

from .ResetViewButton import OVERLAY_BUTTON_STYLE


class SaveDisplayButton(QToolButton):

    def __init__(
        self,
        canvas,
        *,
        get_directory,
        settings,
        filename="catan_plot",
    ):
        super().__init__(canvas.native)

        self.canvas = canvas
        self.get_directory = get_directory
        self.settings = settings
        self.filename = filename

        self.setIcon(
            self.style().standardIcon(QStyle.StandardPixmap.SP_DialogSaveButton)
        )
        self.setToolTip("Export plot image")
        self.setAutoRaise(True)
        self.setStyleSheet(OVERLAY_BUTTON_STYLE)
        self.setFixedSize(28, 28)

        self.clicked.connect(self.save_image)
        canvas.native.installEventFilter(self)

        self._reposition()
        self.show()

    def eventFilter(self, watched, event):
        if watched is self.canvas.native and event.type() in (
            QEvent.Type.Resize,
            QEvent.Type.Show,
        ):
            self._reposition()

        return super().eventFilter(watched, event)

    def _reposition(self):
        self.move(
            6,
            max(6, self.canvas.native.height() - self.height() - 6),
        )
        self.raise_()

    def _choose_resolution(self):
        dialog = QDialog(self.canvas.native)
        dialog.setWindowTitle("Export plot")
        form = QFormLayout(dialog)

        width = QDoubleSpinBox(dialog)
        width.setRange(10, 1000)
        width.setDecimals(1)
        width.setSuffix(" mm")
        width.setValue(self.settings.value("export/width_mm", 180.0, type=float))

        dpi = QSpinBox(dialog)
        dpi.setRange(72, 1200)
        dpi.setValue(self.settings.value("export/dpi", 300, type=int))

        dimensions = QLabel(dialog)

        canvas_width, canvas_height = self.canvas.size
        aspect = canvas_height / max(1, canvas_width)

        def update_dimensions(*_):
            pixels_x = max(1, round(width.value() / 25.4 * dpi.value()))
            pixels_y = max(1, round(pixels_x * aspect))
            dimensions.setText(f"{pixels_x} × {pixels_y} pixels")

        width.valueChanged.connect(update_dimensions)
        dpi.valueChanged.connect(update_dimensions)
        update_dimensions()

        transparent = QCheckBox("Transparent background (PNG)", dialog)
        transparent.setChecked(
            self.settings.value("export/transparent", False, type=bool)
        )

        form.addRow("Print width:", width)
        form.addRow("Resolution (DPI):", dpi)
        form.addRow(transparent)
        form.addRow("Image size:", dimensions)

        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel,
            parent=dialog,
        )
        buttons.accepted.connect(dialog.accept)
        buttons.rejected.connect(dialog.reject)
        form.addRow(buttons)

        result = None
        if dialog.exec() == QDialog.DialogCode.Accepted:
            result = (width.value(), dpi.value(), transparent.isChecked())

        dialog.deleteLater()
        return result

    def save_image(self):
        options = self._choose_resolution()
        if options is None:
            return

        width_mm, dpi, transparent = options
        state = self.canvas.state

        last_directory = self.settings.value("export/last_directory", "", type=str)
        directory = Path(
            last_directory or self.get_directory() or Path.home()
        ).expanduser()

        filters = [("PNG image", ("*.png",))]
        if not transparent:
            filters.append(("JPEG image", ("*.jpg", "*.jpeg")))

        filename = choose_path(
            self,
            init_path=str(directory),
            display_text="Save plot image",
            only_existing=False,
            default_suffix="png",
            file_filters=filters,
            state=state,
            default_filename=self.filename,
        )
        if filename is None:
            return

        path = Path(filename)

        try:
            suffix = path.suffix.lower()

            if suffix not in (".png", ".jpg", ".jpeg"):
                raise ValueError("Use a .png, .jpg, or .jpeg filename.")

            if transparent and suffix != ".png":
                raise ValueError("Transparent export requires a .png filename.")

            canvas_width, canvas_height = self.canvas.size
            if canvas_width <= 0 or canvas_height <= 0:
                raise ValueError("The plot has no drawable size.")

            pixels_x = max(1, round(width_mm / 25.4 * dpi))
            pixels_y = max(1, round(pixels_x * canvas_height / canvas_width))

            if pixels_x * pixels_y > 40_000_000:
                raise ValueError(
                    "The requested image exceeds 40 megapixels. "
                    "Reduce the print width or DPI."
                )

            # Rendering stays on the GUI thread, using the existing
            # VisPy/OpenGL context.
            pixels = np.ascontiguousarray(
                self.canvas.render(
                    size=(pixels_x, pixels_y),
                    bgcolor=(0, 0, 0, 0) if transparent else None,
                    alpha=transparent,
                ),
                dtype=np.uint8,
            )

            height, width, _ = pixels.shape
            image_format = (
                QImage.Format.Format_RGBA8888
                if transparent
                else QImage.Format.Format_RGB888
            )

            image = QImage(
                pixels.data,
                width,
                height,
                pixels.strides[0],
                image_format,
            ).copy()

            dots_per_meter = round(dpi / 0.0254)
            image.setDotsPerMeterX(dots_per_meter)
            image.setDotsPerMeterY(dots_per_meter)

            # Encode in memory: QImageWriter never opens the destination.
            encoded = QByteArray()
            buffer = QBuffer(encoded)

            if not buffer.open(QIODevice.OpenModeFlag.WriteOnly):
                raise OSError("Could not allocate the image export buffer.")

            try:
                image_type = b"png" if suffix == ".png" else b"jpg"
                writer = QImageWriter(buffer, image_type)

                if image_type == b"jpg":
                    writer.setQuality(95)

                if not writer.write(image):
                    raise OSError(writer.errorString())

                payload = bytes(encoded)
            finally:
                buffer.close()

        except Exception as exc:
            state.issue(
                "warning",
                "Plot export failed",
                str(exc),
                parent=self.window(),
            )
            return

        # Capture the settings object rather than the button in the callback.
        settings = self.settings

        def saved(_result):
            settings.setValue("export/last_directory", str(path.parent))
            settings.setValue("export/width_mm", width_mm)
            settings.setValue("export/dpi", dpi)
            settings.setValue("export/transparent", transparent)

        state.tasks.start(
            "saving",
            f"Saving plot image: {path.name}",
            save_bytes_task,
            state,
            str(path),
            payload,
            on_result=saved,
        )
