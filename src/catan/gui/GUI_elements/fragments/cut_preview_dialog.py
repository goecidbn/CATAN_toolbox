"""Cut at the neck with a preview: the two parts of a footprint for the
chosen lobe parameters (catan.core.footprint_lobes), which part stays in the
neuron and which becomes a new one. Lower "Min. lobe peak" / "Min. lobe
size" or a fixed threshold force a cut the defaults do not find."""

import numpy as np
from PySide6.QtCore import Qt
from PySide6.QtGui import QImage, QPixmap
from PySide6.QtWidgets import (
    QCheckBox,
    QDialog,
    QDialogButtonBox,
    QDoubleSpinBox,
    QFormLayout,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from catan.core.footprint_lobes import DEFAULTS
from catan.gui.utils.session_labels import session_label

KEPT_RGB = np.array([79, 179, 191], dtype=np.float32)  # stays in the neuron
NEW_RGB = np.array([255, 179, 107], dtype=np.float32)  # new neuron
PREVIEW_PX = 300
MARGIN_PX = 3


def render_parts(parts, kept, dims, size=PREVIEW_PX):
    """RGB image (uint8) of both parts on black, brightness by weight,
    cropped to the footprint; None without pixels."""
    columns = [part.tocsc() for part in parts]
    rows = np.concatenate([column.indices for column in columns])
    if not len(rows):
        return None
    y, x = np.divmod(rows, int(dims[1]))
    y0, x0 = max(int(y.min()) - MARGIN_PX, 0), max(int(x.min()) - MARGIN_PX, 0)
    y1 = min(int(y.max()) + MARGIN_PX + 1, int(dims[0]))
    x1 = min(int(x.max()) + MARGIN_PX + 1, int(dims[1]))
    peak = max(float(column.max()) for column in columns) or 1.0

    image = np.zeros((y1 - y0, x1 - x0, 3), dtype=np.float32)
    for index, column in enumerate(columns):
        py, px = np.divmod(column.indices, int(dims[1]))
        weight = np.clip(column.data / peak, 0, 1)[:, None] ** 0.6
        colour = KEPT_RGB if index == kept else NEW_RGB
        image[py - y0, px - x0] = weight * colour

    scale = max(1, size // max(image.shape[:2]))
    image = np.repeat(np.repeat(image, scale, axis=0), scale, axis=1)
    return np.ascontiguousarray(image.astype(np.uint8))


class CutPreviewDialog(QDialog):
    """Returns the chosen lobe parameters (``parameters()``) on accept."""

    def __init__(self, data, component, parent=None):
        super().__init__(parent)
        self.data = data
        self.component = component
        self.setWindowTitle(
            f"Cut at the neck — neuron {component.neuron_id} · "
            f"{session_label(data, component.session_id)}"
        )

        layout = QVBoxLayout(self)

        self.image = QLabel(self)
        self.image.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.image.setMinimumSize(PREVIEW_PX, PREVIEW_PX)
        self.image.setStyleSheet("background-color: black;")
        layout.addWidget(self.image)

        legend = QLabel(
            f"<span style='color:rgb{tuple(int(v) for v in KEPT_RGB)}'>■</span> "
            f"stays in neuron {component.neuron_id} &nbsp; "
            f"<span style='color:rgb{tuple(int(v) for v in NEW_RGB)}'>■</span> "
            "new neuron",
            self,
        )
        layout.addWidget(legend)

        form = QFormLayout()
        self.peak = self._spin(0.05, 1.0, 0.05, DEFAULTS["peak"])
        self.peak.setToolTip(
            "Each lobe needs its own peak at least this high (relative to "
            "the footprint's peak). Lower it for a dim second cell."
        )
        form.addRow("Min. lobe peak", self.peak)
        self.size = self._spin(0.01, 1.0, 0.01, DEFAULTS["size"])
        self.size.setToolTip(
            "Each lobe needs at least this share of the footprint's area "
            "(above 10 % of its peak)."
        )
        form.addRow("Min. lobe size", self.size)

        threshold_row = QWidget(self)
        row = QHBoxLayout(threshold_row)
        row.setContentsMargins(0, 0, 0, 0)
        self.fixed = QCheckBox("Fixed", threshold_row)
        self.fixed.setToolTip(
            "Off: the threshold where the most lobes separate is searched. "
            "On: the lobes are the parts above this threshold (relative to "
            "the peak)."
        )
        self.threshold = self._spin(0.05, 0.95, 0.05, 0.3)
        self.threshold.setEnabled(False)
        self.fixed.toggled.connect(self.threshold.setEnabled)
        row.addWidget(self.fixed)
        row.addWidget(self.threshold, 1)
        form.addRow("Threshold", threshold_row)
        layout.addLayout(form)

        self.status = QLabel(self)
        self.status.setWordWrap(True)
        layout.addWidget(self.status)

        buttons = QDialogButtonBox(self)
        self.cut_button = buttons.addButton("Cut", QDialogButtonBox.ButtonRole.AcceptRole)
        reset = QPushButton("Defaults", self)
        buttons.addButton(reset, QDialogButtonBox.ButtonRole.ResetRole)
        buttons.addButton(QDialogButtonBox.StandardButton.Cancel)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        reset.clicked.connect(self._defaults)
        layout.addWidget(buttons)

        for widget in (self.peak, self.size, self.threshold):
            widget.valueChanged.connect(self.update_preview)
        self.fixed.toggled.connect(self.update_preview)

        self.update_preview()

    def _spin(self, minimum, maximum, step, value):
        spin = QDoubleSpinBox(self)
        spin.setRange(minimum, maximum)
        spin.setSingleStep(step)
        spin.setDecimals(2)
        spin.setValue(value)
        return spin

    def _defaults(self):
        self.fixed.setChecked(False)
        self.peak.setValue(DEFAULTS["peak"])
        self.size.setValue(DEFAULTS["size"])

    def parameters(self):
        """Lobe parameters for Data.cut_footprint (only non-defaults)."""
        chosen = {
            "peak": round(self.peak.value(), 2),
            "size": round(self.size.value(), 2),
            "threshold": round(self.threshold.value(), 2) if self.fixed.isChecked() else None,
        }
        return {key: value for key, value in chosen.items() if value != DEFAULTS[key]}

    def update_preview(self, *_):
        try:
            parts, kept = self.data.cut_preview(self.component, **self.parameters())
        except ValueError as exc:
            self.parts = None
            self.status.setText(str(exc))
            self.cut_button.setEnabled(False)
            self._show_footprint()
            return

        self.parts = parts
        dims = tuple(self.data.sessions[self.component.session_id].dims)
        self._set_image(render_parts(parts, kept, dims))
        sizes = [int(part.nnz) for part in parts]
        self.status.setText(
            f"Stays in neuron {self.component.neuron_id}: {sizes[kept]} px · "
            f"new neuron: {sizes[1 - kept]} px"
        )
        self.cut_button.setEnabled(True)

    def _show_footprint(self):
        """Without a cut: the whole footprint (grey)."""
        session = self.data.sessions[self.component.session_id]
        footprint_id = int(
            self.data.assignments.ids[self.component.neuron_id, self.component.session_id]
        )
        column = session.footprints[:, footprint_id]
        image = render_parts([column], 0, tuple(session.dims))
        if image is not None:
            image = np.ascontiguousarray(image.mean(axis=2, keepdims=True).repeat(3, axis=2).astype(np.uint8))
        self._set_image(image)

    def _set_image(self, image):
        if image is None:
            self.image.clear()
            return
        height, width, _ = image.shape
        self._buffer = image  # QImage does not copy
        qimage = QImage(image.data, width, height, 3 * width, QImage.Format.Format_RGB888)
        self.image.setPixmap(QPixmap.fromImage(qimage))
