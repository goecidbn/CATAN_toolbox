"""Session × session correlation of the aligned backgrounds (alignment editor
inset, alternative to the mean-shift inset)."""

import numpy as np
from vispy.color import get_colormap

from PySide6.QtCore import QPoint, QRect, Qt, Signal
from PySide6.QtGui import QColor, QImage, QPainter, QPen
from PySide6.QtWidgets import QToolTip, QWidget

from catan.core.spatial_coverage import session_valid_mask


def _binned(image, factor):
    height, width = (np.asarray(image.shape[:2]) // factor) * factor
    image = np.asarray(image[:height, :width], dtype=np.float64)
    return image.reshape(height // factor, factor, width // factor, factor).mean(
        axis=(1, 3)
    )


def background_correlation_matrix(images, masks, *, factor=2, min_overlap=0.25):
    """Pearson correlation of every image pair over their jointly valid pixels
    (images binned by ``factor``; a binned pixel is valid if all its pixels
    are). Entries are NaN for missing images or too little overlap.

    Returns (correlation, overlap fraction), both (N, N).
    """
    n = len(images)
    available = [
        index
        for index, (image, mask) in enumerate(zip(images, masks))
        if image is not None and mask is not None
    ]

    correlation = np.full((n, n), np.nan)
    overlap = np.zeros((n, n))
    if not available:
        return correlation, overlap

    values, valid = [], []
    for index in available:
        mask = _binned(np.asarray(masks[index], dtype=np.float64), factor) >= 1.0
        image = _binned(np.nan_to_num(np.asarray(images[index])), factor)
        mask &= np.isfinite(image)
        valid.append(mask.ravel())
        values.append(np.where(mask, image, 0.0).ravel())

    M = np.asarray(valid, dtype=np.float64)
    X = np.asarray(values)
    X2 = X * X

    # Sums over the joint valid pixels of each pair, as matrix products.
    count = M @ M.T
    sum_x = X @ M.T  # [i, j]: sum of image i over joint pixels
    sum_xx = X2 @ M.T
    sum_xy = X @ X.T

    with np.errstate(invalid="ignore", divide="ignore"):
        mean_i = sum_x / count
        mean_j = sum_x.T / count
        covariance = sum_xy / count - mean_i * mean_j
        variance_i = sum_xx / count - mean_i**2
        variance_j = sum_xx.T / count - mean_j**2
        values = covariance / np.sqrt(variance_i * variance_j)

    fraction = count / M.shape[1]
    values[fraction < min_overlap] = np.nan

    index = np.asarray(available)
    correlation[np.ix_(index, index)] = values
    overlap[np.ix_(index, index)] = fraction
    np.fill_diagonal(correlation, np.nan)
    return correlation, overlap


def session_background_inputs(sessions):
    """(images, masks) of aligned sessions' backgrounds; None if unavailable."""
    images, masks = [], []

    for session in sessions:
        try:
            mask = session_valid_mask(session)
        except ValueError:
            mask = None

        image = getattr(session, "background", None)
        if mask is None or image is None or tuple(image.shape) != tuple(mask.shape):
            images.append(None)
            masks.append(None)
        else:
            images.append(np.asarray(image, dtype=np.float32))
            masks.append(np.asarray(mask, dtype=bool))

    return images, masks


class AlignmentCorrelationInset(QWidget):
    """Clickable raster; chosen(row, column) on click."""

    chosen = Signal(int, int)

    MARGIN_LEFT = 30
    MARGIN_TOP = 22
    MARGIN_BOTTOM = 30

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setFixedSize(260, 220)
        self.setMouseTracking(True)
        self.setAttribute(Qt.WidgetAttribute.WA_NoMousePropagation, True)

        self.correlation = None
        self.overlap = None
        self.labels = []
        self.outdated = set()
        self.current = None
        self.message = "Calculating session correlations…"
        self._image = None
        self._range = (0.0, 1.0)
        self._colormap = get_colormap("viridis")

    # ------------------------------------------------------------------

    def set_message(self, message):
        self.message = message
        self.update()

    def set_matrix(self, correlation, overlap, labels, *, outdated=(), current=None):
        self.correlation = np.asarray(correlation, dtype=float)
        self.overlap = np.asarray(overlap, dtype=float)
        self.labels = list(labels)
        self.outdated = set(outdated)
        self.current = current
        self.message = None

        finite = self.correlation[np.isfinite(self.correlation)]
        if finite.size:
            low, high = np.percentile(finite, [1, 99])
            if high - low < 1e-6:
                low, high = low - 0.05, high + 0.05
            self._range = (float(low), float(high))

        self._image = self._render()
        self.update()

    def set_current(self, current):
        if current != self.current:
            self.current = current
            self.update()

    def _render(self):
        n = len(self.labels)
        low, high = self._range
        normalized = np.clip((self.correlation - low) / (high - low), 0.0, 1.0)

        rgba = np.zeros((n, n, 4), dtype=np.uint8)
        finite = np.isfinite(self.correlation)
        if finite.any():
            colors = self._colormap.map(normalized[finite].reshape(-1, 1))
            rgba[finite] = (np.asarray(colors) * 255).astype(np.uint8)
        rgba[~finite] = (52, 57, 65, 255)

        image = QImage(
            rgba.tobytes(), n, n, 4 * n, QImage.Format.Format_RGBA8888
        )
        return image.copy()

    # ------------------------------------------------------------------

    def _plot_rect(self):
        size = min(
            self.width() - self.MARGIN_LEFT - 8,
            self.height() - self.MARGIN_TOP - self.MARGIN_BOTTOM,
        )
        return QRect(self.MARGIN_LEFT, self.MARGIN_TOP, max(1, size), max(1, size))

    def _cell_at(self, position):
        n = len(self.labels)
        rect = self._plot_rect()
        if not n or not rect.contains(position):
            return None

        column = int((position.x() - rect.left()) * n / rect.width())
        row = int((position.y() - rect.top()) * n / rect.height())
        return min(row, n - 1), min(column, n - 1)

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.fillRect(self.rect(), QColor("#20252b"))
        painter.setPen(QColor("#e8eaed"))

        font = painter.font()
        font.setPointSizeF(8)
        painter.setFont(font)
        painter.drawText(
            QRect(4, 2, self.width() - 8, 16),
            Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter,
            "Background correlation · aligned sessions",
        )

        rect = self._plot_rect()

        if self._image is None or self.message:
            painter.setPen(QColor("#aab2bd"))
            painter.drawText(
                rect.adjusted(-20, 0, 20, 0),
                Qt.AlignmentFlag.AlignCenter | Qt.TextFlag.TextWordWrap,
                self.message or "",
            )
            return

        painter.drawImage(rect, self._image)
        n = len(self.labels)
        cell = rect.width() / n

        # Session ticks.
        step = max(1, int(np.ceil(n / 6 / 5.0)) * 5) if n > 10 else 1
        painter.setPen(QColor("#aab2bd"))
        font.setPointSizeF(7)
        painter.setFont(font)
        for index in range(0, n, step):
            offset = int(rect.left() + (index + 0.5) * cell)
            painter.drawText(
                QRect(offset - 12, rect.bottom() + 2, 24, 12),
                Qt.AlignmentFlag.AlignCenter,
                str(index),
            )
            offset = int(rect.top() + (index + 0.5) * cell)
            painter.drawText(
                QRect(0, offset - 6, self.MARGIN_LEFT - 3, 12),
                Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter,
                str(index),
            )

        # Outdated alignments: marked on the axes.
        painter.setPen(QPen(QColor("#ffb36b"), 2))
        for index in self.outdated:
            offset = int(rect.left() + (index + 0.5) * cell)
            painter.drawLine(offset, rect.top() - 4, offset, rect.top() - 1)

        # Current session: row and column outlined.
        if self.current is not None and 0 <= self.current < n:
            painter.setPen(QPen(QColor("#ffffff"), 1))
            painter.setBrush(Qt.BrushStyle.NoBrush)
            top = rect.top() + self.current * cell
            left = rect.left() + self.current * cell
            painter.drawRect(int(rect.left()), int(top), rect.width(), max(1, int(cell)))
            painter.drawRect(int(left), int(rect.top()), max(1, int(cell)), rect.height())

        # Colour scale.
        low, high = self._range
        bar = QRect(rect.right() - 90, self.height() - 12, 60, 8)
        gradient = (
            np.asarray(self._colormap.map(np.linspace(0, 1, 60).reshape(-1, 1))) * 255
        ).astype(np.uint8)
        strip = np.ascontiguousarray(gradient[None, :, :])
        painter.drawImage(
            bar,
            QImage(strip.tobytes(), 60, 1, 240, QImage.Format.Format_RGBA8888).copy(),
        )
        painter.setPen(QColor("#aab2bd"))
        painter.drawText(
            QRect(bar.left() - 34, bar.top() - 3, 32, 12),
            Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter,
            f"{low:.2f}",
        )
        painter.drawText(
            QRect(bar.right() + 3, bar.top() - 3, 32, 12),
            Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter,
            f"{high:.2f}",
        )

    def mouseMoveEvent(self, event):
        cell = self._cell_at(event.position().toPoint())
        if cell is None or self.correlation is None:
            QToolTip.hideText()
            return

        row, column = cell
        value = self.correlation[row, column]
        text = f"{self.labels[row]}  ↔  {self.labels[column]}\n"
        if row == column:
            text += "Same session."
        elif np.isfinite(value):
            text += (
                f"Correlation {value:.3f} · overlap "
                f"{100 * self.overlap[row, column]:.0f}%"
            )
        else:
            text += "Not available (not aligned or too little overlap)."
        outdated = [i for i in (row, column) if i in self.outdated]
        if outdated:
            text += "\nOutdated alignment: " + ", ".join(f"S{i}" for i in outdated)
        if row != column:
            text += "\nClick: align the later session against the earlier one."

        QToolTip.showText(event.globalPosition().toPoint() + QPoint(12, 12), text, self)

    def mousePressEvent(self, event):
        cell = self._cell_at(event.position().toPoint())
        if cell is not None and event.button() == Qt.MouseButton.LeftButton:
            row, column = cell
            if row != column:
                self.chosen.emit(row, column)
        event.accept()

    def leaveEvent(self, event):
        QToolTip.hideText()
        super().leaveEvent(event)
