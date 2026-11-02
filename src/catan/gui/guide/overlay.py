"""The guide's overlay (dims the window except a cut-out), the frame
around the step's target and its bubble."""

from PySide6.QtCore import QRect, Qt, QTimer, Signal
from PySide6.QtGui import QColor, QPainter, QPainterPath, QPen, QRegion
from PySide6.QtWidgets import (
    QFrame,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QScrollArea,
    QVBoxLayout,
    QWidget,
)

DIM = QColor(0, 0, 0, 150)
# Inside an open area, around the target: lighter (still clickable).
SOFT_DIM = QColor(0, 0, 0, 95)
HIGHLIGHT = QColor(255, 196, 0)
OUTLINE = QColor(20, 20, 20, 220)
MARGIN = 6  # around the cut-out
RING = 4  # frame width
# The bubble grows with its text up to this fraction of the window height.
MAX_HEIGHT = 0.7
PULSE_MS = 60


class GuideOverlay(QWidget):
    """Child of the main window covering it. Its mask leaves out the
    cut-out, so clicks there reach the real widget; elsewhere the overlay
    takes them."""

    def __init__(self, window):
        super().__init__(window)
        self.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        self.hole = QRect()
        self.holes = ()
        self.hide()

    def set_hole(self, rect, extra=()):
        """Cut-out ``rect`` plus further open ``extra`` rects."""
        window_rect = self.parentWidget().rect()
        holes = tuple(
            r.adjusted(-MARGIN, -MARGIN, MARGIN, MARGIN).intersected(window_rect)
            for r in (rect, *extra)
            if r is not None
        )
        holes = tuple(r for r in holes if not r.isEmpty())

        if holes == self.holes and self.geometry() == window_rect:
            return

        self.holes = holes
        self.hole = holes[0] if holes else QRect()
        self.setGeometry(window_rect)
        region = QRegion(self.rect())
        for hole in holes:
            region = region.subtracted(QRegion(hole))
        self.setMask(region)
        self.update()

    def paintEvent(self, event):
        QPainter(self).fillRect(self.rect(), DIM)

    def mousePressEvent(self, event):
        event.accept()  # outside the cut-out: blocked


class GuideFrame(QWidget):
    """Covers the open area (cut-out) without taking clicks: dims it
    lightly except the step's target parts and draws a pulsing frame
    around each of them."""

    def __init__(self, window):
        super().__init__(window)
        self.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents, True)
        self.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        self._rects = []
        self._dim = False
        self._area = QRect()
        self._phase = 0
        self._pulse = QTimer(self)
        self._pulse.setInterval(PULSE_MS)
        self._pulse.timeout.connect(self._step_pulse)
        self.hide()

    def frame(self, area, rects):
        """Frames around ``rects`` inside ``area`` (window coordinates);
        the rest of ``area`` is dimmed lightly when the targets are only a
        part of it. No rects: hidden."""
        rects = [r for r in rects or () if r is not None and not r.isEmpty()]
        if not rects:
            self._rects = []
            self._pulse.stop()
            self.hide()
            return
        grow = MARGIN + RING
        outer = [r.adjusted(-grow, -grow, grow, grow) for r in rects]
        bounds = outer[0]
        for r in outer[1:]:
            bounds = bounds.united(r)
        if area is not None and not area.isEmpty():
            bounds = bounds.united(area.adjusted(-MARGIN, -MARGIN, MARGIN, MARGIN))
        self.setGeometry(bounds)
        offset = bounds.topLeft()
        self._rects = [
            r.adjusted(-MARGIN, -MARGIN, MARGIN, MARGIN).translated(-offset) for r in rects
        ]
        self._dim = area is not None and not any(r.contains(area) for r in rects)
        # Only the open area is dimmed lightly (targets may lie outside it).
        self._area = (
            area.adjusted(-MARGIN, -MARGIN, MARGIN, MARGIN).translated(-offset)
            if self._dim
            else QRect()
        )
        self.update()
        self.show()
        self.raise_()
        if not self._pulse.isActive():
            self._pulse.start()

    def hideEvent(self, event):
        self._pulse.stop()
        super().hideEvent(event)

    def _step_pulse(self):
        self._phase = (self._phase + 1) % 30
        self.update()

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)

        if self._dim:
            path = QPainterPath()
            path.addRect(self._area)
            for r in self._rects:
                path.addRoundedRect(r, 4, 4)
            path.setFillRule(Qt.FillRule.OddEvenFill)
            painter.fillPath(path, SOFT_DIM)

        # 0 … 1 … 0 over 30 steps
        wave = abs(15 - self._phase) / 15
        glow = QColor(HIGHLIGHT)
        glow.setAlpha(int(60 + 120 * wave))
        for r in self._rects:
            painter.setPen(QPen(OUTLINE, RING + 2))
            painter.drawRoundedRect(r, 5, 5)
            painter.setPen(QPen(HIGHLIGHT, RING))
            painter.drawRoundedRect(r, 5, 5)
            painter.setPen(QPen(glow, 3))
            painter.drawRoundedRect(r.adjusted(-RING, -RING, RING, RING), 7, 7)


class GuideBubble(QFrame):
    backRequested = Signal()
    nextRequested = Signal()
    doItRequested = Signal()
    skipChapterRequested = Signal()
    closeRequested = Signal()
    moreRequested = Signal()

    WIDTH = 380

    def __init__(self, window):
        super().__init__(window)
        self.setObjectName("GuideBubble")
        self.setStyleSheet(
            "#GuideBubble { background: #2b2f36; border: 2px solid #50aaff;"
            " border-radius: 8px; }"
            " QLabel { background: transparent; }"
        )
        self.setFixedWidth(self.WIDTH)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(14, 12, 14, 12)
        layout.setSpacing(8)

        self.counter = QLabel()
        self.counter.setStyleSheet("color: #9aa4b2; font-size: 9pt;")
        self.counter.setToolTip("Drag the box to move it out of the way")
        self.counter.setCursor(Qt.CursorShape.OpenHandCursor)
        layout.addWidget(self.counter)

        self.title = QLabel()
        self.title.setStyleSheet("font-weight: bold; font-size: 12pt;")
        self.title.setWordWrap(True)
        layout.addWidget(self.title)

        self.text = QLabel()
        self.text.setWordWrap(True)
        self.text.setTextFormat(Qt.TextFormat.RichText)
        self.text.setOpenExternalLinks(False)
        self.text.setAlignment(Qt.AlignmentFlag.AlignTop | Qt.AlignmentFlag.AlignLeft)
        # Scrolls only when the window is too small for the whole text.
        self.text_area = QScrollArea()
        self.text_area.setWidget(self.text)
        self.text_area.setWidgetResizable(True)
        self.text_area.setFrameShape(QFrame.Shape.NoFrame)
        self.text_area.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.text_area.setStyleSheet("QScrollArea, QScrollArea > QWidget > QWidget { background: transparent; }")
        layout.addWidget(self.text_area)

        self.status = QLabel()
        self.status.setWordWrap(True)
        self.status.setStyleSheet("color: #f0c674;")
        layout.addWidget(self.status)

        self.do_it = QPushButton("Do it for me")
        self.do_it.clicked.connect(self.doItRequested.emit)
        layout.addWidget(self.do_it)

        # Quick tour: the stop's part of the detailed guide.
        self.more_button = QPushButton("Learn more →")
        self.more_button.setToolTip(
            "Open the detailed guide for this; it returns here afterwards"
        )
        self.more_button.clicked.connect(self.moreRequested.emit)
        self.more_button.hide()
        layout.addWidget(self.more_button)

        buttons = QHBoxLayout()
        buttons.setSpacing(6)
        self.close_button = QPushButton("Close")
        self.close_button.setToolTip("Close the guide (reopen it with the ? button)")
        self.skip_button = QPushButton("Skip chapter")
        self.back_button = QPushButton("Back")
        self.next_button = QPushButton("Next")
        self.next_button.setDefault(True)

        self.close_button.clicked.connect(self.closeRequested.emit)
        self.skip_button.clicked.connect(self.skipChapterRequested.emit)
        self.back_button.clicked.connect(self.backRequested.emit)
        self.next_button.clicked.connect(self.nextRequested.emit)

        buttons.addWidget(self.close_button)
        buttons.addWidget(self.skip_button)
        buttons.addStretch()
        buttons.addWidget(self.back_button)
        buttons.addWidget(self.next_button)
        layout.addLayout(buttons)

        self.hide()

    # Dragging (anywhere but the buttons and the text area's scroll bar)
    # moves the bubble out of the way for the rest of the step.

    user_placed = False
    _drag_offset = None

    def release(self):
        """Back to automatic placement (a new step)."""
        self.user_placed = False

    def mousePressEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton:
            self._drag_offset = event.globalPosition().toPoint() - self.pos()
            self.setCursor(Qt.CursorShape.ClosedHandCursor)
            event.accept()
            return
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event):
        if self._drag_offset is not None:
            position = event.globalPosition().toPoint() - self._drag_offset
            parent = self.parentWidget().rect()
            position.setX(max(0, min(position.x(), parent.width() - self.width())))
            position.setY(max(0, min(position.y(), parent.height() - self.height())))
            self.move(position)
            self.user_placed = True
            event.accept()
            return
        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event):
        if self._drag_offset is not None:
            self._drag_offset = None
            self.unsetCursor()
            event.accept()
            return
        super().mouseReleaseEvent(event)

    def _text_width(self):
        margins = self.layout().contentsMargins()
        # 2 px border on each side.
        return self.WIDTH - margins.left() - margins.right() - 4

    def fit_height(self, available):
        """Height for the whole text (word-wrapped rich text: height for
        the fixed width) up to ``available``; only beyond, the text
        scrolls."""
        full = self._full_text_height(self._text_width())
        self.text_area.setFixedHeight(max(full, 1))
        self.layout().activate()
        height = self.layout().totalHeightForWidth(self.WIDTH)
        if height > available:
            bar = self.text_area.verticalScrollBar().sizeHint().width()
            # Narrower with the scroll bar: the text gets taller.
            full = self._full_text_height(self._text_width() - bar)
            self.text_area.setFixedHeight(max(60, min(full, full - (height - available))))
            self.layout().activate()
            height = self.layout().totalHeightForWidth(self.WIDTH)
        return height

    def _full_text_height(self, width):
        # QLabel's heightForWidth can fall short for rich text (lists):
        # the larger of it and the laid-out document.
        from PySide6.QtGui import QTextDocument

        document = QTextDocument()
        document.setDefaultFont(self.text.font())
        document.setDocumentMargin(0)
        document.setHtml(self.text.text())
        document.setTextWidth(width)
        return max(self.text.heightForWidth(width), int(document.size().height()) + 2)

    def settle(self):
        """After layout: grow by what still scrolls while there is room
        (fonts and styles can differ from the estimate)."""
        bar = self.text_area.verticalScrollBar()
        missing = bar.maximum()
        limit = getattr(self, "_height_limit", None)
        if missing <= 0 or limit is None or self.height() >= limit:
            return
        grow = min(missing + 2, limit - self.height())
        self.text_area.setFixedHeight(self.text_area.height() + grow)
        geometry = self.geometry()
        geometry.setHeight(geometry.height() + grow)
        bottom = self.parentWidget().rect().bottom() - 8
        if geometry.bottom() > bottom:
            geometry.moveBottom(bottom)
        self.setGeometry(geometry)

    def text_cropped(self):
        """Text not fully visible without scrolling (tests)."""
        return self.text_area.verticalScrollBar().maximum() > 0

    def place(self, target, bounds, avoid=None, keep_clear=()):
        """Next to ``target`` (window coordinates; None: centred), outside
        ``avoid`` (e.g. an open dialog) where there is room, not covering
        ``keep_clear`` (the framed parts of the step) where possible. Once
        the user dragged the bubble, it stays there (only its height
        follows the text) until ``release`` (next step)."""
        width = self.WIDTH
        # Up to a large part of the window, then the text scrolls.
        self._height_limit = max(200, min(bounds.height() - 16, int(self.window().height() * MAX_HEIGHT)))
        height = self.fit_height(self._height_limit)
        gap = 16

        if self.user_placed:
            x = max(bounds.left() + 8, min(self.x(), bounds.right() - width - 8))
            y = max(bounds.top() + 8, min(self.y(), bounds.bottom() - height - 8))
            self.setGeometry(x, y, width, height)
            QTimer.singleShot(0, self, self.settle)
            return

        keep_clear = [r for r in keep_clear if r is not None and not r.isEmpty()]

        def fits(cx, cy):
            if not (
                cx >= bounds.left()
                and cx + width <= bounds.right()
                and cy + min(height, 80) <= bounds.bottom()
                and cy >= bounds.top()
            ):
                return False
            box = QRect(cx, cy, width, min(height, bounds.bottom() - cy))
            return not any(box.intersects(r) for r in keep_clear)

        if avoid is not None and not avoid.isEmpty():
            # Beside the dialog: the side with the most room.
            sides = [
                (avoid.left() - bounds.left(), (avoid.left() - gap - width, avoid.top())),
                (bounds.right() - avoid.right(), (avoid.right() + gap, avoid.top())),
                (avoid.top() - bounds.top(), (avoid.left(), avoid.top() - gap - height)),
                (bounds.bottom() - avoid.bottom(), (avoid.left(), avoid.bottom() + gap)),
            ]
            fits = [
                (room, position)
                for index, (room, position) in enumerate(sides)
                if room >= (width if index < 2 else height) + gap
            ]
            x, y = max(fits or sides)[1]
        elif target is None or target.isEmpty():
            x = bounds.center().x() - width // 2
            y = bounds.center().y() - height // 2
        else:
            candidates = [
                (target.right() + gap, target.top()),  # right
                (target.left() - gap - width, target.top()),  # left
                (target.left(), target.bottom() + gap),  # below
                (target.left(), target.top() - gap - height),  # above
            ]
            # Large targets: inside, in a corner that leaves the framed
            # parts visible (top right first).
            inside = [
                (target.right() - width - gap, target.top() + gap),
                (target.right() - width - gap, target.bottom() - height - gap),
                (target.left() + gap, target.bottom() - height - gap),
                (target.left() + gap, target.top() + gap),
            ]
            x, y = inside[0]
            for cx, cy in candidates + inside:
                if fits(cx, cy):
                    x, y = cx, cy
                    break
            else:
                # Nothing leaves them all clear: beside the target if there
                # is room, else inside it.
                keep_clear = []
                for cx, cy in candidates:
                    if fits(cx, cy):
                        x, y = cx, cy
                        break

        x = max(bounds.left() + 8, min(x, bounds.right() - width - 8))
        y = max(bounds.top() + 8, min(y, bounds.bottom() - height - 8))
        self.setGeometry(x, y, width, height)
        QTimer.singleShot(0, self, self.settle)  # dropped if the bubble is deleted


class GuideOutline(QFrame):
    """Table of contents: the quick tour, then the detailed guide's
    chapters with their steps, an estimated duration and what you will
    learn; start at any chapter."""

    chapterRequested = Signal(str)
    closeRequested = Signal()

    WIDTH = 620

    def __init__(self, window):
        super().__init__(window)
        self.setObjectName("GuideOutline")
        self.setStyleSheet(
            "#GuideOutline { background: #2b2f36; border: 2px solid #50aaff;"
            " border-radius: 8px; }"
            " QLabel { background: transparent; }"
        )
        self.setFixedWidth(self.WIDTH)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(16, 14, 16, 14)
        layout.setSpacing(10)

        title = QLabel("Starter guide")
        title.setStyleSheet("font-weight: bold; font-size: 13pt;")
        layout.addWidget(title)

        # The quick tour (filled by populate).
        self.quick_row = QHBoxLayout()
        self.quick_row.setSpacing(12)
        self.quick_label = QLabel()
        self.quick_label.setWordWrap(True)
        self.quick_label.setTextFormat(Qt.TextFormat.RichText)
        self.quick_button = QPushButton("Start")
        self.quick_button.clicked.connect(
            lambda: self._quick is not None and self.chapterRequested.emit(self._quick)
        )
        self.quick_row.addWidget(self.quick_label, 1)
        self.quick_row.addWidget(self.quick_button, 0, Qt.AlignmentFlag.AlignTop)
        layout.addLayout(self.quick_row)
        self._quick = None

        detailed = QLabel("Detailed guide")
        detailed.setStyleSheet("font-weight: bold; font-size: 11pt;")
        layout.addWidget(detailed)

        intro = QLabel(
            "Works on a small demo dataset and leads through a typical CATAN "
            "session. Start at the beginning, or jump to any chapter: chapters "
            "that need earlier results set them up for you."
        )
        intro.setWordWrap(True)
        layout.addWidget(intro)

        self.grid = QGridLayout()
        self.grid.setHorizontalSpacing(12)
        self.grid.setVerticalSpacing(8)
        self.grid.setColumnStretch(1, 1)
        layout.addLayout(self.grid)

        self.total = QLabel()
        self.total.setStyleSheet("color: #9aa4b2;")
        layout.addWidget(self.total)

        buttons = QHBoxLayout()
        self.close_button = QPushButton("Close")
        self.close_button.setToolTip("Close the guide (reopen it with the ? button)")
        self.start_button = QPushButton("Start with chapter 1")
        self.start_button.setDefault(True)
        self.close_button.clicked.connect(self.closeRequested.emit)
        self.start_button.clicked.connect(self._start_first)
        buttons.addWidget(self.close_button)
        buttons.addStretch()
        buttons.addWidget(self.start_button)
        layout.addLayout(buttons)

        self._first = None
        self.hide()

    def _start_first(self):
        if self._first is not None:
            self.chapterRequested.emit(self._first)

    def populate(self, chapters, completed, quick=None):
        self._quick = None if quick is None else quick.id
        for widget in (self.quick_label, self.quick_button):
            widget.setVisible(quick is not None)
        if quick is not None:
            done = "  ✓" if quick.id in completed else ""
            minutes = f" · ~{quick.minutes} min" if quick.minutes else ""
            self.quick_label.setText(
                f"<b>{quick.title}</b>{done}"
                f"<span style='color: #9aa4b2;'> · {len(quick.steps)} stops{minutes}</span><br>"
                f"<span style='color: #c5ccd6;'>{quick.description or quick.summary}</span>"
            )
            self.quick_button.setText("Again" if quick.id in completed else "Start")

        while self.grid.count():
            item = self.grid.takeAt(0)
            if item.widget() is not None:
                item.widget().deleteLater()

        ready = [c for c in chapters if c.ready]
        self._first = ready[0].id if ready else None
        steps = minutes = 0

        for row, chapter in enumerate(chapters):
            number = QLabel(f"{row + 1}.")
            number.setStyleSheet("font-weight: bold;")
            self.grid.addWidget(number, row, 0, Qt.AlignmentFlag.AlignTop)

            done = "  ✓" if chapter.id in completed else ""
            label = QLabel(
                f"<b>{chapter.title}</b>{done}<br>"
                f"<span style='color: #c5ccd6;'>{chapter.description or chapter.summary}</span>"
            )
            label.setWordWrap(True)
            label.setTextFormat(Qt.TextFormat.RichText)
            self.grid.addWidget(label, row, 1)

            count = len(chapter.steps)
            info = QLabel(
                f"{count} steps" + (f"<br>~{chapter.minutes} min" if chapter.minutes else "")
                if chapter.ready
                else "coming soon"
            )
            info.setStyleSheet("color: #9aa4b2;")
            self.grid.addWidget(info, row, 2, Qt.AlignmentFlag.AlignTop)

            button = QPushButton("Again" if chapter.id in completed else "Start")
            button.setEnabled(chapter.ready)
            button.clicked.connect(
                lambda _=False, c=chapter.id: self.chapterRequested.emit(c)
            )
            self.grid.addWidget(button, row, 3, Qt.AlignmentFlag.AlignTop)

            if chapter.ready:
                steps += count
                minutes += chapter.minutes

        self.total.setText(
            f"{len(ready)} chapters · {steps} steps"
            + (f" · about {minutes} min in total" if minutes else "")
        )

    def place(self, bounds):
        # Word-wrapped labels: the height for the fixed width.
        self.adjustSize()
        height = self.heightForWidth(self.WIDTH)
        if height <= 0:
            height = self.sizeHint().height()
        height = min(height, bounds.height() - 16)
        x = bounds.center().x() - self.WIDTH // 2
        y = bounds.center().y() - height // 2
        x = max(bounds.left() + 8, x)
        y = max(bounds.top() + 8, y)
        self.setGeometry(x, y, self.WIDTH, height)
