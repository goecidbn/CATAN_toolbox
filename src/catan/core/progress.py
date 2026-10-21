"""Optional progress reporting for headless core/tracking code.

``ctx`` is any object providing progress(percent), message(text) and
check_cancelled(), e.g. the GUI task context; None disables reporting.
A ProgressRange maps its own 0..100 onto a part of its parent's range and
provides the same methods, so it can be passed on as ``ctx`` again.
"""


class ProgressRange:
    __slots__ = ("ctx", "start", "stop")

    def __init__(self, ctx=None, start=0.0, stop=100.0):
        self.ctx = ctx
        self.start = float(start)
        self.stop = float(stop)

    def progress(self, value):
        if self.ctx is not None:
            value = min(100.0, max(0.0, float(value)))
            self.ctx.progress(self.start + (self.stop - self.start) * value / 100)

    def message(self, text):
        if self.ctx is not None:
            self.ctx.message(text)

    def check_cancelled(self):
        if self.ctx is not None:
            self.ctx.check_cancelled()

    def cancelled(self):
        return self.ctx is not None and self.ctx.cancelled()

    def range(self, start, stop, message=None):
        """Sub-range [start, stop] of this range's 0..100."""
        self.check_cancelled()

        if message:
            self.message(message)

        self.progress(start)

        return ProgressRange(self, start, stop)

    def step(self, index, total, *, start=0.0, stop=100.0, message=None):
        """Range of item ``index`` among ``total`` equal items."""
        width = (stop - start) / max(1, total)
        return self.range(start + width * index, start + width * (index + 1), message)
