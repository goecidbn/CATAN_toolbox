from copy import deepcopy

from PySide6.QtCore import QThreadPool, Qt, Signal, Slot
from PySide6.QtWidgets import (
    QWidget,
    QLabel,
    QToolButton,
    QHBoxLayout,
    QVBoxLayout,
    QMessageBox,
    QDialog,
)

from catan.core.io.isolated_read import read_operation
from catan.gui.background_tasks.Worker import Worker
from catan.gui.background_tasks.runtime import current_task_context


def _run_inspection(operation, path, parameters):
    return read_operation(
        operation,
        path,
        ctx=current_task_context(),
        timeout=60.0,
        **parameters,
    )


def _cancel_worker(lifetime):
    worker = lifetime.get("worker")
    if worker is not None:
        worker.cancel()


class FileInspection(QWidget):
    result = Signal(object)
    failed = Signal()
    cancelled = Signal()
    busy_changed = Signal(bool)

    # Serialize metadata requests so creating several session rows
    # does not start a separate reader for every row at once.
    _pool = None
    _navigation_pool = None

    def __init__(self, parent=None):
        super().__init__(parent)

        if FileInspection._pool is None:
            FileInspection._pool = QThreadPool()
            FileInspection._pool.setMaxThreadCount(1)

        if FileInspection._navigation_pool is None:
            FileInspection._navigation_pool = QThreadPool()
            FileInspection._navigation_pool.setMaxThreadCount(1)

        self._generation = 0
        self._running_generation = None
        self._worker = None
        self._pending = None
        self._last_request = None
        self._details = ""
        self._error_text = ""

        self.failure_info = None

        # This callback deliberately does not capture the widget.
        self._lifetime = {"worker": None}
        lifetime = self._lifetime
        self.destroyed.connect(lambda *_: _cancel_worker(lifetime))

        self.label = QLabel()
        self.label.setTextFormat(Qt.TextFormat.PlainText)
        self.label.setWordWrap(True)

        self.retry_button = QToolButton()
        self.retry_button.setText("Retry")

        self.cancel_button = QToolButton()
        self.cancel_button.setText("Cancel")

        self.details_button = QToolButton()
        self.details_button.setText("Details")

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(5)
        layout.addWidget(self.label)

        button_row = QHBoxLayout()
        button_row.setContentsMargins(0, 0, 0, 0)
        button_row.setSpacing(8)

        for button in (
            self.retry_button,
            self.cancel_button,
            self.details_button,
        ):
            button_row.addWidget(button)
            button.setEnabled(False)

        button_row.addStretch(1)
        layout.addLayout(button_row)

        self.message_formatter = None

        self.retry_button.clicked.connect(self.retry)
        self.cancel_button.clicked.connect(self.cancel)
        self.details_button.clicked.connect(self.show_details)

    def set_message(self, message):
        full_message = str(message)
        displayed = (
            self.message_formatter(full_message)
            if self.message_formatter is not None
            else full_message
        )

        self.label.setText(displayed)
        self.label.setToolTip(full_message if displayed != full_message else "")

    def _set_busy(self, busy):
        self.cancel_button.setEnabled(busy)
        self.retry_button.setEnabled(not busy and self._last_request is not None)
        self.busy_changed.emit(busy)

    def invalidate(self):
        """Cancel and discard results belonging to the previous request."""
        self._generation += 1
        self.failure_info = None
        self._pending = None
        self._last_request = None

        _cancel_worker(self._lifetime)

        self._details = ""
        self.details_button.setEnabled(False)
        self.label.clear()
        self._set_busy(False)

    def start(self, operation, path, **parameters):
        self.invalidate()

        request = (operation, str(path), deepcopy(parameters))
        self._last_request = request
        self._pending = (self._generation, request)

        self.label.setText("Waiting to inspect source…")
        self._set_busy(True)
        self._launch_pending()

    @Slot()
    def retry(self):
        if self._last_request is not None:
            operation, path, parameters = self._last_request
            self.start(operation, path, **parameters)

    @Slot()
    def cancel(self):
        request = self._last_request

        self.invalidate()
        self._last_request = request
        self._set_busy(False)

        self.label.setText("Inspection cancelled. Retry when ready.")
        self.cancelled.emit()

    def _launch_pending(self):
        if self._worker is not None or self._pending is None:
            return

        generation, request = self._pending
        self._pending = None
        self._running_generation = generation
        self._error_text = ""

        worker = Worker(_run_inspection, *request)
        self._worker = worker
        self._lifetime["worker"] = worker

        connection = Qt.ConnectionType.QueuedConnection
        worker.signals.message.connect(self._on_message, connection)
        worker.signals.error.connect(self._on_error, connection)
        worker.signals.finished.connect(self._on_finished, connection)

        navigation_operations = {
            "browse_directory",
            "select_path",
            "create_directory",
        }

        pool = (
            FileInspection._navigation_pool
            if request[0] in navigation_operations
            else FileInspection._pool
        )
        pool.start(worker)

    @Slot(str)
    def _on_message(self, message):
        if self._running_generation == self._generation:
            self.set_message(message)

    @Slot(str)
    def _on_error(self, text):
        self._error_text = text

    @Slot(object)
    def _on_finished(self, result):
        worker = self._worker
        current = self._running_generation == self._generation

        self._worker = None
        self._lifetime["worker"] = None

        if current:
            self._set_busy(False)

            if worker.is_failed():
                self._details = self._error_text
                self.details_button.setEnabled(bool(self._details))

                info = getattr(worker, "failure_info", None) or {}

                summary = info.get("summary") or "Source inspection failed."
                recovery = info.get("recovery") or (
                    "Restore source access or correct the source/configuration, "
                    "then retry."
                )

                self.failure_info = {
                    "summary": summary,
                    "recovery": recovery,
                }

                self.label.setText(
                    f"{summary}\n{recovery}\n" "See Details for the underlying error."
                )
                self.failed.emit()

            elif worker.is_cancelled():
                self.label.setText("Inspection cancelled. Retry when ready.")
                self.cancelled.emit()

            else:
                self.label.setText("Source inspected.")
                self.result.emit(result)

        self._launch_pending()

    @Slot()
    def show_details(self):
        box = QMessageBox(self)
        box.setWindowTitle("Source inspection failed")
        box.setTextFormat(Qt.TextFormat.PlainText)
        box.setText(self.label.text())
        box.setDetailedText(self._details)
        box.setStandardButtons(QMessageBox.StandardButton.Close)
        box.exec()
        box.deleteLater()

    @staticmethod
    def get_result(
        operation,
        path,
        *,
        state,
        parent=None,
        title="Inspecting source",
        **parameters,
    ):
        """Return (completed, result), keeping the GUI responsive."""
        dialog = QDialog(parent)
        dialog.setWindowTitle(title)
        dialog.resize(540, 160)

        layout = QVBoxLayout(dialog)
        inspection = FileInspection(dialog)
        layout.addWidget(inspection)

        outcome = {
            "result": None,
            "failure": None,
        }

        def on_result(result):
            outcome["result"] = result
            dialog.accept()

        def on_failure():
            outcome["failure"] = dict(
                inspection.failure_info
                or {
                    "summary": "Source inspection failed.",
                    "recovery": (
                        "Restore source access or correct the path, " "then try again."
                    ),
                }
            )
            dialog.reject()

        inspection.result.connect(on_result)
        inspection.failed.connect(on_failure)
        inspection.cancelled.connect(dialog.reject)

        dialog.finished.connect(lambda _result: inspection.invalidate())

        inspection.start(operation, path, **parameters)

        try:
            completed = dialog.exec() == QDialog.DialogCode.Accepted
        finally:
            inspection.invalidate()
            dialog.deleteLater()

        failure = outcome["failure"]

        if failure is not None:
            state.issue(
                "warning",
                "Source inspection failed",
                f"{failure['summary']}\n\n{failure['recovery']}",
                parent=parent,
            )
            return False, None

        return completed, outcome["result"] if completed else None
