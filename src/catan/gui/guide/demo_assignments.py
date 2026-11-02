"""The demo's second assignment set (guide chapter "Comparing assignments").

The demo sessions registered as usual (default parameters), then changed
at random (seed ``SEED``) the way registration errors look:
``N_SPLITS`` tracks are cut into two neurons (the footprints from a random
later session on start a new neuron) and ``N_SWAPS`` footprints are
swapped between neighbouring neurons (union centroids closer than
``SWAP_RADIUS_PX``) in one session. Statistics move with their
footprints; swapped footprints get a low match probability. The unions are
rebuilt when the file is loaded.

Generate it (offscreen, ~1 min) into the demo folder:

    QT_QPA_PLATFORM=offscreen python -m catan.gui.guide.demo_assignments <demo folder>

The file name is ``tour.json`` ``demo.alternative_assignments``.
"""

import os
import sys
import tempfile
import time
from pathlib import Path

SEED = 29
N_SPLITS = 15
N_SWAPS = 10
SWAP_RADIUS_PX = 12.0


def perturb(assignments, seed=SEED):
    """Split ``N_SPLITS`` tracks and swap ``N_SWAPS`` footprints between
    neighbours, in place. Returns (splits, swaps)."""
    import numpy as np

    rng = np.random.default_rng(seed)
    ids = assignments.ids
    n_neurons, n_sessions = ids.shape
    present = ids >= 0

    # Swaps first (between original neurons), then splits.
    centroids = np.asarray(assignments.union.centroids)[:, :2]
    swaps = []
    order = rng.permutation(n_neurons)
    used = set()
    for a in order:
        if len(swaps) >= N_SWAPS:
            break
        if a in used:
            continue
        distance = np.hypot(*(centroids - centroids[a]).T)
        distance[a] = np.inf
        b = int(np.argmin(distance))
        if not np.isfinite(distance[b]) or distance[b] > SWAP_RADIUS_PX or b in used:
            continue
        shared = np.flatnonzero(present[a] & present[b])
        if not len(shared):
            continue
        s = int(rng.choice(shared))
        ids[a, s], ids[b, s] = ids[b, s], ids[a, s]
        for values in assignments.stats.values():
            values[[a, b], s] = values[[b, a], s]
        if "p_matched" in assignments.stats:
            assignments.stats["p_matched"][[a, b], s, ...] = 0.1
        used.update((int(a), b))
        swaps.append((int(a), b, s))

    candidates = [
        n for n in rng.permutation(n_neurons)
        if present[n].sum() >= 3 and n not in used
    ][:N_SPLITS]
    splits = []
    for n in candidates:
        sessions = np.flatnonzero(ids[n] >= 0)
        cut = int(rng.choice(sessions[1:]))
        new = assignments.ids.shape[0]
        assignments.pad_empty(n_neurons=1, n_sessions=0)
        ids = assignments.ids
        moved = np.flatnonzero((ids[n] >= 0) & (np.arange(n_sessions) >= cut))
        ids[new, moved] = ids[n, moved]
        ids[n, moved] = -1
        for values in assignments.stats.values():
            values[new, moved] = values[n, moved]
            values[n, moved] = np.nan
        splits.append((int(n), new, cut))
    return splits, swaps


def main(folder):
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    from PySide6.QtCore import QSettings
    from PySide6.QtWidgets import QApplication, QDialog, QMessageBox

    app = QApplication.instance() or QApplication([sys.argv[0]])
    # Confirmation dialogs (recipe review) are accepted, questions answered yes.
    QDialog.exec = lambda self: QDialog.DialogCode.Accepted
    QMessageBox.exec = lambda self: QMessageBox.StandardButton.Yes
    for kind in ("information", "warning", "critical", "question"):
        setattr(
            QMessageBox,
            kind,
            staticmethod(lambda *a, **k: QMessageBox.StandardButton.Yes),
        )

    from catan.gui.GUI_elements.main_window import MainWindow
    from catan.gui.guide.tour import load_tour

    folder = Path(folder).resolve()
    tour = load_tour()
    target = folder / tour.demo_alternative_assignments

    ini = Path(tempfile.mkdtemp()) / "settings.ini"
    settings = QSettings(str(ini), QSettings.Format.IniFormat)
    settings.setValue("guide/demo_path", str(folder))
    settings.setValue("guide/disabled", True)

    window = MainWindow(settings=settings)
    data, tasks, guide = window.data, window.state.tasks, window.guide

    def busy():
        return bool(tasks.tasks) or tasks.processing_busy() or tasks.processing_requested

    def wait(predicate, timeout=900):
        start = time.perf_counter()
        while time.perf_counter() - start < timeout:
            app.processEvents()
            time.sleep(0.01)
            if predicate() and not busy():
                return True
        raise TimeoutError("demo setup did not finish")

    guide.actions["set_demo_root"]()
    guide.actions["load_demo_recipe"]()
    wait(lambda: guide._check("sessions_loaded"))
    guide.actions["fit_model"]()
    wait(lambda: guide._check("model_fitted"))

    guide.actions["register_neurons"]()
    wait(lambda: guide._check("neurons_registered"))

    original = data.assignments.ids.shape[0]
    splits, swaps = perturb(data.assignments)
    # No provenance: the set belongs to whatever demo sessions are loaded.
    data.queue_save_result("assignments", str(target), provenance={})
    wait(lambda: target.is_file())
    print(
        f"{target}: {original} -> {data.assignments.ids.shape[0]} neurons, "
        f"{len(splits)} split tracks {splits}, {len(swaps)} swaps {swaps}"
    )
    sys.stdout.flush()
    os._exit(0)


if __name__ == "__main__":
    main(sys.argv[1])
