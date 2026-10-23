"""Memory diagnostics for "Print debug info": where does the process size go?"""

import gc
import os

import numpy as np
from scipy import sparse

from catan.core.memory import release_free_memory


def _array_bytes(obj, seen, depth=0):
    """NumPy/sparse bytes reachable from ``obj`` (CATAN objects, containers)."""
    if id(obj) in seen or depth > 6:
        return 0
    seen.add(id(obj))

    if isinstance(obj, np.ndarray):
        # A view counts once, through its base array.
        base = obj
        while isinstance(base.base, np.ndarray):
            base = base.base
        if base is not obj:
            return _array_bytes(base, seen, depth)
        return obj.nbytes

    if sparse.issparse(obj):
        return sum(
            getattr(obj, name).nbytes
            for name in ("data", "indices", "indptr", "row", "col")
            if isinstance(getattr(obj, name, None), np.ndarray)
        )

    if isinstance(obj, dict):
        return sum(_array_bytes(value, seen, depth + 1) for value in obj.values())

    if isinstance(obj, (list, tuple)):
        return sum(_array_bytes(value, seen, depth + 1) for value in obj)

    if (
        hasattr(obj, "__dict__")
        and not isinstance(obj, type)
        and type(obj).__module__.startswith("catan")
    ):
        return sum(_array_bytes(value, seen, depth + 1) for value in vars(obj).values())

    return 0


def _pending_glir_bytes():
    """Array data waiting in vispy command queues (sent when canvases draw).
    Large values mean canvases update without being drawn."""
    try:
        from vispy.gloo import glir
    except ImportError:
        return 0, 0

    held = commands = 0

    for obj in gc.get_objects():
        if isinstance(obj, glir._GlirQueueShare):
            for command in obj._commands:
                commands += 1
                held += sum(a.nbytes for a in command if isinstance(a, np.ndarray))

    return commands, held


def memory_report(data):
    """Lines describing process memory and the data CATAN holds."""
    try:
        import psutil

        process = psutil.Process(os.getpid())
        rss = lambda: process.memory_info().rss
    except ImportError:
        rss = lambda: float("nan")

    def mb(value):
        return f"{value / 2**20:,.0f} MB"

    before = rss()
    gc.collect()
    release_free_memory()
    after = rss()

    seen = set()
    sessions = data.sessions
    parts = {
        "footprints": sum(_array_bytes(s.footprints, seen) for s in sessions),
        "traces": sum(_array_bytes(getattr(s, "_traces", None), seen) for s in sessions),
        "backgrounds": sum(
            _array_bytes(getattr(s, "background", None), seen)
            + _array_bytes(getattr(s, "background_template", None), seen)
            for s in sessions
        ),
        "alignment": sum(_array_bytes(getattr(s, "remap", None), seen) for s in sessions),
        "sessions (other)": sum(_array_bytes(s, seen) for s in sessions),
        "models": _array_bytes(getattr(data, "_model", {}), seen),
        "assignments + unions": _array_bytes(getattr(data, "_assignments", {}), seen),
    }

    engine = getattr(data, "statistic_engine", None)
    if engine is not None:
        parts["statistics cache"] = _array_bytes(getattr(engine, "_cache", {}), seen)
        parts["suggestion cache"] = _array_bytes(
            getattr(engine, "_sample_cache", {}), seen
        )

    commands, pending = _pending_glir_bytes()

    lines = [
        "=== Memory ===",
        f"Process (RSS): {mb(before)}; after returning freed memory: {mb(after)}",
        f"Arrays held by CATAN: {mb(sum(parts.values()))}",
    ]
    lines += [f"  {name}: {mb(value)}" for name, value in parts.items() if value]
    lines.append(
        f"Pending display updates (vispy): {commands} commands, {mb(pending)}"
    )

    return lines
