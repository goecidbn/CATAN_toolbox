import json
from importlib.metadata import version
from pathlib import Path
from threading import RLock

import numpy as np
from scipy import sparse

from catan.gui.background_tasks.runtime import (
    current_task_context,
    progress_scope,
    progress_step,
)

_LOCK = RLock()


def trace_keys(data):
    keys = set()

    for session in data.sessions:
        keys.update(session.traces)
        keys.update(session.known_trace_keys)

        config = session.source_config
        if config is not None and "traces" in config.groups:
            keys.update(config.groups["traces"].fields)

    return tuple(sorted(keys))


def _cache_key(session, key, prctile, frequency):
    config = session.source_config
    recipe = None if config is None else config.to_dict()

    paths = {str(session.path)} if session.path else set()

    if config is not None and "traces" in config.groups:
        for spec in config.groups["traces"].fields.values():
            if getattr(spec, "source_path", None):
                paths.add(str(spec.source_path))

    sources = []

    for name in sorted(paths):
        path = Path(name).expanduser().resolve()

        try:
            stat = path.stat()
            stamp = (stat.st_size, stat.st_mtime_ns)
        except OSError:
            stamp = None

        sources.append((str(path), stamp))

    return json.dumps(
        [
            "event_rate_v1",
            version("event-estimator"),
            key,
            float(prctile),
            float(frequency),
            session.trace_revision,
            session.n_neurons,
            recipe,
            sources,
        ],
        sort_keys=True,
        default=str,
    )


def _load(data, sid):
    session = data.sessions[sid]

    if session.status["traces_loaded"]:
        return

    config = session.source_config
    if config is None or "traces" not in config.groups:
        return

    data.toggle_session_data(sid, "traces", to_present=True)


def calculate_event_rate(
    data,
    state,
    indexers=None,
    filters=(),
    key="S_dff",
    prctile=20.0,
    frequency=15.0,
    load_all=False,
    save_results=True,
):
    # Avoid overlapping event-rate requests loading/publishing twice.
    with _LOCK:
        return _calculate(
            data,
            state,
            indexers or {},
            key,
            prctile,
            frequency,
            load_all,
            save_results,
        )


def _calculate(
    data,
    state,
    indexers,
    key,
    prctile,
    frequency,
    load_all,
    save_results,
):
    from event_estimator import get_events
    from .calculations import requested_indices, drop_indexed_axes

    ids = state.assignments.copy()
    neurons = requested_indices(ids.shape[0], "neuron", indexers)
    sessions = requested_indices(ids.shape[1], "session", indexers)

    out = np.full((len(neurons), len(sessions)), np.nan)
    ctx = current_task_context()

    with progress_scope(0, 20, "Loading trace data"):
        if load_all:
            for i, sid in enumerate(range(len(data.sessions))):
                with progress_step(i, len(data.sessions)):
                    _load(data, sid)

    with progress_scope(20, 100, "Calculating event rates"):
        for j, sid in enumerate(sessions):
            with progress_step(
                j,
                len(sessions),
                message=f"Event rates: session {sid}",
            ):
                session = data.sessions[sid]
                fp_ids = ids[neurons, sid]
                needed = np.unique(fp_ids[fp_ids >= 0])

                if not len(needed):
                    continue

                token = _cache_key(session, key, prctile, frequency)
                cached = session.derived_statistics.get(token, {})
                missing = [int(fp) for fp in needed if str(int(fp)) not in cached]

                with progress_scope(0, 25, "Loading required traces"):
                    if missing:
                        _load(data, int(sid))

                # Loading can replace traces and invalidate older records.
                token = _cache_key(session, key, prctile, frequency)
                rates = dict(session.derived_statistics.get(token, {}))
                missing = [int(fp) for fp in needed if str(int(fp)) not in rates]
                traces = session.traces.get(key)

                with progress_scope(25, 95, "Estimating events"):
                    if missing and traces is not None:
                        if traces.ndim != 2:
                            raise ValueError(
                                f"Trace {key!r} must have shape "
                                "(footprints, frames)."
                            )

                        for i, fp in enumerate(missing):
                            with progress_step(i, len(missing)):
                                if fp >= traces.shape[0]:
                                    continue

                                row = traces[fp : fp + 1]
                                if sparse.issparse(row):
                                    row = row.toarray()

                                row = np.array(
                                    row,
                                    dtype=float,
                                    copy=True,
                                ).reshape(-1)
                                row[row < 1e-10] = 0.0

                                # Invalid data is unavailable, not silence.
                                if not row.size or not np.all(np.isfinite(row)):
                                    continue

                                if np.any(row < 0):
                                    print(row)
                                    print("less 0:", row[row < 0])
                                    raise ValueError(
                                        f"Trace {key!r} contains negative "
                                        "values; choose a nonnegative "
                                        "deconvolved trace."
                                    )

                                _, rate, _ = get_events(
                                    row,
                                    f=frequency,
                                    Ns_thr=1,
                                    prctile=prctile,
                                )

                                if np.isfinite(rate):
                                    rates[str(fp)] = float(rate)

                if ctx is not None:
                    ctx.check_cancelled()

                if save_results and rates:
                    session.derived_statistics = {
                        **session.derived_statistics,
                        token: rates,
                    }

                for i, fp in enumerate(fp_ids):
                    if fp >= 0:
                        out[i, j] = rates.get(str(int(fp)), np.nan)

    return drop_indexed_axes(
        out,
        ("neuron", "session"),
        indexers,
    )
