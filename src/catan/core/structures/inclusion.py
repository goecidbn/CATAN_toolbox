import numpy as np


def inclusion_mask(values, count, label="Included"):
    """Validate a boolean mask, accepting MATLAB row/column vectors."""
    array = np.asarray(values)

    if array.ndim == 2 and 1 in array.shape:
        array = array.reshape(-1)
    elif array.ndim == 0 and count == 1:
        array = array.reshape(1)

    if array.shape != (count,):
        raise ValueError(
            f"{label}: mask has shape {array.shape}; " f"expected ({count},)."
        )

    if array.dtype.kind not in "biuf" or not np.isin(array, (0, 1)).all():
        raise ValueError(f"{label}: inclusion values must be booleans or 0/1.")

    return array.astype(bool, copy=True)


def has_exclusions(session):
    values = np.asarray(session.included)
    return (
        values.shape == (session.n_neurons,)
        and values.size > 0
        and bool(np.any(values == 0))
    )


def snapshot_inclusion(sessions):
    """Store session descriptors, with arrays only for non-default masks."""
    records = {}

    for index, session in enumerate(sessions):
        mask = inclusion_mask(
            session.included,
            session.n_neurons,
            f"Session {index}",
        )

        record = {
            "source_path": str(session.path or ""),
            "n_footprints": int(session.n_neurons),
        }

        if np.any(~mask):
            record["included"] = mask

        records[f"s{index:06d}"] = record

    return {
        "version": 1,
        "sessions": records,
    }


def inclusion_to_restore(document, sessions):
    """Validate everything before returning masks; never mutate sessions."""
    if document is None:
        return None

    if not isinstance(document, dict) or document.get("version") != 1:
        raise ValueError("Unsupported session inclusion metadata.")

    records = document.get("sessions")
    expected = {f"s{index:06d}" for index in range(len(sessions))}

    if not isinstance(records, dict) or set(records) != expected:
        raise ValueError(
            "Saved inclusion metadata does not match the current "
            "session count. Load the matching sessions and retry."
        )

    masks = []

    for index, session in enumerate(sessions):
        record = records[f"s{index:06d}"]

        if not isinstance(record, dict):
            raise ValueError(f"Invalid inclusion record for session {index}.")

        count = record.get("n_footprints")
        if count != session.n_neurons:
            raise ValueError(
                f"Session {index} ({session.name}): saved inclusion "
                f"expects {count} footprints, but the session has "
                f"{session.n_neurons}.\n"
                f"Saved source: {record.get('source_path', 'unknown')}\n"
                "Load the matching session data and retry."
            )

        masks.append(
            inclusion_mask(
                record.get(
                    "included",
                    np.ones(session.n_neurons, dtype=bool),
                ),
                session.n_neurons,
                f"Session {index}",
            )
        )

    return masks
