import numpy as np
from copy import deepcopy

from catan.core.structures.remap import Remapping


class AlignmentDraft:
    def __init__(self, data, session_id, *, proposal=None):
        self.session_id = session_id
        self.session = data.sessions[session_id]
        self.expected_version = data.state.data_version

        self.template = self.session.background_template.copy()
        self.dims = tuple(self.session.dims)

        current = self.session.remap if proposal is None else proposal
        self.source_remap = current
        self.transpose = bool(current is not None and current.transpose)

        shift, angle = self._initial_geometry(data, current)

        # UI ordering: dx, dy, angle.
        self.initial = np.array([shift[1], shift[0], angle], dtype=float)
        self.values = self.initial.copy()

    @property
    def dirty(self):
        return not np.array_equal(self.values, self.initial)

    def is_current(self, data):
        return (
            data.state.data_version == self.expected_version
            and 0 <= self.session_id < len(data.sessions)
            and data.sessions[self.session_id] is self.session
        )

    def make_remap(self):
        dx, dy, angle = self.values
        if not np.isfinite(self.values).all():
            raise ValueError("Manual alignment values must be finite.")

        remap = Remapping.identity(self.dims)
        remap.method = "manual"
        remap.transpose = self.transpose
        remap.shift = np.array([dy, dx], dtype=float)
        remap.rotation = float(angle)
        remap.matrix = Remapping._rigid_matrix(self.dims, remap.shift, remap.rotation)

        # Preserve automatic comparison diagnostics after manual acceptance.
        remap.remap_data = deepcopy(getattr(self.source_remap, "remap_data", {}) or {})
        return remap

    def _initial_geometry(self, data, current):
        if current is None:
            return np.zeros(2), 0.0

        if current.success:
            return Remapping._rigid_parameters(self.dims, current.matrix)

        references = {
            str(item.path): item
            for item in data.sessions[: self.session_id]
            if item.path is not None and item.status["aligned"]
        }

        candidates = []
        for path, record in current.remap_data.items():
            reference = references.get(str(path))
            local = record.get("matrix")
            if reference is None or local is None:
                continue

            local = np.asarray(local, dtype=float)
            base = (
                np.eye(3)
                if reference.remap is None
                else np.asarray(reference.remap.matrix, dtype=float)
            )

            if (
                local.shape != (3, 3)
                or base.shape != (3, 3)
                or not np.isfinite(local).all()
                or not np.isfinite(base).all()
            ):
                continue

            # Retain finite estimates even when automatic quality checks failed.
            shift, angle = Remapping._rigid_parameters(self.dims, base @ local)
            if np.isfinite(shift).all() and np.isfinite(angle):
                candidates.append((*shift, angle))

        if not candidates:
            return np.zeros(2), 0.0

        values = np.median(np.asarray(candidates), axis=0)
        return values[:2], float(values[2])
