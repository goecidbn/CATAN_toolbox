import numpy as np
from copy import deepcopy

from catan.core.structures.remap import Remapping


class AlignmentDraft:
    def __init__(self, data, session_id, *, proposal=None):
        self.session_id = session_id
        self.session = data.sessions[session_id]
        self.expected_version = data.state.data_version

        self._base_template = self.session.background_template.copy()
        self.background_transposed = False
        self.dims = tuple(self.session.dims)

        current = self.session.remap if proposal is None else proposal
        self.source_remap = current
        self.transpose = bool(current is not None and current.transpose)

        shift, angle = self._initial_geometry(data, current)

        # UI ordering: dx, dy, angle.
        self.initial = np.array([shift[1], shift[0], angle], dtype=float)
        self.values = self.initial.copy()
        self._refined_flow = None

    @property
    def template(self):
        if self.background_transposed:
            return self._base_template.T
        return self._base_template

    @property
    def can_transpose_background(self):
        return self._base_template.T.shape == self.dims

    @property
    def geometry_changed(self):
        return self.background_transposed or not np.array_equal(
            self.values, self.initial
        )

    @property
    def geometry_key(self):
        return (
            tuple(map(float, self.values)),
            bool(self.transpose),
            bool(self.background_transposed),
        )

    @property
    def has_refined_flow(self):
        return (
            self._refined_flow is not None
            and self._refined_flow[0] == self.geometry_key
        )

    @property
    def dirty(self):
        return self.geometry_changed or self.has_refined_flow

    def accept_refined_flow(self, remap):
        if remap.flow is None:
            raise ValueError("The proposed remapping contains no flow.")

        self._refined_flow = (
            self.geometry_key,
            remap.flow.copy(),
            deepcopy(remap.flow_info),
        )

    def clear_refined_flow(self):
        self._refined_flow = None

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
        flow = self.effective_flow
        remap.flow = None if flow is None else flow.copy()

        if self.has_refined_flow:
            remap.flow_info = deepcopy(self._refined_flow[2])

        elif (
            getattr(self.source_remap, "flow", None) is not None
            and self.geometry_changed
        ):
            remap.flow_info = {
                "status": "manual_geometry_changed",
                "reason": "Manual geometry changed after flow estimation.",
            }

        else:
            remap.flow_info = deepcopy(
                getattr(self.source_remap, "flow_info", {}) or {}
            )
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

    @property
    def effective_flow(self):
        if self.has_refined_flow:
            return self._refined_flow[1]

        if self.source_remap is None or self.geometry_changed:
            return None

        return getattr(self.source_remap, "flow", None)

    @property
    def flow_message(self):
        if self.has_refined_flow:
            return (
                "Refined optical flow is included in this preview. "
                "Apply changes to commit it."
            )

        source_flow = getattr(self.source_remap, "flow", None)

        if source_flow is not None and self.geometry_changed:
            return (
                "The geometry has changed; the previous flow is disabled. "
                "Use Refine flow to calculate a new correction, "
                "or Reset to restore the applied alignment."
            )

        if self.source_remap is not None:
            return self.source_remap.flow_message

        return "No flow correction applied."
