import numpy as np
from copy import deepcopy

from catan.core.structures.remap import Remapping
from catan.core import footprint_alignment


class AlignmentDraft:
    """Editable alignment of a session's background.

    ``session``: an unregistered background owner instead of
    data.sessions[session_id] (the overview background; ``external``).
    Its session_id is the number of sessions, so all sessions are earlier
    references.
    """

    def __init__(self, data, session_id, *, proposal=None, session=None):
        self.external = session is not None
        self.session_id = len(data.sessions) if self.external else session_id
        self.session = data.sessions[session_id] if session is None else session
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
        # Session compared against when applied (ManualAlignmentEditor.
        # record_reference): the manual alignment's reference.
        self.reference_path = None
        # Footprint correction calculated for a base geometry; toggled by
        # set_footprint_correction.
        self._footprint = None

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

    def store_footprint_correction(self, remap, result):
        """Cache the footprint correction ``result``
        (footprint_alignment.estimate_correction) calculated for ``remap``,
        the alignment made from the current values (make_remap)."""
        corrected = footprint_alignment.compose_correction(remap, result["matrix"])
        shift, angle = Remapping._rigid_parameters(self.dims, corrected.matrix)
        flow_info = deepcopy(remap.flow_info or {})
        flow_info["footprint_refinement"] = {
            key: result.get(key)
            for key in ("parameters", "accepted", "reason", "references", "matches", "median_before", "median_after")
        }
        self._footprint = {
            "base": (self.values.copy(), self._refined_flow),
            "values": np.array([shift[1], shift[0], angle], dtype=float),
            "flow": corrected.flow,
            "flow_info": flow_info,
            "applied_flow": None,
            "result": result,
        }

    @property
    def footprint_correction(self):
        """The cached footprint correction result, or None."""
        return None if self._footprint is None else self._footprint["result"]

    @property
    def footprint_correction_cached(self):
        """A correction is cached for the current (uncorrected) geometry."""
        footprint = self._footprint
        return (
            footprint is not None
            and np.array_equal(self.values, footprint["base"][0])
            and self._refined_flow is footprint["base"][1]
        )

    @property
    def footprint_correction_applied(self):
        footprint = self._footprint
        return (
            footprint is not None
            and np.array_equal(self.values, footprint["values"])
            and self._refined_flow is footprint["applied_flow"]
        )

    def set_footprint_correction(self, enabled):
        """Add (from the cached base geometry) or remove (back to it) the
        footprint correction."""
        footprint = self._footprint
        if footprint is None:
            return
        if enabled and self.footprint_correction_cached:
            self.values[:] = footprint["values"]
            # The flow moves with the correction; keyed to the new geometry.
            self._refined_flow = (
                None
                if footprint["flow"] is None
                else (self.geometry_key, footprint["flow"], footprint["flow_info"])
            )
            footprint["applied_flow"] = self._refined_flow
        elif not enabled and self.footprint_correction_applied:
            self.values[:] = footprint["base"][0]
            self._refined_flow = footprint["base"][1]

    def is_current(self, data):
        if self.external:
            # Session edits do not invalidate the background being edited.
            return getattr(data, "overview_background", None) is self.session

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

        remap.flow_info["references_used"] = (
            [] if self.reference_path is None else [self.reference_path]
        )

        if self.footprint_correction_applied:
            remap.flow_info["footprint_refinement"] = deepcopy(
                self._footprint["flow_info"]["footprint_refinement"]
            )
        return remap

    def _initial_geometry(self, data, current):
        if current is None:
            return np.zeros(2), 0.0

        if current.success or getattr(current, "confirmation", None):
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
