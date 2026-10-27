from typing import Any, Literal, Optional, TypeVar
from dataclasses import dataclass
from pathlib import Path
import h5py
import h5py
import numpy as np
from scipy import sparse
import cv2

from catan.core.image_correlation import (
    calculate_img_correlation,
    calculate_shift_score_map,
)
from catan.core.alignment import (
    get_session_remap,
    _build_remap,
    _shift_sparse_bilinear,
)
from catan.core.io import load_file, save_file, NATIVE_REMAP_CONFIG
from catan.core.structures.load_config import LoadConfig, FieldSpec
from catan.core.alignment import calculate_residual_flow

# Flow alignment quality relative to the dataset: a correlation below this
# fraction of the median accepted correlation of the reference sessions is
# low (845ad: median 0.80, normal 0.58–0.88, failed 0.35/0.38; other mice
# reach ~0.3 on average). Needs MIN_CORRELATION_HISTORY reference values.
RELATIVE_FLOW_CORRELATION = 0.6
MIN_CORRELATION_HISTORY = 3
# A low correlation is accepted if at least AGREEMENT_MIN of the
# AGREEMENT_REFERENCES most recent other references place the session within
# AGREEMENT_PX of the selected alignment (consistent, just a noisy session).
AGREEMENT_REFERENCES = 3
AGREEMENT_MIN = 2
AGREEMENT_PX = 3.0
# An older reference replaces the most recent one only if its score
# (correlation + neuron agreement) is higher by more than this.
FLOW_PREFERENCE_MARGIN = 0.05
# Flow candidates always evaluated (most recent first): a neighbouring
# session may image the plane worse than an older one (ID28 S9: 20240727
# correlation 0.72 / 16 % of neurons matched, 20240726 0.87 / 30 %).
FLOW_RECENT_REFERENCES = 5
# Neuron agreement: fraction of mutual nearest centroid pairs within this
# distance (px) after mapping, among the smaller of the two populations.
NEURON_MATCH_PX = 3.0

MatrixT = TypeVar("MatrixT", sparse.csc_matrix, np.ndarray)


@dataclass(frozen=True)
class AlignmentReport:
    success: bool
    reason: str | None

    shift: np.ndarray | None
    rotation: float | None
    total_shift: float

    correlation: float | None
    correlation_zscore: float | None

    n_references: int
    n_successful_references: int

    remap_data: dict[str, dict]
    method: str = "automatic"


def summarize_rejections(candidates) -> str:
    """One line per kind of rejection of the flow references, with the
    number of references and the range of the measured values (instead
    of one full reason per reference)."""
    groups = {}
    for record in candidates.values():
        reason = record.get("reason") or "Candidate rejected."
        # The first sentence names the kind; the rest are its numbers.
        kind = reason.split(". ")[0].rstrip(".")
        groups.setdefault(kind, []).append(record)

    def value_range(records, key):
        values = [float(r[key]) for r in records if r.get(key) is not None]
        if not values:
            return None
        low, high = min(values), max(values)
        return f"{low:.1f} px" if np.isclose(low, high) else f"{low:.1f}–{high:.1f} px"

    lines = [f"No reference accepted ({len(candidates)} tried)."]
    for kind, records in sorted(groups.items(), key=lambda item: -len(item[1])):
        details = []
        p95 = value_range(records, "p95_displacement")
        if p95:
            limit = records[0].get("p95_displacement_limit")
            details.append(
                f"95th percentile {p95}" + (f" (limit {limit:.0f} px)" if limit else "")
            )
        maximum = value_range(records, "max_displacement")
        if maximum:
            limit = records[0].get("hard_displacement_limit")
            details.append(
                f"maximum {maximum}" + (f" (hard limit {limit:.0f} px)" if limit else "")
            )
        count = f"{len(records)} reference" + ("s" if len(records) != 1 else "")
        lines.append(f"• {kind}: {count}" + ("; " + "; ".join(details) if details else "") + ".")
    return "\n".join(lines)


@dataclass
class Remapping:

    source_type: str = "remapping"
    source_config: LoadConfig | None = None

    shift: Optional[np.ndarray] = None
    c_max: Optional[np.ndarray] = None
    c_zscored: Optional[np.ndarray] = None
    flow: Optional[np.ndarray] = None
    transpose: bool = False

    max_shift: float = 50.0
    min_zcorr: float = 3.0
    c_min: float = 0.1

    HDF5_VERSION = 1

    @property
    def total_shift(self):
        if self.shift is not None:
            return np.sqrt(np.square(self.shift).sum())
        else:
            return np.nan

    @property
    def is_valid(self):
        return self.c_zscored is not None and self.shift is not None

    @staticmethod
    def _from_file(
        path: str | Path,
        fields_to_load: dict[str, dict[str, FieldSpec]] | None = None,
    ) -> "Remapping":
        fields_to_load = fields_to_load or LoadConfig.fields_from_resource(
            NATIVE_REMAP_CONFIG, enabled_only=False
        )
        data = load_file(path, fields_to_load)
        return Remapping._from_dict(data)

    @staticmethod
    def _from_dict(data: dict) -> "Remapping":
        remapping = Remapping()
        remapping.register_data(**data)
        return remapping

    def __init__(
        self,
        template: np.ndarray | None = None,
        references: dict[str, dict[str, Any]] | None = None,
        *,
        use_optical_flow: bool = False,
        evaluate: bool = True,
        max_shift: float = 50.0,
        max_rotation: float = 10.0,
        min_corr: float = 0.1,
        min_zcorr: float = 3.0,
        rotation_step: float = 1.0,
        rotation_refine_step: float = 0.1,
        require_confirmation: bool = True,
        allowed_offsets=(),
        min_flow_correlation: float = 0.10,
        moving_centroids=None,
    ):
        """``require_confirmation``: a flow-only alignment departing from the
        default (see estimate_flow) is proposed unsuccessful, with the
        description in ``confirmation``, for the user to confirm. Otherwise
        it is accepted and the description kept in flow_info["deviation"].

        ``allowed_offsets``: further translations (dx, dy) in the common
        frame around which shifts up to ``max_shift`` are accepted (e.g. the
        imaging window jumped and stayed there); the origin always is.

        ``min_flow_correlation``: background correlation (after warping)
        a flow alignment needs against a reference.

        ``moving_centroids``: neuron centroids (x, y) in the template's
        pixel frame; scored against the references' "centroids" when
        choosing among flow candidates (not stored)."""

        # Transform from THIS SESSION'S original template
        # into CATAN's common/global coordinate system.
        self.matrix = np.eye(3, dtype=np.float64)

        self.method = "automatic"

        self.shift: np.ndarray | None = None
        self.rotation: float | None = None
        self.flow = None
        self.flow_info = {}

        self.transpose = False

        # path -> pairwise alignment information
        self.remap_data: dict[str, dict[str, Any]] = {}

        self.max_shift = float(max_shift)
        self.allowed_offsets = np.vstack(
            [np.zeros((1, 2))]
            + [np.asarray(offset, dtype=float).reshape(1, 2) for offset in allowed_offsets]
        )
        self.max_rotation = float(max_rotation)
        self.min_flow_correlation = float(min_flow_correlation)
        self.c_min = float(min_corr)
        self.min_zcorr = float(min_zcorr)

        self.rotation_step = float(rotation_step)
        self.rotation_refine_step = float(rotation_refine_step)

        self.success = False
        self.dims = None

        self.require_confirmation = bool(require_confirmation)
        self.moving_centroids = (
            None
            if moving_centroids is None or not len(moving_centroids)
            else np.asarray(moving_centroids, dtype=np.float64).reshape(-1, 2)
        )
        # Why the proposed alignment needs the user's confirmation, or None.
        self.confirmation = None

        if evaluate and template is not None and references:
            self.evaluate(template, references, use_optical_flow=use_optical_flow)

    def _allowed_offsets(self):
        """Origin plus allowed offsets; older remaps have none."""
        offsets = getattr(self, "allowed_offsets", None)
        if offsets is None or np.size(offsets) == 0:
            return np.zeros((1, 2))
        return np.asarray(offsets, dtype=float).reshape(-1, 2)

    def offset_distance(self, translation):
        """Distance (px) of a translation (dx, dy) to the nearest allowed
        position (origin or allowed offsets)."""
        offsets = self._allowed_offsets()
        translation = np.asarray(translation, dtype=float).reshape(1, 2)
        return float(np.min(np.linalg.norm(offsets - translation, axis=1)))

    def register_data(self, **data):
        self.shift = data.get("shift")
        self.c_max = data.get("c_max")
        self.c_zscored = data.get("c_zscored")
        self.flow = data.get("flow")
        self.transpose = data.get("transpose", False)

        if self.shift is not None or self.flow is not None:
            self.success = True
            return

    def evaluate(
        self,
        template: np.ndarray,
        references: dict[str, dict[str, Any]],
        *,
        use_optical_flow: bool = False,
    ):

        template = np.asarray(template, dtype=np.float32)

        self.dims = tuple(template.shape)
        self.remap_data = {}
        self.flow = None
        self.flow_info = {}
        self.confirmation = None
        if use_optical_flow:
            self.method = "automatic_flow"
            self.shift = np.zeros(2, dtype=np.float64)
            self.rotation = 0.0
            self.matrix = np.eye(3, dtype=np.float64)

            # Identity geometry lets estimate_flow compare the original
            # moving image with references already in the common frame.
            # This is provisional; acceptance is determined below.
            self.success = True
            try:
                self.estimate_flow(
                    template,
                    references,
                    max_displacement=self.max_shift,
                    hard_max_displacement=1.5 * self.max_shift,
                    residual=False,
                    min_correlation=getattr(self, "min_flow_correlation", 0.10),
                )
            finally:
                self.success = self.flow is not None

            self.flow_info["mode"] = "flow_only"

            if self.success:
                selected = self.flow_info["candidates"][self.flow_info["reference"]]

                self._factor_flow_rigid(selected["rigid_matrix"])
                self.flow_info["representation"] = "rigid_plus_residual"
                self.remap_data = self._flow_comparison_records(references)

                deviation = self.flow_info.get("deviation")
                if deviation and self.require_confirmation:
                    note = self.flow_info.get("reference_note")
                    self.confirmation = f"{deviation} {note}" if note else deviation
                    self.success = False

            return

        global_candidates = []

        # Optical flow cannot sensibly be aggregated across
        # several independently transformed references.

        for reference_path, reference_data in references.items():

            reference = np.asarray(reference_data["template"], dtype=np.float32)

            if reference.shape != template.shape:

                self.remap_data[reference_path] = {
                    "shift": None,
                    "rotation": None,
                    "c_max": None,
                    "c_zscored": None,
                    "success": False,
                    "reason": ("dimension_mismatch"),
                    "matrix": None,
                    "global_matrix": None,
                }

                continue

            result = self._estimate_pairwise(
                reference, template, use_optical_flow=False
            )

            self.remap_data[reference_path] = result

            if not result["success"]:
                continue

            reference_matrix = np.asarray(
                reference_data.get("matrix", np.eye(3)), dtype=float
            )

            # current -> reference -> global
            global_matrix = reference_matrix @ result["matrix"]

            result["global_matrix"] = global_matrix

            # Global translation (dx, dy) near the origin or an allowed offset.
            global_shift, _ = self._rigid_parameters(self.dims, global_matrix)
            if self.offset_distance(global_shift[::-1]) > self.max_shift:
                result["success"] = False
                result["reason"] = "shift_too_large"
                continue

            global_candidates.append(global_matrix)

        # ==========================================================
        # No usable reference
        # ==========================================================

        if not global_candidates:

            self.shift = None
            self.rotation = None

            self.matrix = np.eye(3, dtype=np.float64)

            self.flow = None
            self.success = False

            return

        # ==========================================================
        # Convert each candidate global transform back into
        # interpretable shift + rotation parameters.
        # ==========================================================

        candidate_shifts = []
        candidate_rotations = []

        for matrix in global_candidates:

            shift, rotation = self._rigid_parameters(self.dims, matrix)
            candidate_shifts.append(shift)
            candidate_rotations.append(rotation)

        self.shift = np.nanmedian(np.asarray(candidate_shifts), axis=0)
        self.rotation = float(np.nanmedian(candidate_rotations))
        self.matrix = self._rigid_matrix(self.dims, self.shift, self.rotation)

        self.success = True

        if use_optical_flow:
            self.estimate_flow(
                template,
                references,
                min_correlation=getattr(self, "min_flow_correlation", 0.10),
            )

    def _estimate_pairwise(
        self,
        reference: np.ndarray,
        template: np.ndarray,
        *,
        use_optical_flow: bool = False,
    ) -> dict[str, Any]:

        if self.max_rotation == 0.0:
            shift, flow, corr, zscore = get_session_remap(
                reference,
                template,
                self.dims,
                None,
                use_optical_flow=use_optical_flow,
            )
            return self._pairwise_result(shift, flow, corr, zscore, 0.0)

        # Search somewhat beyond the accepted maximum so
        # "rotation_too_large" can actually be diagnosed.
        rotation_search = max(self.max_rotation * 1.5, self.max_rotation + 2.0)

        def evaluate_angle(angle):

            rotation_matrix = self._rotation_matrix(self.dims, angle)

            rotated = self._warp_dense(template, rotation_matrix)

            shift, _, corr, zscore = get_session_remap(
                reference, rotated, self.dims, None, use_optical_flow=False
            )

            return (shift, corr, zscore)

        # ==========================================================
        # Coarse rotation search
        # ==========================================================

        angles = np.arange(
            -rotation_search,
            rotation_search + self.rotation_step / 2,
            self.rotation_step,
        )

        candidates = []

        for angle in angles:
            shift, corr, zscore = evaluate_angle(angle)

            if corr is None or not np.isfinite(corr):
                continue

            candidates.append((float(corr), float(angle), shift, zscore))

        if not candidates:

            return {
                "shift": None,
                "rotation": None,
                "c_max": None,
                "c_zscored": None,
                "success": False,
                "reason": ("correlation_unavailable"),
                "matrix": None,
            }

        _, best_angle, _, _ = max(candidates, key=lambda item: item[0])

        # ==========================================================
        # Fine rotation search around coarse optimum
        # ==========================================================

        fine_angles = np.arange(
            best_angle - self.rotation_step,
            best_angle + self.rotation_step + self.rotation_refine_step / 2,
            self.rotation_refine_step,
        )

        candidates = []

        for angle in fine_angles:

            shift, corr, zscore = evaluate_angle(angle)

            if corr is None or not np.isfinite(corr):
                continue

            candidates.append((float(corr), float(angle), shift, zscore))

        if not candidates:

            return {
                "shift": None,
                "rotation": None,
                "c_max": None,
                "c_zscored": None,
                "success": False,
                "reason": ("correlation_unavailable"),
                "matrix": None,
            }

        _, best_angle, _, _ = max(candidates, key=lambda item: item[0])

        # ==========================================================
        # Final calculation at selected angle
        # ==========================================================

        rotation_matrix = self._rotation_matrix(self.dims, best_angle)

        rotated = self._warp_dense(template, rotation_matrix)

        shift, flow, corr, zscore = get_session_remap(
            reference, rotated, self.dims, None, use_optical_flow=(use_optical_flow)
        )

        return self._pairwise_result(shift, flow, corr, zscore, best_angle)

    def _pairwise_result(self, shift, flow, corr, zscore, angle):
        if shift is None:
            return {
                "shift": None,
                "rotation": angle,
                "c_max": corr,
                "c_zscored": zscore,
                "success": False,
                "reason": "shift_unavailable",
                "matrix": None,
            }

        shift = np.asarray(shift, dtype=float)
        total_shift = float(np.linalg.norm(shift))

        reason = None
        if not np.all(np.isfinite(shift)):
            reason = "shift_invalid"
        elif total_shift > self.max_shift + float(
            np.max(np.linalg.norm(self._allowed_offsets(), axis=1))
        ):
            # Checked against the allowed positions in the common frame
            # once the reference's transform is known (evaluate()).
            reason = "shift_too_large"
        elif abs(angle) > self.max_rotation:
            reason = "rotation_too_large"
        elif corr is None or not np.isfinite(corr) or corr < self.c_min:
            reason = "correlation_too_low"
        elif zscore is None or not np.isfinite(zscore) or zscore < self.min_zcorr:
            reason = "zscore_too_low"

        return {
            "shift": shift,
            "rotation": float(angle),
            "c_max": None if corr is None else float(corr),
            "c_zscored": None if zscore is None else float(zscore),
            "success": reason is None,
            "reason": reason,
            "matrix": self._rigid_matrix(self.dims, shift, angle),
            "flow": flow,
        }

    def apply_remap(self, A: MatrixT, use_optical_flow: bool = True) -> MatrixT:
        if self.transpose:
            A = self.fix_transpose(A)

        if not self.success:
            return A

        use_flow = use_optical_flow and self.flow is not None
        matrix = np.asarray(self.matrix, dtype=np.float64)
        identity = np.allclose(matrix, np.eye(3), rtol=0, atol=1e-8)

        if identity and not use_flow:
            return A

        # Preserve the inexpensive translation-only sparse path.
        translation_only = np.allclose(matrix[:2, :2], np.eye(2), rtol=0, atol=1e-8)
        if sparse.issparse(A) and translation_only and not use_flow:
            return _shift_sparse_bilinear(
                A,
                self.dims,
                matrix[1, 2],
                matrix[0, 2],
                order="C",
            )

        x_map, y_map = self.sampling_maps(use_optical_flow=use_optical_flow)

        def warp(image):
            image = np.asarray(image, dtype=np.float32)
            if image.shape != tuple(self.dims):
                raise ValueError(
                    f"Image shape {image.shape} does not match {self.dims}."
                )

            # One interpolation from the original image.
            return cv2.remap(
                image,
                x_map,
                y_map,
                interpolation=cv2.INTER_LINEAR,
                borderMode=cv2.BORDER_CONSTANT,
                borderValue=0,
            )

        if sparse.issparse(A):
            source = sparse.csc_matrix(A)

            if source.shape[0] != int(np.prod(self.dims)):
                raise ValueError("Footprint pixel count does not match dimensions.")

            if source.shape[1] == 0:
                return source.copy()

            columns = []
            for index in range(source.shape[1]):
                image = source.getcol(index).toarray().reshape(self.dims)
                columns.append(sparse.csc_matrix(warp(image).reshape(-1, 1)))

            return sparse.hstack(columns, format="csc")

        if isinstance(A, np.ndarray):
            return warp(A)

        raise TypeError("Expected a NumPy image or sparse footprint matrix.")

    def sampling_maps(self, *, use_optical_flow=True):
        """Map global output pixels to original-image sampling coordinates.

        Flow is (dx, dy), defined in the global output coordinate system.
        Transposition is handled separately by apply_remap().
        """
        yy, xx = np.indices(self.dims, dtype=np.float32)

        if use_optical_flow and self.flow is not None:
            flow = np.asarray(self.flow, dtype=np.float32)
            if flow.shape != tuple(self.dims) + (2,):
                raise ValueError("Flow shape does not match session dimensions.")
            if not np.isfinite(flow).all():
                raise ValueError("Flow contains nonfinite values.")

            xx = xx + flow[..., 0]
            yy = yy + flow[..., 1]

        inverse = np.linalg.inv(np.asarray(self.matrix, dtype=np.float64))

        source_x = inverse[0, 0] * xx + inverse[0, 1] * yy + inverse[0, 2]
        source_y = inverse[1, 0] * xx + inverse[1, 1] * yy + inverse[1, 2]
        return (
            source_x.astype(np.float32),
            source_y.astype(np.float32),
        )

    def valid_mask(self, *, use_optical_flow=True):
        x, y = self.sampling_maps(use_optical_flow=use_optical_flow)
        height, width = self.dims
        return (
            np.isfinite(x)
            & np.isfinite(y)
            & (x >= 0)
            & (x <= width - 1)
            & (y >= 0)
            & (y <= height - 1)
        )

    @property
    def flow_message(self):
        message = self._flow_message()
        refinement = (self.flow_info or {}).get("footprint_refinement")
        if refinement:
            dx, dy, angle = refinement.get("parameters") or (0.0, 0.0, 0.0)
            message += (
                f"\nFootprint refinement: dx {dx:+.2f} px, dy {dy:+.2f} px, "
                f"rotation {angle:+.2f}°"
                + (f" ({refinement['matches']} matched cells)." if refinement.get("matches") else ".")
                if refinement.get("accepted")
                else f"\nFootprint refinement not applied: {refinement.get('reason')}."
            )
        return message

    def _flow_message(self):
        info = self.flow_info or {}
        status = info.get("status")

        if self.flow is not None:
            reference = info.get("reference")
            label = ""
            if reference:
                record = info.get("candidates", {}).get(reference, {})
                # Session name ("S12 Session13") rather than its file path.
                name = record.get("name") or Path(str(reference)).name
                quality = []
                if record.get("correlation_after") is not None:
                    quality.append(f"correlation {record['correlation_after']:.2f}")
                if record.get("neuron_agreement") is not None:
                    quality.append(f"{100 * record['neuron_agreement']:.0f} % of neurons matched")
                label = f" Reference: {name}" + (f" ({', '.join(quality)})" if quality else "") + "."
            note = info.get("reference_note")
            return "Flow correction applied." + label + (f" {note}" if note else "")

        if info.get("mode") == "flow_only":
            return (
                "Flow-only alignment failed; this session is not aligned.\n"
                + info.get("reason", "")
            )

        if status == "skipped":
            return "Rigid alignment retained; flow correction skipped.\n" + info.get(
                "reason", ""
            )

        if status == "manual_geometry_changed":
            return (
                "Stored flow disabled because the manual geometry changed. "
                "Apply will use rigid alignment only; Reset restores the flow."
            )

        return "No flow correction applied."

    @classmethod
    def identity(cls, dims) -> "Remapping":

        remap = cls(evaluate=False)
        remap.method = "identity"

        remap.dims = tuple(dims)
        remap.shift = np.zeros(2, dtype=float)
        remap.rotation = 0.0
        remap.matrix = np.eye(3, dtype=np.float64)
        remap.success = True

        return remap

    def test_transpose(
        self,
        reference: np.ndarray,
        template: np.ndarray,
        should_be_identical: bool = False,
    ):

        ## compare correlation between template and reference to check for possible transposition
        opts = {"shift": True, "mode": "cosine_weighted", "outside_weight": 0.5, "reference_threshold": 0.1}
        c_max, z_scored, shift = calculate_img_correlation(
            reference,template,**opts
        )
        c_max_T, z_scored_T, shift_T = calculate_img_correlation(
            reference,template.T,**opts
        )

        print(
            f"c_max: {c_max}, z_scored: {z_scored}, c_max_T: {c_max_T}, z_scored_T: {z_scored_T}, shift: {shift}, shift_T: {shift_T}"
        )
        if c_max is None or c_max_T is None:
            raise ValueError(
                "Correlation calculation failed. Check input arrays for NaN or Inf values."
            )

        shift_is_alright, shift_T_is_alright = True, True
        if should_be_identical:
            shift_is_alright = np.all(np.isclose(shift, 0.0, atol=1.0))
            shift_T_is_alright = np.all(np.isclose(shift_T, 0.0, atol=1.0))

        if c_max > c_max_T and c_max > self.c_min and shift_is_alright:
            self.transpose = False
        elif c_max_T > c_max and c_max_T > self.c_min and shift_T_is_alright:
            ## if transpose yields better results...
            self.transpose = True
            print("Template will be transposed.")
            # self.template = self.template.T
        else:
            ## if both fail, rather take the projection image
            raise ValueError(
                "Warning: Cn is not consistent with A, returning projection image instead of Cn."
            )

    def fix_transpose(self, A: MatrixT) -> MatrixT:

        if not self.transpose:
            return A
        # print("Fixing transpose of A to match reference Cn.")
        if isinstance(A, sparse.csc_matrix):
            # A = sparse.csc_matrix(
            A = sparse.hstack(
                [
                    sparse.csc_matrix(img.reshape(self.dims).transpose().reshape(-1, 1))
                    for img in A.transpose()
                ],
                format="csc",
            )
            # )
        elif isinstance(A, np.ndarray):
            A = A.T
        else:
            raise ValueError(
                "Input A must be either a sparse.csc_matrix or a np.ndarray"
            )
        return A

    @staticmethod
    def _rotation_matrix(dims, angle: float) -> np.ndarray:

        height, width = dims
        center = ((width - 1) / 2.0, (height - 1) / 2.0)

        affine = cv2.getRotationMatrix2D(center, angle, 1.0)

        matrix = np.eye(3, dtype=np.float64)
        matrix[:2, :] = affine

        return matrix

    @classmethod
    def _rigid_matrix(cls, dims, shift, rotation: float) -> np.ndarray:

        matrix = cls._rotation_matrix(dims, rotation)

        # CATAN shifts are [dy, dx].
        matrix[0, 2] += float(shift[1])
        matrix[1, 2] += float(shift[0])

        return matrix

    @classmethod
    def _rigid_parameters(cls, dims, matrix: np.ndarray) -> tuple[np.ndarray, float]:

        # OpenCV rotation matrix convention:
        #
        # [ cos(a)   sin(a) ]
        # [-sin(a)   cos(a) ]

        rotation = float(np.degrees(np.arctan2(matrix[0, 1], matrix[0, 0])))
        rotation_matrix = cls._rotation_matrix(dims, rotation)

        dx = matrix[0, 2] - rotation_matrix[0, 2]
        dy = matrix[1, 2] - rotation_matrix[1, 2]

        return (np.asarray([dy, dx], dtype=float), rotation)

    @staticmethod
    def _warp_dense(image: np.ndarray, matrix: np.ndarray) -> np.ndarray:

        height, width = image.shape[:2]

        return cv2.warpAffine(
            image,
            matrix[:2, :],
            (width, height),
            flags=cv2.INTER_CUBIC,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=0,
        )

    def _flow_source_coordinates(self, flow):
        """Original-image coordinates sampled by this matrix + flow."""
        yy, xx = np.indices(self.dims, dtype=np.float64)
        flow = np.asarray(flow, dtype=np.float64)

        mapped_x = xx + flow[..., 0]
        mapped_y = yy + flow[..., 1]
        inverse = np.linalg.inv(np.asarray(self.matrix, dtype=np.float64))

        source_x = inverse[0, 0] * mapped_x + inverse[0, 1] * mapped_y + inverse[0, 2]
        source_y = inverse[1, 0] * mapped_x + inverse[1, 1] * mapped_y + inverse[1, 2]
        return xx, yy, source_x, source_y

    def _fit_flow_rigid(self, flow, mask):
        """Fit source -> common-frame rotation and translation."""
        xx, yy, source_x, source_y = self._flow_source_coordinates(flow)

        mask = np.asarray(mask, dtype=bool)
        mask = mask & np.isfinite(source_x) & np.isfinite(source_y)

        if np.count_nonzero(mask) < 16:
            raise ValueError("Insufficient overlap to summarize rigid motion.")

        source = np.column_stack((source_x[mask], source_y[mask]))
        target = np.column_stack((xx[mask], yy[mask]))

        source_center = source.mean(axis=0)
        target_center = target.mean(axis=0)
        source_zero = source - source_center
        target_zero = target - target_center

        # Orthogonal Procrustes fit, excluding reflections and scaling.
        u, singular_values, vt = np.linalg.svd(source_zero.T @ target_zero)
        if singular_values[-1] <= 1e-12:
            raise ValueError("Overlap geometry cannot constrain rigid motion.")

        rotation = vt.T @ u.T
        if np.linalg.det(rotation) < 0:
            vt[-1, :] *= -1
            rotation = vt.T @ u.T

        translation = target_center - rotation @ source_center

        matrix = np.eye(3, dtype=np.float64)
        matrix[:2, :2] = rotation
        matrix[:2, 2] = translation

        error = source @ rotation.T + translation - target
        rms = float(np.sqrt(np.mean(np.sum(error**2, axis=1))))
        return matrix, rms

    def _factor_flow_rigid(self, matrix):
        """Change representation while preserving the full sampling map."""
        xx, yy, source_x, source_y = self._flow_source_coordinates(self.flow)
        matrix = np.asarray(matrix, dtype=np.float64)

        residual = np.empty(tuple(self.dims) + (2,), dtype=np.float32)
        residual[..., 0] = (
            matrix[0, 0] * source_x + matrix[0, 1] * source_y + matrix[0, 2] - xx
        )
        residual[..., 1] = (
            matrix[1, 0] * source_x + matrix[1, 1] * source_y + matrix[1, 2] - yy
        )

        self.matrix = matrix.copy()
        self.shift, self.rotation = self._rigid_parameters(self.dims, self.matrix)
        self.flow = residual

    def _flow_comparison_records(self, references):
        """Rigid summaries of independently accepted flow candidates."""
        records = {}

        for path, reference in references.items():
            candidate = self.flow_info.get("candidates", {}).get(str(path))
            if not candidate or not (
                candidate.get("accepted") or candidate.get("selected")
            ):
                continue

            global_matrix = candidate.get("rigid_matrix")
            if global_matrix is None:
                continue

            global_matrix = np.asarray(global_matrix, dtype=np.float64)
            reference_matrix = np.asarray(
                reference.get("matrix", np.eye(3)), dtype=np.float64
            )

            # Current original coordinates -> reference original coordinates,
            # considering only their rigid components.
            local_matrix = np.linalg.solve(reference_matrix, global_matrix)
            shift, rotation = self._rigid_parameters(self.dims, local_matrix)

            records[str(path)] = {
                "kind": "flow_rigid",
                "success": True,
                "reason": None,
                "shift": shift,
                "rotation": rotation,
                "matrix": local_matrix,
                "global_matrix": global_matrix.copy(),
                # This measures full-flow agreement, not rigid-only agreement.
                "c_max": candidate.get("correlation_after"),
                "c_zscored": None,
                "score_kind": "full_flow_correlation",
                "rigid_residual_rms": candidate.get("rigid_residual_rms"),
            }

        return records

    @staticmethod
    def _translation_prior(reference, reference_valid, template, limit, offsets=None):
        """(translation matrix in the common frame, peak z-score) maximizing
        the cosine similarity of the high-passed ``template`` with the
        high-passed aligned ``reference``, within ``limit`` px. Without
        high-pass filtering, overall brightness makes zero shift win."""

        def high_pass(image, valid):
            image = np.asarray(image, dtype=np.float32)
            valid = valid & np.isfinite(image)
            image = np.where(valid, image, image[valid].mean()).astype(np.float32)
            return (image - cv2.GaussianBlur(image, (0, 0), 10.0)).astype(np.float64)

        reference = high_pass(reference, np.asarray(reference_valid, dtype=bool))
        moving = high_pass(template, np.ones(np.shape(template), dtype=bool))

        scores = calculate_shift_score_map(
            reference, moving, mode="cosine", min_overlap=0.25
        )
        xs = np.arange(scores.shape[1]) - (moving.shape[1] - 1)
        ys = np.arange(scores.shape[0]) - (moving.shape[0] - 1)
        offsets = np.zeros((1, 2)) if offsets is None else np.asarray(offsets, dtype=float)
        near = np.zeros(scores.shape, dtype=bool)
        for ox, oy in offsets:
            near |= (xs[None, :] - ox) ** 2 + (ys[:, None] - oy) ** 2 <= float(limit) ** 2
        allowed = np.isfinite(scores) & near
        if not allowed.any():
            raise ValueError("No translation could be estimated.")

        values = scores[allowed]
        iy, ix = np.unravel_index(
            np.argmax(np.where(allowed, scores, -np.inf)), scores.shape
        )
        # Peak distinctness: robust z-score among all allowed shifts.
        median = np.median(values)
        spread = 1.4826 * np.median(np.abs(values - median))
        zscore = float((scores[iy, ix] - median) / spread) if spread > 0 else np.nan

        matrix = np.eye(3, dtype=np.float64)
        matrix[0, 2] = float(xs[ix])
        matrix[1, 2] = float(ys[iy])
        return matrix, zscore

    def _flow_deviation(self, path, record, preferred, max_displacement):
        """Text describing how a selected flow departs from the default
        (most recent reference, within limits, good agreement), or None."""
        candidates = self.flow_info["candidates"]
        notes = []

        # Choosing the best of the recent references (also an older one) is
        # automatic: only doubts about the selected alignment itself need
        # confirmation. Why an older one was used is kept for the message.
        newest = candidates[preferred]
        if path != preferred:
            why = (
                f"correlation {newest['correlation_after']:.2f}"
                if newest.get("correlation_after") is not None
                and not newest.get("exceeds_limit")
                else (newest.get("reason") or "rejected").split(". ")[0]
            )
            self.flow_info["reference_note"] = (
                f"Aligned to an older session ({record['name']}) instead of the "
                f"most recent one ({newest['name']}: {why})."
            )

        if record.get("exceeds_limit"):
            notes.append(
                f"Displacement {record['p95_displacement']:.0f} px exceeds the "
                f"maximum session shift ({max_displacement:.0f} px)."
            )

        good = self.flow_info.get("good_correlation")
        correlation = record["correlation_after"]
        if good is not None and correlation < good:
            agreement = record.get("agreement")
            if agreement is None or agreement["agreeing"] < AGREEMENT_MIN:
                compared = (
                    "no other references to compare"
                    if not agreement or not agreement["distances"]
                    else "other references disagree ("
                    + ", ".join(
                        f"{name}: {distance:.0f} px"
                        for name, distance in agreement["distances"]
                    )
                    + ")"
                )
                notes.append(
                    f"Low background correlation {correlation:.2f} (median of "
                    f"previous alignments {self.flow_info['median_correlation']:.2f}; "
                    f"low below {good:.2f}); {compared}."
                )

        return " ".join(notes) or None

    def _reference_agreement(self, path, record, usable):
        """How many of the most recent other usable references place the
        image centre within AGREEMENT_PX of the selected alignment."""
        selected = record.get("rigid_matrix")
        if selected is None:
            return None

        centre = np.array([(self.dims[1] - 1) / 2, (self.dims[0] - 1) / 2, 1.0])
        position = np.asarray(selected) @ centre

        distances = []
        for other_path, _, other in usable:
            if other_path == path or other.get("rigid_matrix") is None:
                continue
            other_position = np.asarray(other["rigid_matrix"]) @ centre
            distances.append(
                (other["name"], float(np.linalg.norm(other_position[:2] - position[:2])))
            )
            if len(distances) == AGREEMENT_REFERENCES:
                break

        return {
            "agreeing": sum(distance <= AGREEMENT_PX for _, distance in distances),
            "distances": distances,
        }

    def _score_neuron_agreement(self, record, flow, reference_data):
        """record["score"]: background correlation plus the fraction of
        neurons agreeing with the reference (mutual nearest centroids within
        NEURON_MATCH_PX); neurons carry the alignment where the background
        is dominated by vessels or diffuse signal."""
        record["score"] = record["correlation_after"]
        points = getattr(self, "moving_centroids", None)
        reference = reference_data.get("centroids")
        if points is None or reference is None or not len(reference):
            return

        from catan.core import footprint_alignment

        if self.transpose:
            points = points[:, ::-1]
        mapped = footprint_alignment.map_points(points, record["prior_matrix"], flow)
        i, _ = footprint_alignment.mutual_matches(
            mapped, np.asarray(reference, dtype=np.float64), NEURON_MATCH_PX
        )
        agreement = len(i) / max(1, min(len(points), len(reference)))
        record.update(
            neuron_agreement=float(agreement),
            neuron_matches=int(len(i)),
            score=record["correlation_after"] + agreement,
        )

    def reference_paths(self):
        """Paths of the sessions this alignment was derived from, or None
        if unknown (then everything earlier counts). Manual alignments
        record the session compared against on Apply ("references_used";
        none: placed freely in the common frame)."""
        info = self.flow_info or {}
        if "references_used" in info:
            return {str(path) for path in info["references_used"]}
        if getattr(self, "method", "automatic") == "manual":
            return set()
        paths = set()
        if info.get("reference") and self.flow is not None:
            paths.add(str(info["reference"]))
        if info.get("mode") != "flow_only":
            paths.update(
                str(path)
                for path, record in (self.remap_data or {}).items()
                if record.get("success", True)
            )
        return paths or None

    @property
    def alignment_correlation(self):
        """Background correlation of the selected flow alignment, or None."""
        info = self.flow_info or {}
        record = info.get("candidates", {}).get(info.get("reference"), {})
        value = record.get("correlation_after")
        return float(value) if value is not None and np.isfinite(value) else None

    def estimate_flow(
        self,
        template,
        references,
        *,
        min_overlap=0.30,
        min_gain=0.001,
        min_correlation=0.10,
        max_displacement=30.0,
        hard_max_displacement=50.0,
        residual=True,
        zero_mean=False,
    ):
        """Select one flow against already aligned references.

        ``zero_mean``: remove the mean displacement of the flow inside the
        interior overlap, so that it keeps the given rigid geometry (manual
        refinement of a chosen shift).

        References are supplied in session order. The FLOW_RECENT_REFERENCES
        most recent ones are evaluated, older ones only while none of these
        aligns well (accepted, correlation at least
        RELATIVE_FLOW_CORRELATION × the median "alignment_correlation" of the
        references). Candidates are scored by background correlation plus
        neuron agreement (_score_neuron_agreement); the most recent one
        within FLOW_PREFERENCE_MARGIN of the best score is used; a low
        correlation is then checked for agreement with other references
        (_reference_agreement). Without ``residual`` (flow-only alignment) a
        candidate beyond ``max_displacement`` (but within the hard limit)
        stays usable. Doubts about the selected alignment (displacement
        beyond the limit, low correlation not backed by other references)
        are described in ``flow_info["deviation"]``; using an older
        reference is not one (``flow_info["reference_note"]``).
        """
        self.flow = None
        self.flow_info = {
            "status": "skipped",
            "reason": "No suitable reference among the previous sessions.",
            "candidates": {},
            "deviation": None,
        }

        if not self.success:
            self.flow_info["reason"] = "Rigid alignment was unsuccessful."
            return

        yy, xx = np.indices(self.dims, dtype=np.float32)
        base_matrix = np.asarray(self.matrix, dtype=np.float64).copy()

        # Dataset-relative quality level from the references' own alignments.
        history = [
            float(entry["alignment_correlation"])
            for entry in references.values()
            if entry.get("alignment_correlation") is not None
            and np.isfinite(entry["alignment_correlation"])
        ]
        good_correlation = None
        if len(history) >= MIN_CORRELATION_HISTORY:
            median_correlation = float(np.median(history))
            good_correlation = RELATIVE_FLOW_CORRELATION * median_correlation
            self.flow_info.update(
                median_correlation=median_correlation,
                good_correlation=good_correlation,
            )

        def correlation(a, b, mask):
            x = np.asarray(a[mask], dtype=np.float64)
            y = np.asarray(b[mask], dtype=np.float64)
            x -= x.mean()
            y -= y.mean()
            denominator = np.linalg.norm(x) * np.linalg.norm(y)
            if denominator <= 0:
                return np.nan
            return float(np.dot(x, y) / denominator)

        # (path, flow, record) of candidates passing all checks except,
        # without residual, the displacement limit.
        usable = []
        preferred = None

        # References are supplied in session order.
        # Prefer the most recent eligible session, then work backward.
        def evaluate_start(matrix, reference, reference_valid, record):
            """Flow starting from ``matrix``; fills ``record``, returns the
            flow or raises ValueError."""
            self.matrix = np.asarray(matrix, dtype=np.float64).copy()
            moving = np.asarray(
                self.apply_remap(template, use_optical_flow=False),
                dtype=np.float32,
            )
            moving_valid = self.valid_mask(use_optical_flow=False)

            joint = moving_valid & reference_valid
            if joint.mean() < min_overlap:
                raise ValueError("Insufficient valid overlap.")

            # Equalize the images outside overlap, then fade the resulting
            # field toward zero at the overlap boundary.
            reference_input = reference.copy()
            reference_input[~joint] = moving[~joint]

            flow = calculate_residual_flow(reference_input, moving)

            if not np.isfinite(flow).all():
                raise ValueError("Estimated flow contains nonfinite values.")

            padded = np.pad(joint.astype(np.uint8), 1)
            distance = cv2.distanceTransform(padded, cv2.DIST_L2, 3)[1:-1, 1:-1]
            taper = np.clip(distance / 8.0, 0.0, 1.0)

            flow = np.asarray(flow, dtype=np.float32)

            # Regularize displacement, not image intensities.
            # Apply the same regularized field that we evaluate below.
            smoothing_sigma = 16.0  # spatial pixels
            flow = cv2.GaussianBlur(
                flow,
                (0, 0),
                sigmaX=smoothing_sigma,
                sigmaY=smoothing_sigma,
                borderType=cv2.BORDER_REFLECT_101,
            )

            record["flow_smoothing_sigma"] = smoothing_sigma

            if zero_mean:
                interior = joint & (distance >= 8.0)
                if np.any(interior):
                    mean = flow[interior].mean(axis=0)
                    flow = flow - mean
                    record["removed_mean_flow"] = tuple(map(float, mean))

            if residual:
                flow = flow * taper[..., None]

            # Measure typical displacement inside valid overlap,
            # beyond the 8 px boundary taper. Use a fixed mask so
            # large displacements cannot exclude themselves.
            displacement_mask = joint & (distance >= 8.0)

            if residual:
                magnitude = np.linalg.norm(flow, axis=-1)
            else:
                # Total displacement: starting translation + flow, relative
                # to the nearest allowed position (origin or offset).
                _, _, source_x, source_y = self._flow_source_coordinates(flow)
                shift_x, shift_y = xx - source_x, yy - source_y
                offsets = self._allowed_offsets()
                medians = [
                    np.median(np.hypot(shift_x - ox, shift_y - oy)[displacement_mask])
                    if np.any(displacement_mask)
                    else np.inf
                    for ox, oy in offsets
                ]
                ox, oy = offsets[int(np.argmin(medians))]
                magnitude = np.hypot(shift_x - ox, shift_y - oy)
                record["allowed_offset"] = (float(ox), float(oy))

            if np.count_nonzero(displacement_mask) < 16:
                raise ValueError(
                    "Insufficient interior overlap to assess displacement."
                )

            p95 = float(np.percentile(magnitude[displacement_mask], 95))
            interior_maximum = float(magnitude[displacement_mask].max())

            # Keep an absolute safeguard across the entire applied
            # field, including its tapered boundary.
            maximum = float(magnitude.max())

            record.update(
                p95_displacement=p95,
                interior_max_displacement=interior_maximum,
                max_displacement=maximum,
                p95_displacement_limit=float(max_displacement),
                hard_displacement_limit=float(hard_max_displacement),
                displacement_pixels=int(np.count_nonzero(displacement_mask)),
            )

            diagnostics = (
                f"Interior 95th percentile: {p95:.2f} px "
                f"(limit {max_displacement:.2f} px); "
                f"interior maximum: {interior_maximum:.2f} px; "
                f"whole-field maximum: {maximum:.2f} px "
                f"(hard limit {hard_max_displacement:.2f} px)."
            )

            if maximum > hard_max_displacement:
                raise ValueError(
                    "Residual displacement exceeds the absolute "
                    f"safeguard. {diagnostics}"
                )

            if p95 > max_displacement:
                reason = (
                    "Residual displacement exceeds the interior "
                    f"95th-percentile limit. {diagnostics}"
                )
                if residual:
                    raise ValueError(reason)
                # Flow-only: usable after confirmation.
                record["exceeds_limit"] = True
                record["reason"] = reason
            # Reject folding and extreme local expansion/compression,
            # including deformation introduced by tapering.
            du_dy, du_dx = np.gradient(flow[..., 0])
            dv_dy, dv_dx = np.gradient(flow[..., 1])
            determinant = (1.0 + du_dx) * (1.0 + dv_dy) - du_dy * dv_dx

            # Assess geometry inside valid overlap, excluding the
            # boundary where flow estimates are less constrained.
            geometry_mask = joint & (distance >= 8.0)
            jacobian = determinant[geometry_mask]

            if jacobian.size < 16:
                raise ValueError(
                    "Insufficient interior overlap to assess deformation."
                )

            if not np.isfinite(jacobian).all():
                raise ValueError(
                    "Deformation contains nonfinite spatial derivatives."
                )

            folded = jacobian <= 0.0
            extreme = (jacobian < 0.20) | (jacobian > 5.0)

            folded_fraction = float(folded.mean())
            extreme_fraction = float(extreme.mean())
            low, high = np.percentile(jacobian, [1, 99])

            record.update(
                jacobian_min=float(jacobian.min()),
                jacobian_max=float(jacobian.max()),
                jacobian_p01=float(low),
                jacobian_p99=float(high),
                folded_fraction=folded_fraction,
                extreme_deformation_fraction=extreme_fraction,
            )

            diagnostics = (
                f"Interior Jacobian: minimum {jacobian.min():.3f}, "
                f"1st–99th percentile {low:.3f}–{high:.3f}, "
                f"maximum {jacobian.max():.3f}. "
                f"Folded pixels: {100 * folded_fraction:.3f}%; "
                f"strongly compressed/expanded pixels: "
                f"{100 * extreme_fraction:.3f}%."
            )

            # Retain strict rejection of folding in the interior.
            # Allow isolated positive compression/expansion outliers.
            if folded.any() or extreme_fraction > 0.01:
                raise ValueError(
                    "Deformation failed the interior geometry check. " + diagnostics
                )

            sample_x = xx + flow[..., 0]
            sample_y = yy + flow[..., 1]

            destination_valid = (
                cv2.remap(
                    moving_valid.astype(np.float32),
                    sample_x,
                    sample_y,
                    cv2.INTER_LINEAR,
                    borderMode=cv2.BORDER_CONSTANT,
                    borderValue=0,
                )
                > 0.999
            )

            evaluate = joint & destination_valid & (distance >= 4)
            if evaluate.mean() < min_overlap or (
                residual
                and np.count_nonzero(evaluate) < 0.90 * np.count_nonzero(joint)
            ):
                raise ValueError("Too much overlap is lost by the deformation.")

            warped = cv2.remap(
                moving,
                sample_x,
                sample_y,
                cv2.INTER_LINEAR,
                borderMode=cv2.BORDER_CONSTANT,
                borderValue=0,
            )

            # Identical evaluation pixels for before/after comparison.
            before = correlation(reference, moving, evaluate)
            after = correlation(reference, warped, evaluate)
            gain = after - before

            record.update(
                correlation_before=before,
                correlation_after=after,
                gain=gain,
                overlap=float(evaluate.mean()),
            )

            if not np.isfinite(before) or not np.isfinite(after):
                raise ValueError("Insufficient background intensity variation.")
            if after < min_correlation:
                raise ValueError("Background agreement remains too weak.")
            # Without residual, the gain is relative to the starting
            # translation and may legitimately be tiny.
            if residual and gain < min_gain:
                raise ValueError(
                    "Flow does not improve background agreement enough."
                )

            if not residual:
                rigid_matrix, rigid_rms = self._fit_flow_rigid(
                    flow,
                    evaluate & (distance >= 8.0),
                )
                record["rigid_matrix"] = rigid_matrix
                record["rigid_residual_rms"] = rigid_rms

            return flow

        def well_aligned(record):
            return record["accepted"] and (
                good_correlation is None
                or record["correlation_after"] >= good_correlation
            )

        evaluated = 0
        for path, reference_data in reversed(list(references.items())):
            if not reference_data.get("flow_candidate", False):
                continue
            # Beyond the most recent references only while none of them
            # aligns well.
            if evaluated >= FLOW_RECENT_REFERENCES and any(
                well_aligned(record) for _, _, record in usable
            ):
                break
            evaluated += 1

            record = {
                "accepted": False,
                "name": reference_data.get("name") or Path(str(path)).name,
            }
            self.flow_info["candidates"][str(path)] = record
            if preferred is None:
                preferred = str(path)

            try:
                reference = np.asarray(
                    reference_data["aligned_template"], dtype=np.float32
                )
                reference_valid = np.asarray(reference_data["valid_mask"], dtype=bool)
                if reference.shape != tuple(
                    self.dims
                ) or reference_valid.shape != tuple(self.dims):
                    raise ValueError("Reference dimensions differ.")

                if not np.isfinite(reference).all():
                    raise ValueError("Reference contains nonfinite values.")
            except ValueError as exc:
                record["reason"] = str(exc)
                continue

            starts = [("current", base_matrix)]
            if not residual:
                # Flow alone converges to small, wrong shifts when the field
                # of view moved far (845ad S53: 57 px); a start at the best
                # translation finds them. Both starts are evaluated, the
                # better agreement wins.
                try:
                    prior, record["prior_zscore"] = self._translation_prior(
                        reference,
                        reference_valid,
                        template,
                        hard_max_displacement,
                        offsets=self._allowed_offsets(),
                    )
                    if np.abs(prior[:2, 2] - base_matrix[:2, 2]).max() > 1.0:
                        starts.append(("translation", prior))
                except ValueError:
                    pass

            results, failures = [], []
            for start, matrix in starts:
                trial = {}
                try:
                    flow = evaluate_start(matrix, reference, reference_valid, trial)
                    results.append((trial["correlation_after"], start, matrix, flow, trial))
                except (ValueError, cv2.error) as exc:
                    failures.append(str(exc))

            if not results:
                record["reason"] = " ".join(sorted(set(failures)))
                continue

            after, start, matrix, flow, trial = max(results, key=lambda item: item[0])
            record.update(trial)
            record["start"] = start
            record["prior_matrix"] = np.asarray(matrix, dtype=np.float64).copy()

            record["accepted"] = not record.get("exceeds_limit", False)
            self._score_neuron_agreement(record, flow, reference_data)
            usable.append((str(path), flow, record))

        if usable:
            best = max(record["score"] for _, _, record in usable)
            path, flow, record = next(
                item
                for item in usable
                if item[2]["score"] >= best - FLOW_PREFERENCE_MARGIN
            )
            record["selected"] = True
            if (
                good_correlation is not None
                and record["correlation_after"] < good_correlation
            ):
                record["agreement"] = self._reference_agreement(path, record, usable)

            self.matrix = record.get("prior_matrix", base_matrix).copy()
            self.flow = flow
            self.flow_info.update(
                status="applied",
                reference=path,
                reason="",
                deviation=self._flow_deviation(
                    path, record, preferred, max_displacement
                ),
            )
        elif self.flow_info["candidates"]:
            self.matrix = base_matrix
            self.flow_info["reason"] = summarize_rejections(
                self.flow_info["candidates"]
            )

    @property
    def report(self) -> AlignmentReport:

        entries = list(self.remap_data.values())
        if self.flow_info.get("mode") == "flow_only":
            entries = [
                {
                    "success": candidate.get("selected", False),
                    "c_max": candidate.get("correlation_after"),
                    "c_zscored": None,
                    "reason": candidate.get("reason"),
                }
                for candidate in self.flow_info.get("candidates", {}).values()
            ]

        successful = [item for item in entries if item.get("success", False)]
        source = successful if successful else entries

        correlations = [
            item["c_max"]
            for item in source
            if (item.get("c_max") is not None and np.isfinite(item["c_max"]))
        ]

        zscores = [
            item["c_zscored"]
            for item in source
            if (item.get("c_zscored") is not None and np.isfinite(item["c_zscored"]))
        ]

        correlation = float(np.median(correlations)) if correlations else None

        zscore = float(np.median(zscores)) if zscores else None

        if self.success:
            reason = None
        elif getattr(self, "confirmation", None):
            reason = "confirmation_required"
        elif not entries:
            reason = "no_reference"
        else:

            reasons = [item.get("reason") for item in entries if item.get("reason")]
            reason = (
                reasons[0] if len(set(reasons)) == 1 else "no_reference_passed_quality"
            )

        return AlignmentReport(
            success=bool(self.success),
            reason=reason,
            shift=(None if self.shift is None else self.shift.copy()),
            rotation=self.rotation,
            total_shift=(
                float(np.linalg.norm(self.shift)) if self.shift is not None else np.nan
            ),
            correlation=correlation,
            correlation_zscore=zscore,
            n_references=len(entries),
            n_successful_references=len(successful),
            remap_data=self.remap_data,
            method=self.method,
        )

    def save(
        self, path: str | Path, *, mat_version: Literal["pre73", "7.3"] = "7.3"
    ) -> None:
        """Save one or several sessions as a CATAN-native session container.

        If ``fields_to_save`` is omitted, the packaged ``catan_session.json``
        structure is used.
        """

        fields_to_save = LoadConfig.fields_from_resource(
            NATIVE_REMAP_CONFIG, enabled_only=False
        )

        save_file(
            path,
            self,
            fields_to_save,
            mat_version=mat_version,
            root_attributes={"object_type": "Remapping", "format_version": 1},
            root="/",
        )
