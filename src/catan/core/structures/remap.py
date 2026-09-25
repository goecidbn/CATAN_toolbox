from typing import Any, Literal, Optional, TypeVar
from dataclasses import dataclass
from pathlib import Path
import h5py
import h5py
import numpy as np
from scipy import sparse
import cv2

from catan.core.image_correlation import calculate_img_correlation
from catan.core.alignment import (
    get_session_remap,
    _build_remap,
    _shift_sparse_bilinear,
)
from catan.core.io import load_file, save_file, NATIVE_REMAP_CONFIG
from catan.core.structures.load_config import LoadConfig, FieldSpec

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
    ):

        self.shift: np.ndarray | None = None
        self.rotation: float | None = None

        # Transform from THIS SESSION'S original template
        # into CATAN's common/global coordinate system.
        self.matrix = np.eye(3, dtype=np.float64)

        self.flow = None
        self.transpose = False

        # path -> pairwise alignment information
        self.remap_data: dict[str, dict[str, Any]] = {}

        self.max_shift = float(max_shift)
        self.max_rotation = float(max_rotation)
        self.c_min = float(min_corr)
        self.min_zcorr = float(min_zcorr)

        self.rotation_step = float(rotation_step)

        self.rotation_refine_step = float(rotation_refine_step)

        self.success = False
        self.dims = None

        if evaluate and template is not None and references:
            self.evaluate(template, references, use_optical_flow=use_optical_flow)

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

        global_candidates = []

        # Optical flow cannot sensibly be aggregated across
        # several independently transformed references.
        use_flow = use_optical_flow and len(references) == 1

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
                reference, template, use_optical_flow=use_flow
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

        # Keep optical flow only for the unambiguous
        # single-reference case.
        if use_flow and len(global_candidates) == 1:

            successful = [item for item in self.remap_data.values() if item["success"]]

            self.flow = successful[0].get("flow")

        else:
            self.flow = None

        self.success = True

    # def evaluate(
    #     self,
    #     template: np.ndarray,
    #     template_reference: np.ndarray,
    #     use_optical_flow: bool = True,
    # ):
    #     self.dims = template.shape
    #     print("calculate / apply rotation as well?!")

    #     if len(template_reference.shape) > len(template.shape):
    #         ## stacked templates - use average remap
    #         assert template_reference.shape[1:] == template.shape
    #         nT = template_reference.shape[0]
    #         if template_reference.shape[0] > 1:
    #             use_optical_flow = False
    #     else:
    #         nT = 1
    #         template_reference = template_reference[np.newaxis, ...]

    #     assert nT > 0, "No templates provided for remapping."

    #     self.reference = template_reference
    #     t = 1
    #     while t <= nT:
    #         if np.all(np.isnan(template_reference[-t, ...])):
    #             print(f"Template {-t} is empty.")
    #             t += 1
    #         else:
    #             break

    #         assert t <= nT, f"Alignment stopped, all provided templates are empty."
    #     # print(f"Using template {-t} for alignment.")

    #     self.test_transpose(template_reference[-t, ...], template)

    #     self.c_max = np.full(nT, np.nan)
    #     self.c_zscored = np.full(nT, np.nan)

    #     shifts = np.full((nT, 2), np.nan)
    #     for t in range(nT):

    #         if np.all(np.isnan(template_reference[t, ...])):
    #             print(f"Template {t} is empty; skipping alignment.")
    #             continue

    #         # flow = None
    #         # if use_optical_flow:
    #         shift, flow, self.c_max[t], self.c_zscored[t] = get_session_remap(
    #             template_reference[t, ...],
    #             template,
    #             self.dims,
    #             None,
    #             use_optical_flow=use_optical_flow,
    #         )
    #         # else:
    #         #     self.c_max[t], self.c_zscored[t], shift = calculate_img_correlation(
    #         #         template_reference[t, ...], template, mode="correlation"
    #         #     )
    #         #     print("shift without flow:", shift)

    #         if shift is None:
    #             print(
    #                 f"Alignment with Session {t+1} failed: correlation calculation failed."
    #             )
    #             continue
    #         total_shift = np.sqrt(np.square(shift).sum())
    #         # print(
    #         #     f"with Session {t+1}: {self.c_max[t]=}, {self.c_zscored[t]=}, {shift=}"
    #         # )
    #         # if use_optical_flow:
    #         self.flow = flow

    #         if (
    #             self.c_max[t] > self.c_min
    #             and self.c_zscored[t] > self.min_zcorr
    #             and total_shift < self.max_shift
    #         ):
    #             shifts[t, :] = shift
    #         else:
    #             print(
    #                 f"Alignment with Session {t+1} failed: c_max={self.c_max[t]}, c_zscored={self.c_zscored[t]}, shift={shift}"
    #             )

    #     # print(f"Shifts from all templates: {shifts}")
    #     # print(f"correlation max from all templates: {self.c_max}")
    #     self.shift = np.nanmedian(shifts, axis=0)
    #     # print(f"Median shift from all templates: {self.shift}")
    #     self.success = np.isfinite(self.total_shift)

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
        elif total_shift > self.max_shift:
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

    def apply_remap(self, A: MatrixT, use_optical_flow: bool = False) -> MatrixT:

        if self.transpose:
            A = self.fix_transpose(A)

        if not self.success:
            return A

        rotation = 0.0 if self.rotation is None else self.rotation
        shift = np.zeros(2) if self.shift is None else self.shift
        use_rigid = np.linalg.norm(shift) > 1.0 or abs(rotation) > 0.05

        # Fast path for identity.
        if not use_rigid and (not use_optical_flow or self.flow is None):
            return A

        # Translation-only sparse fast path.
        if abs(rotation) <= 0.05 and self.flow is None and sparse.issparse(A):
            return _shift_sparse_bilinear(A, self.dims, *shift, order="C")

        matrix = self.matrix

        def warp(image):

            image = self._warp_dense(image, matrix)

            if use_optical_flow and self.flow is not None:

                zero_shift = np.zeros(2, dtype=float)
                y_remap, x_remap = _build_remap(self.dims, zero_shift, self.flow)
                image = cv2.remap(image, x_remap, y_remap, cv2.INTER_CUBIC)

            return image

        if sparse.issparse(A):

            return sparse.hstack(
                [
                    sparse.csc_matrix(
                        warp(column.toarray().reshape(self.dims)).reshape(-1, 1)
                    )
                    for column in A.T
                ],
                format="csc",
            )

        if isinstance(A, np.ndarray):
            return warp(A)

        raise ValueError("Input A must be either a sparse matrix " "or numpy array.")

    # def apply_remap(
    #     self,
    #     A: MatrixT,
    #     use_optical_flow: bool = True,
    # ) -> MatrixT:
    #     """
    #     general function to apply remapping to either a sparse matrix or a dense array
    #     """

    #     if self.transpose:
    #         A = self.fix_transpose(A)

    #     if self.shift is None and self.flow is None or not self.success:
    #         return A

    #     ## check, if shift is significant enough to apply remapping
    #     use_shift = self.total_shift > 1.0
    #     if use_optical_flow and self.flow is not None:
    #         flow_prctl = np.percentile(self.flow, [5, 95], axis=(0, 1))
    #         # print(f"Flow percentiles: {flow_prctl}")
    #         use_optical_flow = bool(np.any(np.abs(flow_prctl) > 0.5))

    #     if not (use_shift or use_optical_flow):
    #         # print("No significant shift/flow detected; skipping alignment.")
    #         return A

    #     elif (
    #         self.shift is not None
    #         and self.flow is None
    #         and isinstance(A, sparse.spmatrix)
    #     ):
    #         # print("aligning sparse matrix with shift only", self.shift)
    #         return _shift_sparse_bilinear(A, self.dims, *self.shift, order="C")
    #         # return _shift_sparse_bilinear(A, dy=0, dx=0)

    #     y_remap, x_remap = _build_remap(self.dims, self.shift, self.flow)

    #     if isinstance(A, sparse.csc_matrix):
    #         # print("aligning sparse matrix with shift and flow", self.shift)
    #         return sparse.csc_matrix(
    #             sparse.hstack(
    #                 [
    #                     sparse.csc_matrix(
    #                         cv2.remap(
    #                             a.reshape(self.dims),
    #                             x_remap,
    #                             y_remap,
    #                             cv2.INTER_CUBIC,
    #                         ).reshape(-1, 1)
    #                     )
    #                     for a in A.toarray().T
    #                 ],
    #                 format="csc",
    #             )
    #         )
    #     elif isinstance(A, np.ndarray):
    #         return cv2.remap(A, x_remap, y_remap, cv2.INTER_CUBIC)
    #     else:
    #         raise ValueError(
    #             "Input A must be either a sparse.csc_matrix or a np.ndarray"
    #         )

    @classmethod
    def identity(cls, dims) -> "Remapping":

        remap = cls(evaluate=False)

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
        _, cmax, shift = calculate_img_correlation(
            reference, template, mode="correlation"
        )
        _, cmax_T, shift_T = calculate_img_correlation(
            reference, template.T, mode="correlation"
        )

        # print(f"cmax: {cmax}, cmax_T: {cmax_T}, shift: {shift}, shift_T: {shift_T}")
        if cmax is None or cmax_T is None or shift is None or shift_T is None:
            raise ValueError(
                "Correlation calculation failed. Check input arrays for NaN or Inf values."
            )

        shift_is_alright, shift_T_is_alright = True, True
        if should_be_identical:
            shift_is_alright = np.all(np.isclose(shift, 0.0, atol=1.0))
            shift_T_is_alright = np.all(np.isclose(shift_T, 0.0, atol=1.0))

        if cmax > cmax_T and cmax > self.c_min and shift_is_alright:
            self.transpose = False
        elif cmax_T > cmax and cmax_T > self.c_min and shift_T_is_alright:
            ## if transpose yields better results...
            self.transpose = True
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

    @property
    def report(self) -> AlignmentReport:

        entries = list(self.remap_data.values())
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
