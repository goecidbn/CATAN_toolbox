from typing import Optional, Tuple, TypeVar
from dataclasses import dataclass

import numpy as np
from scipy import sparse
import cv2

from catan.core.image_correlation import calculate_img_correlation
from catan.core.alignment import (
    get_session_remap,
    _build_remap,
    _shift_sparse_bilinear,
)

MatrixT = TypeVar("MatrixT", sparse.csc_matrix, np.ndarray)


@dataclass
class Remapping:

    shift: Optional[np.ndarray] = None
    c_max: Optional[np.ndarray] = None
    c_zscored: Optional[np.ndarray] = None
    flow: Optional[np.ndarray] = None
    transpose: bool = False

    max_shift: float = 50.0
    min_zcorr: float = 3.0
    c_min: float = 0.1

    @property
    def total_shift(self):
        if self.shift is not None:
            return np.sqrt(np.square(self.shift).sum())
        else:
            return np.nan

    @property
    def is_valid(self):
        return self.c_zscored is not None and self.shift is not None

    def __init__(
        self,
        A: np.ndarray,
        reference: np.ndarray,
        use_optical_flow: bool = True,
        evaluate: bool = True,
    ):
        # self.dims = reference.shape[-2:]

        self.success = False
        if evaluate:
            self.evaluate(A, reference, use_optical_flow=use_optical_flow)

    def evaluate(
        self,
        template: np.ndarray,
        template_reference: np.ndarray,
        use_optical_flow: bool = True,
    ):
        self.dims = template.shape

        if len(template_reference.shape) > len(template.shape):
            ## stacked templates - use average remap
            assert template_reference.shape[1:] == template.shape
            nT = template_reference.shape[0]
            if template_reference.shape[0] > 1:
                use_optical_flow = False
        else:
            nT = 1
            template_reference = template_reference[np.newaxis, ...]

        assert nT > 0, "No templates provided for remapping."

        self.reference = template_reference
        t = 1
        while t <= nT:
            if np.all(np.isnan(template_reference[-t, ...])):
                print(f"Template {-t} is empty.")
                t += 1
            else:
                break

            assert t <= nT, f"Alignment stopped, all provided templates are empty."
        # print(f"Using template {-t} for alignment.")

        self.test_transpose(template_reference[-t, ...], template)

        self.c_max = np.full(nT, np.nan)
        self.c_zscored = np.full(nT, np.nan)

        shifts = np.full((nT, 2), np.nan)
        for t in range(nT):

            if np.all(np.isnan(template_reference[t, ...])):
                print(f"Template {t} is empty; skipping alignment.")
                continue

            # flow = None
            # if use_optical_flow:
            shift, flow, self.c_max[t], self.c_zscored[t] = get_session_remap(
                template_reference[t, ...],
                template,
                self.dims,
                None,
                use_optical_flow=use_optical_flow,
            )
            # else:
            #     self.c_max[t], self.c_zscored[t], shift = calculate_img_correlation(
            #         template_reference[t, ...], template, mode="correlation"
            #     )
            #     print("shift without flow:", shift)

            if shift is None:
                print(
                    f"Alignment with Session {t+1} failed: correlation calculation failed."
                )
                continue
            total_shift = np.sqrt(np.square(shift).sum())
            # print(
            #     f"with Session {t+1}: {self.c_max[t]=}, {self.c_zscored[t]=}, {shift=}"
            # )
            # if use_optical_flow:
            self.flow = flow

            if (
                self.c_max[t] > self.c_min
                and self.c_zscored[t] > self.min_zcorr
                and total_shift < self.max_shift
            ):
                shifts[t, :] = shift
            else:
                print(
                    f"Alignment with Session {t+1} failed: c_max={self.c_max[t]}, c_zscored={self.c_zscored[t]}, shift={shift}"
                )

        # print(f"Shifts from all templates: {shifts}")
        # print(f"correlation max from all templates: {self.c_max}")
        self.shift = np.nanmedian(shifts, axis=0)
        # print(f"Median shift from all templates: {self.shift}")
        self.success = np.isfinite(self.total_shift)

    def apply_remap(
        self,
        A: MatrixT,
        use_optical_flow: bool = True,
    ) -> MatrixT:
        """
        general function to apply remapping to either a sparse matrix or a dense array
        """

        if self.transpose:
            A = self.fix_transpose(A)

        if self.shift is None and self.flow is None or not self.success:
            return A

        ## check, if shift is significant enough to apply remapping
        use_shift = self.total_shift > 1.0
        if use_optical_flow and self.flow is not None:
            flow_prctl = np.percentile(self.flow, [5, 95], axis=(0, 1))
            # print(f"Flow percentiles: {flow_prctl}")
            use_optical_flow = bool(np.any(np.abs(flow_prctl) > 0.5))

        if not (use_shift or use_optical_flow):
            # print("No significant shift/flow detected; skipping alignment.")
            return A

        elif (
            self.shift is not None
            and self.flow is None
            and isinstance(A, sparse.spmatrix)
        ):
            # print("aligning sparse matrix with shift only", self.shift)
            return _shift_sparse_bilinear(A, self.dims, *self.shift, order="C")
            # return _shift_sparse_bilinear(A, dy=0, dx=0)

        y_remap, x_remap = _build_remap(self.dims, self.shift, self.flow)

        if isinstance(A, sparse.csc_matrix):
            # print("aligning sparse matrix with shift and flow", self.shift)
            return sparse.csc_matrix(
                sparse.hstack(
                    [
                        sparse.csc_matrix(
                            cv2.remap(
                                a.reshape(self.dims),
                                x_remap,
                                y_remap,
                                cv2.INTER_CUBIC,
                            ).reshape(-1, 1)
                        )
                        for a in A.toarray().T
                    ],
                    format="csc",
                )
            )
        elif isinstance(A, np.ndarray):
            return cv2.remap(A, x_remap, y_remap, cv2.INTER_CUBIC)
        else:
            raise ValueError(
                "Input A must be either a sparse.csc_matrix or a np.ndarray"
            )

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
