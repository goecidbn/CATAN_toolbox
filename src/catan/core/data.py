# import h5py
from typing import List, Optional, Tuple, TypeVar
from scipy import sparse
import numpy as np

from .utils import normalize_sparse_array


def center_of_mass(
    A, d1, d2, d3=None, d1_offset=0, d2_offset=0, d3_offset=0, convert=1.0
) -> np.ndarray:
    """
    calculate center of mass of footprints A in 2D or 3D

    TODO:
      - implement offset in each dimension
    """

    if "csc_matrix" not in str(type(A)):
        A = sparse.csc_matrix(A)

    if d3 is None:
        Coor = np.matrix(
            [
                np.outer(np.ones(d2), np.arange(d1)).ravel(),
                np.outer(np.arange(d2), np.ones(d1)).ravel(),
            ],
            dtype=A.dtype,
        )
    else:
        Coor = np.matrix(
            [
                np.outer(
                    np.ones(d3), np.outer(np.ones(d2), np.arange(d1)).ravel()
                ).ravel(),
                np.outer(
                    np.ones(d3), np.outer(np.arange(d2), np.ones(d1)).ravel()
                ).ravel(),
                np.outer(
                    np.arange(d3), np.outer(np.ones(d2), np.ones(d1)).ravel()
                ).ravel(),
            ],
            dtype=A.dtype,
        )

    Anorm = normalize_sparse_array(
        A, relative_threshold=0.001, minimum_nonzero_entries=0
    )

    cm = (Coor * Anorm).T
    cm[np.squeeze(np.array((Anorm > 0).sum(0))) == 0, :] = np.nan

    return np.array(cm) * convert


# @dataclass
# class remap_data:
#     shift: Optional[np.ndarray] = None
#     c_max: Optional[np.ndarray] = None
#     c_zscored: Optional[np.ndarray] = None
#     flow: Optional[np.ndarray] = None

#     @property
#     def total_shift(self):
#         if self.shift is not None:
#             return np.sqrt(np.square(self.shift).sum())
#         else:
#             return np.nan

#     @property
#     def is_valid(self):
#         return (
#             self.c_max is not None
#             and self.c_zscored is not None
#             and self.shift is not None
#         )


# def align_footprints_to_reference(
#     A: sparse.csc_matrix,
#     template: np.ndarray,
#     template_reference: np.ndarray,
#     use_optical_flow: bool = True,
#     params: dict[str, float] = {},
# ) -> Tuple[sparse.csc_matrix, np.ndarray, remap_data]:

#     min_corr = params.get("min_session_correlation", 0.2)
#     min_zcorr = params.get("min_session_correlation_zscore", 3.0)
#     max_shift = params.get("max_session_shift", 50.0)

#     if "csc_matrix" not in str(type(A)):
#         A = sparse.csc_matrix(A)
#     dims = template.shape

#     # remap = remap_data()

#     # # calculate shift (and flow)
#     # if len(template_reference.shape) > len(template.shape):
#     #     ## stacked templates - use average remap
#     #     assert template_reference.shape[1:] == template.shape
#     #     nT = template_reference.shape[0]
#     #     use_optical_flow = False
#     # else:
#     #     nT = 1
#     #     template_reference = template_reference[np.newaxis, ...]

#     # if nT == 0:
#     #     print(f"Alignment stopped, there are is no reference template provided.")
#     #     return A, template, remap

#     # t = 1
#     # while t <= nT:
#     #     if np.all(np.isnan(template_reference[-t, ...])):
#     #         print(f"Template {-t} is empty.")
#     #         t += 1
#     #     else:
#     #         break

#     #     if t > nT:
#     #         print(f"Alignment stopped, all provided templates are empty.")
#     #         return A, template, remap
#     # # print(f"Using template {-t} for alignment.")

#     remap.c_max = np.full(nT, np.nan)
#     remap.c_zscored = np.full(nT, np.nan)
#     ### --------------------------------------------------------------
#     ### ---------------------- check if flipped ----------------------
#     ### --------------------------------------------------------------
#     # _, _, c_max, c_zscored = get_shift_and_flow(
#     #     template_reference[-t, ...], template, dims, None
#     # )
#     # _, _, c_max_T, c_zscored_T = get_shift_and_flow(
#     #     template_reference[-t, ...], template.T, dims, None
#     # )
#     # print(f"Transposed correlation: c_max_T={c_max_T}, vs c_max={remap['c_max']}")

#     # if (c_max_T > c_max) & (c_max_T > min_corr):
#     #     print("Transposed image")
#     #     A = sparse.csc_matrix(
#     #         sparse.hstack(
#     #             [
#     #                 sparse.csc_matrix(img.reshape(dims).transpose().reshape(-1, 1))
#     #                 for img in A.transpose()
#     #             ],
#     #             format="csc",
#     #         )
#     #     )
#     #     template = template.T

#     # alignment_success = False
#     # shifts = np.full((nT, 2), np.nan)
#     # for t in range(nT):

#     #     if np.all(np.isnan(template_reference[t, ...])):
#     #         print(f"Template {t} is empty; skipping alignment.")
#     #         continue

#     #     shift, flow, remap.c_max[t], remap.c_zscored[t] = get_shift_and_flow(
#     #         template_reference[t, ...], template, dims, None
#     #     )
#     #     total_shift = np.sqrt(np.square(shift).sum())
#     #     # print(
#     #     #     f"with Session {t+1}: {remap.c_max[t]=}, {remap.c_zscored[t]=}, {shift=}"
#     #     # )
#     #     if use_optical_flow:
#     #         remap.flow = flow

#     #     if (
#     #         remap.c_max[t] > min_corr
#     #         and remap.c_zscored[t] > min_zcorr
#     #         and total_shift < max_shift
#     #     ):
#     #         shifts[t, :] = shift
#     #         alignment_success = True
#     #     else:
#     #         print(
#     #             f"Alignment with Session {t+1} failed: c_max={remap.c_max[t]}, c_zscored={remap.c_zscored[t]}, shift={shift}"
#     #         )

#     # remap.shift = np.nanmedian(shifts, axis=0)

#     # if not alignment_success:
#     #     print("No valid alignment found based on correlation criteria.")
#     #     print(remap.c_max, remap.c_zscored, remap.shift)
#     #     return A, template, remap

#     ### --------------------------------------------------------------
#     ### ----------------- apply shift (and flow) ---------------------
#     ### --------------------------------------------------------------
#     # do_align = remap.total_shift > 1.0
#     # if use_optical_flow and remap.flow is not None:
#     #     flow_prctl = np.percentile(remap.flow, [5, 95], axis=(0, 1))
#     #     do_align &= np.any(np.abs(flow_prctl) > 0.5)
#     # # do_align &= (use_optical_flow)
#     # if do_align:
#     #     A = apply_remap(A, dims, remap.shift, remap.flow if use_optical_flow else None)
#     #     template = apply_remap(
#     #         template, dims, remap.shift, remap.flow if use_optical_flow else None
#     #     )
#     # else:
#     #     print("No significant shift/flow detected; skipping alignment.")

#     # return A, template, remap
#     ## normalization?
#     ## possibly removing too small footprints?


# def _full_shift_grids(shape1, shape2):
#     """
#     Return dy, dx grids corresponding to a full convolution/correlation map.
#     """
#     h1, w1 = shape1
#     h2, w2 = shape2
#     dy = np.arange(h1 + h2 - 1) - (h2 - 1)
#     dx = np.arange(w1 + w2 - 1) - (w2 - 1)
#     DY, DX = np.meshgrid(dy, dx, indexing="ij")
#     return DY, DX


# def _shift_mask(shape1, shape2, expected_shift=None, max_shift_radius=None):
#     """
#     Boolean mask on the full correlation map selecting allowed shifts.
#     """
#     DY, DX = _full_shift_grids(shape1, shape2)
#     mask = np.ones_like(DY, dtype=bool)

#     if expected_shift is not None and max_shift_radius is not None:
#         ey, ex = expected_shift
#         mask &= (DY - ey) ** 2 + (DX - ex) ** 2 <= max_shift_radius**2

#     return mask


def _best_from_score_map(score_map, shape2, allowed_mask=None):
    """
    Return best shift (dy, dx), best score, and argmax index from a full score map.
    """
    score = np.array(score_map, copy=True)

    if allowed_mask is not None:
        # score[~allowed_mask] = -np.inf
        score[~allowed_mask] = np.nan

    # Calculate weighted shift using exponential weighting
    score_exp = np.exp(score - np.nanmax(score))
    weights = score_exp / np.nansum(score_exp)

    iy, ix = np.unravel_index(np.nanargmax(score), score.shape)
    dy = iy - (shape2[0] - 1)
    dx = ix - (shape2[1] - 1)

    score_max = float(score_map[iy, ix])
    score_zscored = float((score_max - np.nanmedian(score_map)) / np.nanstd(score_map))
    shift = (int(dy), int(dx))

    return score_max, score_zscored, shift


# def pad_axis(a, axis: int = 0, value: float | int = 0):
#     shape = list(a.shape)
#     shape[axis] = 1
#     pad = np.full(shape, value, dtype=a.dtype)
#     return np.concatenate((a, pad), axis=axis)
