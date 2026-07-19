# import numpy as np

# # from scipy import sparse
# from scipy.signal import fftconvolve

# from ...core.data import crop_to_common_bbox, _best_from_score_map


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


# def _embed_shifted(A1, A2, dy, dx):
#     """
#     Put A1 and shifted A2 into a common canvas and return both embedded arrays.

#     Shift convention:
#         positive dy -> A2 moves downward relative to A1
#         positive dx -> A2 moves rightward relative to A1
#     """
#     A1 = np.asarray(A1)
#     A2 = np.asarray(A2)

#     h1, w1 = A1.shape
#     h2, w2 = A2.shape

#     top1 = max(0, -dy)
#     left1 = max(0, -dx)
#     top2 = max(0, dy)
#     left2 = max(0, dx)

#     H = max(top1 + h1, top2 + h2)
#     W = max(left1 + w1, left2 + w2)

#     E1 = np.zeros((H, W), dtype=float)
#     E2 = np.zeros((H, W), dtype=float)

#     E1[top1 : top1 + h1, left1 : left1 + w1] = A1
#     E2[top2 : top2 + h2, left2 : left2 + w2] = A2

#     return E1, E2


# def _centroid_from_array(A, threshold=0.0):
#     """
#     Weighted centroid using values above threshold.
#     Returns (cy, cx). If empty, returns (nan, nan).
#     """
#     A = np.asarray(A, dtype=float)
#     M = A > threshold
#     if not np.any(M):
#         return np.nan, np.nan

#     Y, X = np.indices(A.shape)
#     W = A * M
#     s = W.sum()
#     if s <= 0:
#         return np.nan, np.nan

#     cy = (Y * W).sum() / s
#     cx = (X * W).sum() / s
#     return float(cy), float(cx)


# # def _cosine_union(E1, E2, thr1=0.0, thr2=0.0, eps=1e-12):
# #     M = (E1 > thr1) | (E2 > thr2)
# #     x = E1[M]
# #     y = E2[M]
# #     nx = np.linalg.norm(x)
# #     ny = np.linalg.norm(y)
# #     if nx < eps or ny < eps:
# #         return 0.0
# #     return float(np.dot(x, y) / (nx * ny))


# # def _pearson_union(E1, E2, thr1=0.0, thr2=0.0, eps=1e-12):
# #     M = (E1 > thr1) | (E2 > thr2)
# #     x = E1[M].astype(float)
# #     y = E2[M].astype(float)
# #     if x.size < 2:
# #         return 0.0
# #     x = x - x.mean()
# #     y = y - y.mean()
# #     sx = x.std()
# #     sy = y.std()
# #     if sx < eps or sy < eps:
# #         return 0.0
# #     return float(np.dot(x, y) / (len(x) * sx * sy))


# # def _overlap_coefficient_weighted(E1, E2, eps=1e-12):
# #     s1 = float(E1.sum())
# #     s2 = float(E2.sum())
# #     denom = min(s1, s2)
# #     if denom < eps:
# #         return 0.0
# #     return float(np.minimum(E1, E2).sum() / denom)


# def calculate_footprint_correlation(
#     A1_in, A2_in, gamma=0.5, thr1=0.0, thr2=0.0, dims=(512, 512), return_maps=False
# ):

#     eps = 1e-12

#     # first thing, crop footprints to their common bounding box of nonzero values (with some padding)
#     if A1_in.shape != dims:
#         A1_in = A1_in.reshape(512, 512)
#     if A2_in.shape != dims:
#         A2_in = A2_in.reshape(512, 512)

#     A1, A2 = crop_to_common_bbox(A1_in, A2_in)

#     # Optional centroid displacement estimate from weighted centroids
#     # c1 = _centroid_from_array(A1, threshold=thr1)
#     # c2 = _centroid_from_array(A2, threshold=thr2)
#     # centroid_shift = (c1[0] - c2[0], c1[1] - c2[1])

#     allowed_mask = _shift_mask(
#         A1.shape,
#         A2.shape,
#     )

#     M1 = (A1 > thr1).astype(float)
#     M2 = (A2 > thr2).astype(float)

#     # 2) Correlation map: Overlap-normalized cosine map (calculates union correlation)
#     N = fftconvolve(A1, A2[::-1, ::-1], mode="full")

#     S1 = fftconvolve(A1**2, M2[::-1, ::-1], mode="full")
#     S2 = fftconvolve(M1, (A2**2)[::-1, ::-1], mode="full")
#     # denominator terms restricted to active support
#     denom_overlap_cos = np.sqrt(np.maximum(S1, 0.0) * np.maximum(S2, 0.0))
#     overlap_cosine_map = np.zeros_like(N)
#     valid = denom_overlap_cos > eps
#     overlap_cosine_map[valid] = N[valid] / denom_overlap_cos[valid]

#     # 3) Weighting map: Overlap coefficient map on masks
#     inter = fftconvolve(M1, M2[::-1, ::-1], mode="full")
#     # denom_overlap = max(min(M1.sum(), M2.sum()), eps)
#     denom_overlap = max(np.sqrt(M1.sum() * M2.sum()), eps)
#     overlap_coeff_map = inter / denom_overlap

#     robust_map = overlap_cosine_map * np.power(
#         np.clip(overlap_coeff_map, 0.0, 1.0), gamma
#     )

#     C_max, C_zscored, shift = _best_from_score_map(robust_map, A2.shape, allowed_mask)

#     maps = None
#     if return_maps:
#         maps = {}
#         E1, E2 = _embed_shifted(A1, A2, shift[0], shift[1])
#         maps["overlap_cosine_map"] = overlap_cosine_map
#         maps["overlap_coeff_map"] = overlap_coeff_map
#         maps["robust_map"] = robust_map
#         maps["aligned_A1"] = E1
#         maps["aligned_A2"] = E2

#         return C_max, C_zscored, shift, maps
#     else:
#         return C_max, C_zscored, shift
