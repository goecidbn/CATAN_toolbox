from typing import Tuple, Optional, TypeVar
import numpy as np
from scipy import sparse, signal
import cv2

from .utils import normalize_array
from .image_correlation import calculate_img_correlation


def _build_remap(dims, shifts, flow=None):
    """
    returns remap arrays in 2D

    dims (2,) with (dim_x,dim_y)
    shifts (2,) with (y,x)
    flow (dim1,dim2,d) with d=dimension
    """

    x_grid, y_grid = np.meshgrid(
        np.arange(0.0, dims[1]).astype(np.float32),
        np.arange(0.0, dims[0]).astype(np.float32),
    )

    if not (flow is None):
        y_remap = (y_grid - shifts[0] + flow[..., 0]).astype(np.float32)
        x_remap = (x_grid - shifts[1] + flow[..., 1]).astype(np.float32)
    else:
        y_remap = (y_grid - shifts[0]).astype(np.float32)
        x_remap = (x_grid - shifts[1]).astype(np.float32)

    return y_remap, x_remap


def _shift_sparse_bilinear(
    A: sparse.spmatrix,
    dims: tuple[int, int],
    dy: float,
    dx: float,
    *,
    order: str = "F",
) -> sparse.csc_matrix:
    """
    Shift sparse footprint matrix A with shape (n_pixels, n_neurons).

    Parameters
    ----------
    A:
        Sparse matrix of shape (height * width, n_neurons).

    dims:
        (height, width)

    dy, dx:
        Shift in image coordinates.
        Positive dy moves footprints down.
        Positive dx moves footprints right.

    order:
        Flattening order of the pixel axis.
        CaImAn commonly uses order="F".
        NumPy default reshape uses order="C".
    """
    H, W = dims
    A = A.tocoo()

    if A.shape[0] != H * W:
        raise ValueError(f"A has {A.shape[0]} pixels, but dims={dims} implies {H * W}.")

    pix = A.row.astype(np.int64)
    neuron = A.col.astype(np.int64)
    v = A.data

    if order == "C":
        y = pix // W
        x = pix % W
    elif order == "F":
        y = pix % H
        x = pix // H
    else:
        raise ValueError("order must be 'C' or 'F'.")

    y = y.astype(float) + dy
    x = x.astype(float) + dx

    y0 = np.floor(y).astype(np.int64)
    x0 = np.floor(x).astype(np.int64)

    fy = y - y0
    fx = x - x0

    w00 = (1.0 - fy) * (1.0 - fx)
    w01 = (1.0 - fy) * fx
    w10 = fy * (1.0 - fx)
    w11 = fy * fx

    yy = np.concatenate([y0, y0, y0 + 1, y0 + 1])
    xx = np.concatenate([x0, x0 + 1, x0, x0 + 1])
    nn = np.concatenate([neuron, neuron, neuron, neuron])
    data = np.concatenate([v * w00, v * w01, v * w10, v * w11])

    mask = (yy >= 0) & (yy < H) & (xx >= 0) & (xx < W) & (data != 0)

    yy = yy[mask]
    xx = xx[mask]
    nn = nn[mask]
    data = data[mask]

    if order == "C":
        new_pix = yy * W + xx
    else:
        new_pix = yy + xx * H

    out = sparse.coo_matrix(
        (data, (new_pix, nn)),
        shape=A.shape,
    ).tocsc()

    out.sum_duplicates()
    return out


# def _shift_sparse_bilinear(A, dy, dx) -> sparse.csc_matrix:  # , output_format="csr"):
#     """
#     Apply subpixel shift to sparse matrix using bilinear splatting.
#     Way faster than using cv2.remap on a dense matrix, and keeps the matrix sparse.

#     Works best when A has clustered support (like neuron footprints).

#     Parameters
#     ----------
#     A : sparse matrix
#         Input footprint
#     dy, dx : float
#         Shift in pixels

#     Returns
#     -------
#     shifted sparse matrix (COO)
#     """
#     A = A.tocoo()

#     print(f"Shifting sparse matrix by dy={dy}, dx={dx}")
#     r = A.row.astype(float) - dx
#     c = A.col.astype(float) - dy
#     v = A.data

#     r0 = np.floor(r).astype(int)
#     c0 = np.floor(c).astype(int)

#     fr = r - r0
#     fc = c - c0

#     # Bilinear weights
#     w00 = (1 - fr) * (1 - fc)
#     w01 = (1 - fr) * fc
#     w10 = fr * (1 - fc)
#     w11 = fr * fc

#     rows = np.concatenate([r0, r0, r0 + 1, r0 + 1])
#     cols = np.concatenate([c0, c0 + 1, c0, c0 + 1])
#     data = np.concatenate([v * w00, v * w01, v * w10, v * w11])

#     # Remove out-of-bounds entries
#     m = (
#         (rows >= 0)
#         & (rows < A.shape[0])
#         & (cols >= 0)
#         & (cols < A.shape[1])
#         & (data != 0)
#     )

#     rows = rows[m]
#     cols = cols[m]
#     data = data[m]

#     out = sparse.csc_matrix((data, (rows, cols)), shape=A.shape)
#     out.sum_duplicates()
#     return out

#     # if output_format == "coo":
#     #     return out
#     # elif output_format == "csr":
#     #     return out.tocsr()
#     # elif output_format == "csc":
#     # return out.tocsc()
#     # elif output_format == "lil":
#     #     return out.tolil()
#     # elif output_format == "dok":
#     #     return out.todok()
#     # else:
#     #     raise ValueError(f"Unsupported output_format: {output_format}")


# def get_shift_and_flow(
def get_session_remap(
    A1: np.ndarray,
    A2: np.ndarray,
    dims: Tuple[int, int] = (512, 512),
    projection: Optional[int] = -1,
    use_optical_flow: bool = False,
):
    ## dims:          shape of the (projected) image
    ## projection:    axis, along which to project. If None, no projection needed

    if not (projection is None):
        A1 = np.array(A1.sum(projection))
        A2 = np.array(A2.sum(projection))

    if not isinstance(A1, np.ndarray):
        A1 = np.array(A1)
    if not isinstance(A2, np.ndarray):
        A2 = np.array(A2)

    A1 = A1.reshape(dims)
    A2 = A2.reshape(dims)

    A1 = normalize_array(A1, "uint", 8)
    A2 = normalize_array(A2, "uint", 8)
    c, c_zscored, shift = calculate_img_correlation(A1, A2, mode="correlation")

    if not use_optical_flow:
        return shift, None, c, c_zscored
    y_remap, x_remap = _build_remap(dims, shift)

    A2 = cv2.remap(A2, x_remap, y_remap, interpolation=cv2.INTER_CUBIC)
    A2 = normalize_array(A2, "uint", 8)
    # A2 = normalize_array(A2,'uint',8)

    flow = cv2.calcOpticalFlowFarneback(
        A1, A2, np.array([], dtype=np.float32), 0.5, 5, 128, 3, 7, 1.5, 0
    )

    return shift, flow, c, c_zscored
