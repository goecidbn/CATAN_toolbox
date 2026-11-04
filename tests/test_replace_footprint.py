from copy import copy

import numpy as np
from scipy import sparse

from catan.core.redetection import centroid, detected_position, place_footprint
from catan.core.structures.session import SessionData
from catan.tracking.structures import Assignments


def _blob(dims, y, x, sigma=1.5):
    yy, xx = np.mgrid[: dims[0], : dims[1]]
    image = np.exp(-((yy - y) ** 2 + (xx - x) ** 2) / (2 * sigma**2))
    image[image < 0.05] = 0
    return sparse.csc_matrix(image.reshape(-1, 1).astype(np.float32))


def _session(dims=(32, 32)):
    session = SessionData(name="test")
    footprints = sparse.hstack(
        [_blob(dims, 8, 8), _blob(dims, 20, 20, sigma=3.0)], format="csc"
    )
    session.register_spatial(
        footprints=footprints,
        background=np.zeros(dims, dtype=np.float32),
        dims=dims,
        background_orientation="as_stored",
    )
    return session


def test_replace_and_restore_detected_shape():
    session = _session()
    detected = session.footprints[:, 1].copy()
    replacement = place_footprint(_blob(session.dims, 5, 5), session.dims, (20.0, 20.0))

    # A shallow copy (as for staged sessions) must not see the change.
    other = copy(session)

    session.replace_footprint(1, replacement)
    assert session.n_neurons == 2
    assert abs(session.footprints[:, 1] - replacement).sum() < 1e-6
    assert np.allclose(session.centroids[1], (20.0, 20.0), atol=0.2)
    assert abs(other.footprints[:, 1] - detected).sum() < 1e-6

    session.remove_synthetic_components()
    assert abs(session.footprints[:, 1] - detected).sum() < 1e-6
    assert not session.replaced_footprints


def test_shapes_survive_serialization():
    dims = (16, 16)
    shapes = [_blob(dims, 4, 4), _blob(dims, 10, 9)]
    record = Assignments._serialize_shapes(shapes)
    restored = Assignments._deserialize_shapes(record)
    assert len(restored) == 2
    assert all(abs(a - b).sum() < 1e-6 for a, b in zip(shapes, restored))


def test_union_placed_at_detected_position():
    # Asymmetric position: (y, x) and (x, y) differ.
    dims = (32, 40)
    session = SessionData(name="test")
    session.register_spatial(
        footprints=_blob(dims, 6, 25),
        background=np.zeros(dims, dtype=np.float32),
        dims=dims,
        background_orientation="as_stored",
    )
    position = detected_position(session, 0)
    assert np.allclose(position, (6.0, 25.0), atol=0.05)

    shape = place_footprint(_blob(dims, 20, 10, sigma=2.0), dims, position)
    session.replace_footprint(0, shape)
    assert np.allclose(centroid(session.footprints[:, 0], dims), (6.0, 25.0), atol=0.2)
    # Still the detected position after the replacement.
    assert np.allclose(detected_position(session, 0), (6.0, 25.0), atol=0.05)


def test_union_position_interpolated_from_other_footprints():
    from catan.core.redetection import interpolated_position

    observations = [(0, np.array([10.0, 20.0])), (4, np.array([18.0, 12.0]))]
    # Between: linear in the session index.
    assert np.allclose(interpolated_position(observations, 1), (12.0, 18.0))
    # Before the first / after the last: the nearest one.
    assert np.allclose(interpolated_position(observations[1:], 2), (18.0, 12.0))
    assert np.allclose(interpolated_position(observations, 6), (18.0, 12.0))
