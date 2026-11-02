import numpy as np
from scipy import ndimage, sparse

from catan.core import footprint_frame
from catan.core.structures.session import SessionData


def _cells(dims=(128, 128), count=60, seed=0):
    rng = np.random.default_rng(seed)
    columns = []
    for y, x in rng.integers(12, dims[0] - 12, (count, 2)):
        image = np.zeros(dims, dtype=np.float32)
        image[y, x] = 1.0
        image = ndimage.gaussian_filter(image, 2.0)
        image[image < 0.01 * image.max()] = 0
        columns.append(sparse.csc_matrix(image.reshape(-1, 1)))
    return sparse.hstack(columns, format="csc")


def _moved(image, dx, dy):
    """``image`` moved by (dx, dy) px (right / down)."""
    return ndimage.shift(np.asarray(image, dtype=np.float32), (dy, dx), order=1)


def _session(footprints, background, dims, **params):
    session = SessionData(name="test")
    session.params.update(params)
    session.register_spatial(
        footprints=footprints,
        background=background,
        dims=dims,
        background_orientation="as_stored",
    )
    return session


def test_coherence_matches_and_reports_offset():
    dims = (128, 128)
    projection = footprint_frame.projection(_cells(dims), dims)

    matched = footprint_frame.background_coherence(projection, projection)
    assert matched["coherent"] and np.allclose(matched["offset"], 0, atol=0.1)

    # Footprints moved right / up by (4, -3): the offset moves them back.
    result = footprint_frame.background_coherence(projection, _moved(projection, 4.0, -3.0))
    assert result["coherent"] is False and footprint_frame.matched(result)
    assert np.allclose(result["offset"], (-4.0, 3.0), atol=0.2)

    unrelated = footprint_frame.background_coherence(projection, projection[::-1].copy())
    assert unrelated["coherent"] is False and not footprint_frame.matched(unrelated)


def test_estimate_correction_moves_the_background_onto_the_footprints():
    dims = (128, 128)
    projection = footprint_frame.projection(_cells(dims), dims)
    background = _moved(projection, 5.0, -3.0)

    correction, after, before = footprint_frame.estimate_correction(background, projection)
    assert not before["coherent"]
    assert np.allclose(correction["shift"], (-5.0, 3.0), atol=0.2)
    assert correction["rotation"] == 0.0 and correction["source"] == "auto"
    assert after["coherent"] and np.allclose(after["offset"], 0, atol=0.2)

    # A matching background needs none; an unrelated one cannot be corrected.
    assert footprint_frame.estimate_correction(projection, projection)[0] is None
    assert footprint_frame.estimate_correction(projection[::-1].copy(), projection)[0] is None


def test_estimate_correction_finds_a_rotation():
    dims = (160, 160)
    projection = footprint_frame.projection(_cells(dims, count=120, seed=3), dims)
    from catan.core.structures.remap import Remapping

    rotated = Remapping._warp_dense(projection, Remapping._rotation_matrix(dims, 1.5))
    background = _moved(rotated, 3.0, 2.0)
    correction, after, before = footprint_frame.estimate_correction(background, projection)
    assert abs(correction["rotation"] + 1.5) <= 0.25
    assert after["correlation_zero"] > before["correlation_zero"]
    assert np.hypot(*after["offset"]) < 0.5


def test_session_corrects_its_background_not_its_footprints():
    dims = (128, 128)
    footprints = _cells(dims)
    projection = footprint_frame.projection(footprints, dims)

    reference = _session(footprints, projection, dims)
    assert reference.background_correction is None
    assert reference.alignment_template() is reference.background_template

    session = _session(footprints, _moved(projection, 4.0, -3.0), dims)
    correction = session.background_correction
    assert np.allclose(correction["shift"], (-4.0, 3.0), atol=0.2)
    # The footprints are untouched; the background is aligned corrected.
    assert np.allclose(session.centroids, reference.centroids)
    assert session.background_coherence["coherent"]
    assert np.allclose(session.background_coherence["uncorrected_offset"], (4.0, -3.0), atol=0.2)
    assert session.alignment_source == "background"
    corrected = session.alignment_template()
    assert corrected is session.corrected_template()
    assert np.corrcoef(corrected[20:-20, 20:-20].ravel(), projection[20:-20, 20:-20].ravel())[0, 1] > 0.98

    # Not resembling the footprints at all: aligned by them instead.
    unrelated = _session(footprints, projection[::-1].copy(), dims)
    assert unrelated.background_correction is None
    assert unrelated.alignment_source == "footprints"
    # An explicit choice wins.
    unrelated.params["alignment_source"] = "background"
    assert unrelated.alignment_template() is unrelated.background_template


def test_manual_corrections_and_older_footprint_offsets():
    dims = (128, 128)
    footprints = _cells(dims)
    projection = footprint_frame.projection(footprints, dims)
    background = _moved(projection, 4.0, -3.0)

    # An older footprint offset (moved the footprints onto the background)
    # becomes the opposite background correction, kept as manual.
    legacy = _session(footprints, background, dims, footprint_offset=[4.0, -3.0])
    assert "footprint_offset" not in legacy.params
    correction = legacy.background_correction
    assert correction["source"] == "manual"
    assert np.allclose(correction["shift"], (-4.0, 3.0))

    # A manual zero ("no correction") is not estimated again.
    footprint_frame.set_correction(
        params := {}, {"shift": (0.0, 0.0), "rotation": 0.0, "source": "manual"}
    )
    kept = _session(footprints, background, dims, **params)
    assert kept.background_correction["shift"] == (0.0, 0.0)
    assert kept.corrected_template() is kept.background_template


def test_snapshot_keeps_the_correction(tmp_path):
    from catan.core.structures.session_snapshot import prepare_session_snapshot
    import json

    dims = (64, 64)
    footprints = _cells(dims, count=20)
    projection = footprint_frame.projection(footprints, dims)
    session = _session(footprints, _moved(projection, 3.0, 0.0), dims)
    session.path = str(tmp_path / "session.hdf5")
    document = json.loads(prepare_session_snapshot(session)["document"])
    stored = document["session"]["params"]["background_correction"]
    assert stored["source"] == "auto" and np.allclose(stored["shift"], (-3.0, 0.0), atol=0.2)
    assert document["session"]["background_coherence"]["coherent"] is True


def test_recipe_correction_is_rigid_and_manual():
    params = {}
    footprint_frame.set_correction(
        params,
        {"shift": (1.0, 2.0), "rotation": 0.5, "flow": np.zeros((4, 4, 2)), "source": "manual"},
    )
    correction = footprint_frame.correction_of(params)
    assert correction["flow"].dtype == np.float32
    remap = footprint_frame.correction_remap((4, 4), correction)
    assert np.allclose(remap.shift, (2.0, 1.0)) and remap.rotation == 0.5
