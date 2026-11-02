import numpy as np
from scipy import sparse

from catan.core.footprint_remainder import remainder_footprint
from catan.core.structures import NeuronComponent
from catan.gui.structures.request_handler import RequestHandler


def column(values):
    return sparse.csc_matrix(np.asarray(values, dtype=float).reshape(-1, 1))


def test_remainder_removes_scaled_piece():
    cell_a = np.array([0, 2, 4, 2, 0, 0, 0, 0])
    cell_b = np.array([0, 0, 0, 0, 1, 3, 1, 0])
    whole = cell_a + cell_b

    # The source piece comes from another session at half the intensity.
    rest = remainder_footprint(column(whole), [column(cell_a / 2)])

    assert np.allclose(rest.toarray().ravel(), cell_b)


def test_nothing_remains():
    cell = np.array([0, 1, 3, 1, 0])
    assert remainder_footprint(column(cell), [column(cell)]) is None


def test_split_request_with_one_source():
    request = RequestHandler("split", NeuronComponent(29, 2))
    request.advance_stage()
    assert not request.is_complete

    request.add_component(NeuronComponent(29, 0))
    assert request.is_complete and request.stage_complete
    assert request.roles == {
        NeuronComponent(29, 2): "target",
        NeuronComponent(29, 0): "source",
    }

    # A second source of the same neuron is refused.
    assert request.validation_error(NeuronComponent(29, 1)) is not None

    request.add_component(NeuronComponent(381, 5))
    assert request.is_complete
    assert "Sources" in request.destination_status_text()


def test_merge_still_needs_its_source():
    request = RequestHandler("merge", NeuronComponent(30, 0))
    request.add_component(NeuronComponent(39, 0))
    request.advance_stage()
    assert not request.is_complete
