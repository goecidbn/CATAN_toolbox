import numpy as np
import pytest

from catan.tracking.analytics import two_noise_model as tn

# ID29 fit (2026-10-30).
THETA = {
    "p_same": 0.2036, "h": 7.9, "sigma_h": 2.09, "sigma_det": 0.4196, "sigma_move": 1.4348,
    "nu_move": 2.289, "u_same_mean": 0.0411, "u_same_shape": 3.4328, "u_same_slope": 0.0219,
    "u_diff_mean": 0.0813, "u_diff_shape": 5.7056,
}


def _ctx():
    return tn.Context(lam=6.6e-4, dims=(512.0, 512.0), R=25.0,
                      settings=tn.settings_with_defaults())


def test_same_neuron_correlation_decreases_with_distance():
    assert set(THETA) == set(tn.PARAMETER_NAMES)
    d = np.array([0.0, 2.0, 4.0])
    means = tn.same_mean(THETA, d)
    assert means[0] == pytest.approx(THETA["u_same_mean"])
    assert np.all(np.diff(means) > 0)  # mean of 1 - correlation grows


def test_posterior_monotone():
    ctx = _ctx()
    d = np.linspace(0.25, 20, 40)
    c = np.linspace(0.5, 0.995, 40)
    p = tn.posterior(THETA, ctx, d, c)
    assert np.all(np.diff(p, axis=0) <= 1e-12)  # decreasing with distance
    assert np.all(np.diff(p, axis=1) >= -1e-12)  # increasing with correlation


def test_fit_recovers_correlation_parameters():
    ctx = _ctx()
    H0, H1 = tn.predict(THETA, ctx)
    evidence = {
        "same2": np.round(H0 / H0.sum() * 2e5),
        "cross2": np.repeat(np.round(H1 / H1.sum() * 4e5)[..., None], 3, axis=-1),
        "vector": None,
    }
    fitted, _ = tn.fit(evidence, ctx, fixed={"sigma_det": THETA["sigma_det"]})
    assert set(fitted) == set(tn.PARAMETER_NAMES)
    for name in ("u_same_mean", "u_same_shape", "p_same", "u_diff_mean"):
        assert abs(fitted[name] / THETA[name] - 1) < 0.1, (name, fitted[name])
    assert abs(fitted["u_same_slope"] - THETA["u_same_slope"]) < 0.01


def test_panel_shows_gamma_means_as_mean_correlations():
    QtWidgets = pytest.importorskip("PySide6.QtWidgets")
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    import catan.gui.GUI_elements.main_window  # noqa: F401 (import order)
    from catan.gui.panels import ModelResults

    parent = QtWidgets.QWidget()
    display = ModelResults.Display(parent)
    assert set(display.editors) == set(tn.PARAMETER_NAMES)
    for key in ("u_same_mean", "u_diff_mean"):
        assert display.to_display(key, 0.04) == pytest.approx(0.96)
        assert display.from_display(key, 0.96) == pytest.approx(0.04)
        low, high = tn.PARAMETERS[key][2:4]
        editor = display.editors[key]
        assert editor.minimum() == pytest.approx(1 - high, abs=1e-4)
        assert editor.maximum() == pytest.approx(1 - low, abs=1e-4)
    assert display.to_display("h", 7.9) == pytest.approx(7.9)
    assert "β_same" in display.editors["u_same_slope"].toolTip()
    assert app is not None
