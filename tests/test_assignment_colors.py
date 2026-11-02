import colorsys

import numpy as np
import pytest

from catan.gui.data import assignment_colors as ac


@pytest.fixture(autouse=True)
def no_overrides(monkeypatch):
    monkeypatch.setattr(ac, "_overrides", {})
    monkeypatch.setattr(ac, "_settings", None)


def hue(rgba):
    return colorsys.rgb_to_hls(*rgba[:3])[0]


def test_colliding_hashes_get_distinct_colors():
    # These two names hash to almost the same hue.
    a, b = ac.assignment_colors(["local", "ID58_multisession_registration"])
    assert ac._hue_distance(hue(a), hue(b)) >= ac.MIN_HUE_DISTANCE
    # The first keeps its own (hashed) colour.
    assert np.allclose(a, ac.assignment_color("local"))


def test_same_name_same_color():
    a, b = ac.assignment_colors(["local", "local"])
    assert np.allclose(a, b)


def test_user_color_kept_and_avoided():
    ac.set_assignment_color("other", (0.2, 0.3, 0.8, 1.0))
    a, b = ac.assignment_colors(["other", "local"])
    assert np.allclose(a, (0.2, 0.3, 0.8, 1.0))
    assert ac._hue_distance(hue(a), hue(b)) >= ac.MIN_HUE_DISTANCE

    ac.set_assignment_color("other", None)
    assert not ac.has_custom_color("other")


def test_colors_persist_in_settings():
    class Settings(dict):
        def setValue(self, key, value):
            self[key] = value

        def value(self, key, default=None):
            return self.get(key, default)

    settings = Settings()
    ac.load_assignment_colors(settings)
    ac.set_assignment_color("local", (1.0, 0.0, 0.0, 1.0))

    ac._overrides.clear()
    ac.load_assignment_colors(settings)
    assert np.allclose(ac.assignment_color("local"), (1.0, 0.0, 0.0, 1.0))
