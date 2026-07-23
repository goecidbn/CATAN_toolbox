import pytest
from importlib.metadata import entry_points

@pytest.mark.gui
def test_gui_entrypoint_exists():
    matches = list(
        entry_points(group="gui_scripts", name="catan")
    )

    assert len(matches) == 1

    entry = matches[0]

    assert entry.value == "catan.gui.app:main"
    assert callable(entry.load())