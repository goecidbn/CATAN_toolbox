from pathlib import Path
from types import SimpleNamespace

import h5py
import pytest

# GUI package tests: need PySide6 (the headless CI job runs -m "not gui").
pytest.importorskip("PySide6")
pytestmark = pytest.mark.gui

from catan.gui.structures.data import Data, is_project_set, read_assignments_provenance


def test_project_set_flag():
    created = SimpleNamespace(loaded=False, path=None)
    loaded = SimpleNamespace(loaded=True, path="/x/model.hdf5")
    assert is_project_set(created)
    assert not is_project_set(loaded)
    # Assignment sets have no ``loaded`` attribute: a file path makes them external.
    assert not is_project_set(SimpleNamespace(path="/x/assignments.hdf5"))
    # An explicit flag wins (restored project set, saved created set).
    assert is_project_set(SimpleNamespace(loaded=True, path="/p/a.model.hdf5", project_set=True))
    assert not is_project_set(SimpleNamespace(loaded=False, path=None, project_set=False))


def test_project_set_file_names():
    folder = Path("/p/catan")
    pick = Data._project_model_file
    first = pick(folder, "MS", "local", set())
    assert first == str(folder / "MS.model.hdf5")
    second = pick(folder, "MS", "Fresh copy", {first})
    assert second == str(folder / "MS.Fresh_copy.model.hdf5")
    third = pick(folder, "MS", "Fresh copy", {first, second})
    assert third == str(folder / "MS.Fresh_copy_2.model.hdf5")

    sets = Data._project_set_file
    a = sets(folder, "MS", "local", set(), "assignments")
    assert a == str(folder / "MS.local.assignments.hdf5")
    # Not a project main file name.
    assert not a.endswith(Data.PROJECT_SUFFIX)
    assert sets(folder, "MS", "local", {a}, "assignments") == str(
        folder / "MS.local_2.assignments.hdf5"
    )


def test_provenance_lists_assignment_sets(tmp_path):
    main = tmp_path / "MS.catan.hdf5"
    other = tmp_path / "MS.copy.assignments.hdf5"
    external = tmp_path / "elsewhere" / "ext.hdf5"
    external.parent.mkdir()
    for path in (main, other, external):
        with h5py.File(path, "w") as file:
            file.attrs["object_type"] = "AssignmentsData"
    sets = (
        '[{"name": "local", "path": ".", "active": false, "project": true},'
        ' {"name": "copy", "path": "MS.copy.assignments.hdf5", "active": true, "project": true},'
        ' {"name": "ext", "path": "elsewhere/ext.hdf5", "active": false, "project": false}]'
    )
    with h5py.File(main, "a") as file:
        file["provenance/assignments"] = sets

    provenance = read_assignments_provenance(main)
    entries = {entry["name"]: entry for entry in provenance["assignment_sets"]}
    assert entries["local"]["main"] and entries["local"]["path"] == str(main.resolve())
    assert entries["copy"]["active"] and entries["copy"]["project"]
    assert entries["ext"]["path"] == str(external.resolve())
    assert not entries["ext"]["project"] and not entries["ext"]["main"]
