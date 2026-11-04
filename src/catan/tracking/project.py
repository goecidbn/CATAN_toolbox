"""A CATAN project written without the GUI (``catan match``).

Same layout as the GUI's File ▸ Save project (``Data.queue_save_project``)
for a new project: the main file ``<name>.catan.hdf5`` holds the active
assignment set and the references (relative to it) to the sessions
``<name>.sessions.hdf5`` and the model ``<name>.model.hdf5``, plus the
loading recipe of the original sources. The GUI opens it with File ▸ Open
project."""

import json
import os
from pathlib import Path

from catan.core.io import save_file

PROJECT_SUFFIX = ".catan.hdf5"


def project_files(path):
    """(main, sessions, model) file paths of the project ``path``."""
    path = Path(path)
    if not path.name.endswith(PROJECT_SUFFIX):
        raise ValueError(f"A project file ends with {PROJECT_SUFFIX}.")
    name = path.name[: -len(PROJECT_SUFFIX)]
    return (
        path,
        path.parent / f"{name}.sessions.hdf5",
        path.parent / f"{name}.model.hdf5",
    )


def save_project(tracking, path, *, recipe=None):
    """Write the sessions, the active model and the active assignments of
    ``tracking`` as project ``path`` (``…/<name>.catan.hdf5``). ``recipe``:
    loading recipe document of the sessions (stored as provenance; the GUI
    falls back to it when the session file is missing). Returns the written
    paths (main, sessions, model)."""
    if tracking.assignments is None:
        raise ValueError("No assignments to save.")
    if tracking.model is None or not tracking.model.fitted:
        raise ValueError("The model is not fitted.")

    main, sessions_file, model_file = project_files(path)
    main.parent.mkdir(parents=True, exist_ok=True)

    tracking.save_sessions(sessions_file)
    tracking.model.save(str(model_file))

    def relative(target):
        return os.path.relpath(target, main.parent)

    model_name = tracking.current_model_name
    assignments_name = tracking.current_assignments
    provenance = {
        "sessions_file": relative(sessions_file),
        # Read by older CATAN versions: the active model only.
        "model_path": relative(model_file),
        "models": json.dumps(
            [{"name": model_name, "path": relative(model_file), "active": True, "project": True}],
            ensure_ascii=False,
        ),
        # The main file's own set has path ".".
        "assignments": json.dumps(
            [{"name": assignments_name, "path": ".", "active": True, "project": True}],
            ensure_ascii=False,
        ),
    }
    if recipe is not None:
        provenance["loading_recipe"] = json.dumps(recipe, ensure_ascii=False)

    prepared = tracking.assignments.prepare_save(
        sessions=tracking.sessions,
        provenance=provenance,
        main_file=True,
    )
    save_file(
        main,
        prepared["data"],
        prepared["fields"],
        root_attributes=prepared["attributes"],
        root="/",
    )
    return main, sessions_file, model_file
