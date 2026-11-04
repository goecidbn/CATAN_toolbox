"""Loading recipes: the sessions of a dataset with their load configuration
and load options (JSON, ``"type": "catan-loading-recipe"``). Written by the
GUI (File ▸ Save loading recipe, stored in project files) and by ``catan
match``; read by both."""

import os
from pathlib import Path

import numpy as np

from catan.core.structures.load_config import LoadConfig
from catan.core.structures.session import SessionData

RECIPE_TYPE = "catan-loading-recipe"
RECIPE_VERSION = 1

BACKGROUND_ORIENTATIONS = {"auto", "as_stored", "transpose"}
BACKGROUND_MODES = {"configured", "footprints"}


def recipe_document(entries):
    """Recipe document for session ``entries`` ({path, name, load_config,
    load_options, active})."""
    if not entries:
        raise ValueError("There are no sessions to save.")
    return {"type": RECIPE_TYPE, "version": RECIPE_VERSION, "sessions": list(entries)}


def parse_loading_recipe(document, base_directory, *, ctx=None):
    """Recipe document -> unregistered SessionData (relative session
    paths resolve against ``base_directory``). ``ctx``: task context checked
    for cancellation between entries."""
    path = Path(base_directory) / "recipe.json"

    if (
        not isinstance(document, dict)
        or document.get("type") != RECIPE_TYPE
        or document.get("version") != RECIPE_VERSION
    ):
        raise ValueError("This JSON is not a supported CATAN loading recipe.")

    entries = document.get("sessions")
    if not isinstance(entries, list) or not entries:
        raise ValueError("The loading recipe contains no sessions.")

    sessions = []

    for index, entry in enumerate(entries, start=1):
        if ctx is not None:
            ctx.check_cancelled()

        try:
            source = entry["path"]
            if not isinstance(source, str) or not source.strip():
                raise ValueError("Missing session source path.")

            source = Path(source).expanduser()
            if not source.is_absolute():
                source = path.parent / source

            # Lexical normalization, without inspecting remote sources.
            source = os.path.abspath(str(source))

            config = LoadConfig.from_dict(entry["load_config"])
            if config.source_type != "session":
                raise ValueError("Expected a session load configuration.")

            options = dict(entry.get("load_options", {}))
            options.setdefault("background_orientation", "auto")
            options.setdefault("background_mode", "configured")
            options.setdefault("correct_rotation", False)

            if options["background_orientation"] not in BACKGROUND_ORIENTATIONS:
                raise ValueError("Invalid background orientation.")

            if options["background_mode"] not in BACKGROUND_MODES:
                raise ValueError("Invalid background mode.")

            if not isinstance(options["correct_rotation"], bool):
                raise ValueError("correct_rotation must be true or false.")

            if options.get("alignment_source") not in {None, "background", "footprints"}:
                raise ValueError("Invalid alignment source.")

            if "footprint_offset" in options:
                offset = np.asarray(options["footprint_offset"], dtype=float)
                if offset.shape != (2,) or not np.isfinite(offset).all():
                    raise ValueError("footprint_offset must be two numbers (dx, dy).")

            if "background_correction" in options:
                correction = options["background_correction"]
                shift = np.asarray(
                    (correction or {}).get("shift", ()), dtype=float
                ).reshape(-1)
                rotation = (correction or {}).get("rotation", 0.0)
                if (
                    not isinstance(correction, dict)
                    or shift.shape != (2,)
                    or not np.isfinite(shift).all()
                    or not isinstance(rotation, (int, float))
                ):
                    raise ValueError(
                        "background_correction must be "
                        '{"shift": [dx, dy], "rotation": degrees}.'
                    )

            session = SessionData(
                path=source,
                name=entry.get("name"),
            )
            session.source_config = config
            session._recipe_options = options
            session.active = bool(entry.get("active", True))
            sessions.append(session)

        except (KeyError, TypeError, ValueError, AttributeError) as exc:
            raise ValueError(
                f"Invalid loading recipe, session entry {index}: {exc}"
            ) from exc

    return sessions
