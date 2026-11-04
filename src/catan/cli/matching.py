"""The matching run of ``catan match``: load and align the sessions, build
and fit the model, register the neurons, save a CATAN project.

The same core calls as the GUI (``Data`` in gui/structures/data.py), without
its manual corrections. Every session is checked while loading; a problem
(file not readable, image size, alignment failed or deviating) is reported
with what was measured and the possible ways on (prompts.Prompter): retry
with other alignment settings, accept, leave the session out or stop.
"""

from __future__ import annotations

import json
import time
from collections import Counter
from dataclasses import dataclass, field, replace
from pathlib import Path

import numpy as np

from .detection import (
    SessionPlan,
    check_geometry,
    choose_configs,
    choose_orientation,
    plan_backgrounds,
)
from .prompts import Aborted

ISSUE_POLICIES = ("ask", "accept", "skip", "abort")


@dataclass(frozen=True)
class MatchingConfiguration:
    # µm per pixel; asked when None (the two-noise model's 10 µm slab).
    pixel_size: float | None = None
    n_threads: int = 1
    neighbor_distance: float = 25.0
    # Alignment (as the GUI's alignment settings).
    max_shift: float = 50.0
    min_flow_correlation: float = 0.10
    correct_rotation: bool = False
    refine_with_footprints: bool = False
    # Model evidence from session pairs this far apart (session order).
    session_distances: tuple = (1,)
    # Loading: preset name / JSON, background ("configured", "footprints"
    # or an image pattern), orientation ("as_stored", "transpose", "auto").
    load_config: str | None = None
    background: str | None = None
    orientation: str | None = None
    # Session problems without a terminal: accept (where possible, else
    # skip), skip, abort; "ask" needs a terminal.
    on_issue: str = "ask"
    project_name: str | None = None
    overwrite: bool = False

    def __post_init__(self) -> None:
        if self.pixel_size is not None and self.pixel_size <= 0:
            raise ValueError("pixel_size must be greater than zero.")
        if self.n_threads < 1:
            raise ValueError("n_threads must be at least 1.")
        if self.neighbor_distance <= 0:
            raise ValueError("neighbor_distance must be greater than zero.")
        if self.max_shift <= 0:
            raise ValueError("max_shift must be greater than zero.")
        if not 0 <= self.min_flow_correlation <= 1:
            raise ValueError("min_flow_correlation must be between 0 and 1.")
        if not self.session_distances or min(self.session_distances) < 1:
            raise ValueError("session distances must be 1 or more.")
        if self.orientation not in (None, "auto", "as_stored", "transpose"):
            raise ValueError("orientation must be auto, as_stored or transpose.")
        if self.on_issue not in ISSUE_POLICIES:
            raise ValueError(f"on_issue must be one of {', '.join(ISSUE_POLICIES)}.")


@dataclass
class MatchingResult:
    project: Path
    sessions_file: Path
    model_file: Path
    recipe_file: Path
    sessions: list = field(default_factory=list)
    skipped: list = field(default_factory=list)
    warnings: list = field(default_factory=list)


# ----------------------------------------------------------------------
# Plans: how each session is read
# ----------------------------------------------------------------------


def session_names(paths):
    """Session names: the parent folder (as the GUI), else the file name
    when folders repeat."""
    names = [Path(path).parent.name for path in paths]
    if len(set(names)) == len(names):
        return names
    names = [Path(path).stem for path in paths]
    if len(set(names)) == len(names):
        return names
    return [f"{Path(path).stem}_{index:03d}" for index, path in enumerate(paths)]


def plans_from_recipe(recipe_path):
    from catan.core.structures.loading_recipe import parse_loading_recipe

    recipe_path = Path(recipe_path).expanduser().resolve()
    document = json.loads(recipe_path.read_text(encoding="utf-8"))
    sessions = parse_loading_recipe(document, recipe_path.parent)
    return [
        SessionPlan(
            path=Path(session.path),
            name=session.name or Path(session.path).parent.name,
            config=session.source_config,
            options=dict(session._recipe_options),
        )
        for session in sessions
        if getattr(session, "active", True)
    ]


def config_manager():
    """The GUI's load presets: packaged and the user's own."""
    from platformdirs import user_config_dir

    from catan.core.structures.load_config.manager import LoadConfigManager

    return LoadConfigManager(user_dir=Path(user_config_dir("CATAN")) / "load_configs")


def plan_sessions(paths, config, prompter):
    """SessionPlan per usable file: load preset, background and orientation
    detected (asked where unclear), image geometry checked."""
    prompter.info("\n== Reading the session files")
    configs = choose_configs(list(paths), config_manager(), prompter, requested=config.load_config)
    backgrounds = plan_backgrounds(configs, prompter, requested=config.background)

    # One orientation for all backgrounds, from the first one.
    orientation = config.orientation
    with_background = [path for path, (_, mode) in backgrounds.items() if mode == "configured"]
    if orientation is None and with_background:
        first = with_background[0]
        orientation = choose_orientation(first, backgrounds[first][0], prompter)
    orientation = orientation or "auto"

    names = dict(zip(paths, session_names(paths)))
    plans = []
    expected = None
    for path in paths:
        if path not in backgrounds:
            continue
        load_config, mode = backgrounds[path]
        options = {
            "background_orientation": orientation if mode == "configured" else "auto",
            "background_mode": mode,
            "correct_rotation": bool(config.correct_rotation),
        }
        ok, dims, message = check_geometry(path, load_config, options["background_orientation"], expected)
        if not ok:
            choice = decide(
                prompter,
                config,
                f"{path} cannot be loaded as configured: {message}",
                [("skip", "Leave this session out"), ("abort", "Stop")],
            )
            if choice == "abort":
                raise Aborted(f"{path}: {message}")
            prompter.warn(f"{names[path]} left out: {message}")
            continue
        expected = expected or tuple(dims)
        plans.append(SessionPlan(Path(path), names[path], load_config, options))

    prompter.note(
        f"{len(plans)} session(s) to load, image {expected[0]} × {expected[1]} px."
        if plans
        else "No session can be loaded."
    )
    return plans


# ----------------------------------------------------------------------
# Decisions
# ----------------------------------------------------------------------


def decide(prompter, config, question, choices, *, details=None):
    """A choice about a problematic session: --on-issue answers it without
    asking (accept → accept if offered, else skip)."""
    keys = dict(choices)
    policy = config.on_issue
    if policy != "ask":
        key = policy if policy in keys else ("skip" if policy == "accept" else policy)
        if key in keys:
            prompter.warn(f"{question}\n→ {keys[key]} (--on-issue {policy})")
            return key
    return prompter.choose(
        question, choices, option="--on-issue accept|skip|abort", details=details
    )


# ----------------------------------------------------------------------
# Loading and alignment
# ----------------------------------------------------------------------


def _first_sentence(text):
    return str(text).split(". ")[0].rstrip(". ")


def _describe_report(remap):
    report = remap.report
    parts = []
    proposal = report.success or getattr(remap, "confirmation", None)
    if proposal and report.total_shift is not None and np.isfinite(report.total_shift):
        parts.append(f"shift {report.total_shift:.1f} px")
    if proposal and report.rotation:
        parts.append(f"rotation {report.rotation:.2f}°")
    if report.correlation is not None:
        parts.append(f"correlation {report.correlation:.2f}")
    parts.append(f"{report.n_successful_references}/{report.n_references} references accepted")
    return ", ".join(parts)


def load_session(tracking, plan, config, prompter, *, expected_dims, shared=None):
    """Load ``plan`` aligned to the sessions registered so far; returns the
    SessionData (not registered) or None (left out). Asks on problems.
    ``shared``: alignment settings changed by a retry, kept for the
    following sessions (dataset settings, as in the GUI)."""
    shared = {} if shared is None else shared
    from catan.core.structures.session import SessionData

    settings = {
        "max_session_shift": float(config.max_shift),
        "min_flow_correlation": float(config.min_flow_correlation),
        "correct_rotation": bool(plan.options.get("correct_rotation", False)),
        "refine_alignment_with_footprints": bool(config.refine_with_footprints),
        **shared,
    }
    if plan.options.get("alignment_source"):
        settings["alignment_source"] = plan.options["alignment_source"]
    first = not tracking.sessions

    while True:
        session = SessionData(name=plan.name, path=str(plan.path))
        session.source_config = plan.config
        session.params.update(
            um_per_px=float(tracking.params["um_per_px"]),
            alignment_offsets=[],
            **settings,
        )
        started = time.perf_counter()
        try:
            session.load_data(
                plan.config.get_fields_to_load(),
                alignment_references=tracking.alignment_references_for_session(None),
                background_orientation=plan.options.get("background_orientation", "auto"),
                expected_dims=expected_dims,
            )
        except Exception as exc:
            choice = decide(
                prompter,
                config,
                f"{plan.name}: loading failed ({type(exc).__name__}: {exc}).",
                [("skip", "Leave this session out"), ("abort", "Stop")],
            )
            if choice == "abort":
                raise Aborted(f"{plan.name}: {exc}") from exc
            return None
        elapsed = time.perf_counter() - started

        if session.n_neurons == 0:
            choice = decide(
                prompter,
                config,
                f"{plan.name} has no neurons.",
                [("skip", "Leave this session out"), ("abort", "Stop")],
            )
            if choice == "abort":
                raise Aborted(f"{plan.name} has no neurons.")
            return None

        line = f"{plan.name}: {session.n_neurons} neurons ({elapsed:.1f} s)"
        coherence = session.background_coherence or {}
        if coherence.get("coherent") is False:
            prompter.warn(
                f"{plan.name}: the background does not match the footprints "
                f"({coherence.get('reason')}); "
                + (
                    "the session is aligned by its footprints."
                    if session.alignment_source == "footprints"
                    else "the session is aligned by its background."
                )
            )
        elif coherence.get("correction"):
            line += f"; background moved onto its footprints ({coherence['correction']})"

        if first:
            prompter.note(line + "; reference session.")
            plan.options.update(_recipe_settings(session, settings))
            return session

        remap = session.remap
        report = remap.report
        if report.success:
            prompter.note(line + "; aligned: " + _describe_report(remap) + ".")
            flow = remap.flow_info or {}
            if flow.get("status") == "skipped":
                prompter.warn(f"{plan.name}: {remap.flow_message}")
            plan.options.update(_recipe_settings(session, settings))
            return session

        # Not aligned: ask how to go on.
        confirmation = getattr(remap, "confirmation", None)
        if confirmation:
            question = (
                f"{plan.name}: the automatic alignment deviates "
                f"({_first_sentence(confirmation).lower()}) and needs confirmation."
            )
            details = f"{confirmation}\nProposal: {_describe_report(remap)}."
        else:
            reason = _first_sentence((report.reason or "unknown").replace("_", " "))
            question = f"{plan.name}: automatic alignment failed ({reason.lower()})."
            details = f"{_describe_report(remap)}.\n{remap.flow_message}"

        choices = []
        if confirmation:
            choices.append(("accept", "Accept the proposed alignment"))
        if not settings["correct_rotation"]:
            choices.append(("rotation", "Retry, also correcting rotations"))
        choices.append(("shift", f"Retry with a larger maximal shift (now {settings['max_session_shift']:.0f} px)"))
        choices.append(
            ("flow", f"Retry with a lower minimal flow correlation (now {settings['min_flow_correlation']:.2f})")
        )
        if session.background_origin != "footprints" and session.alignment_source != "footprints":
            choices.append(("footprints", "Retry, aligning by the footprints instead of the background"))
        choices += [("skip", "Leave this session out"), ("abort", "Stop")]

        choice = decide(prompter, config, question, choices, details=details)
        if choice == "abort":
            raise Aborted(f"{plan.name}: alignment {report.reason}.")
        if choice == "skip":
            return None
        if choice == "accept":
            settings["confirm_alignment_deviations"] = False
        elif choice == "rotation":
            settings["correct_rotation"] = shared["correct_rotation"] = True
        elif choice == "shift":
            settings["max_session_shift"] = shared["max_session_shift"] = float(
                prompter.ask_number(
                    "Maximal shift (px)?",
                    default=2 * settings["max_session_shift"],
                    minimum=1,
                )
            )
        elif choice == "flow":
            settings["min_flow_correlation"] = shared["min_flow_correlation"] = float(
                prompter.ask_number(
                    "Minimal flow correlation (0–1)?",
                    default=round(settings["min_flow_correlation"] / 2, 3),
                    minimum=0,
                    maximum=1,
                )
            )
        elif choice == "footprints":
            settings["alignment_source"] = "footprints"
        prompter.note(
            f"{plan.name}: loading again…"
            + (" (the new setting holds for the following sessions too)" if choice in ("rotation", "shift", "flow") else "")
        )


def _recipe_settings(session, settings):
    """Load options the loading recipe keeps for this session."""
    options = {"correct_rotation": bool(settings["correct_rotation"])}
    if settings.get("alignment_source"):
        options["alignment_source"] = settings["alignment_source"]
    return options


def load_sessions(tracking, plans, config, prompter):
    prompter.info(f"\n== Loading and aligning {len(plans)} session(s)")
    loaded, skipped = [], []
    expected = None
    shared = {}
    for index, plan in enumerate(plans, start=1):
        prompter.info(f"[{index}/{len(plans)}] {plan.path}")
        session = load_session(
            tracking, plan, config, prompter, expected_dims=expected, shared=shared
        )
        if session is None:
            prompter.note(f"{plan.name} left out.")
            skipped.append(plan)
            continue
        tracking.register_session(from_data=session)
        expected = expected or tuple(session.dims)
        loaded.append(plan)
    return loaded, skipped


def check_sessions(tracking, prompter):
    counts = np.array([session.n_neurons for session in tracking.sessions])
    if len(counts) < 2:
        raise Aborted(
            f"Matching needs at least two aligned sessions; {len(counts)} loaded."
        )
    median = float(np.median(counts))
    for session, count in zip(tracking.sessions, counts):
        if count < 0.25 * median:
            prompter.warn(
                f"{session.name}: {count} neurons, far fewer than the median "
                f"{median:.0f} (detection problem or partial field of view?)."
            )


# ----------------------------------------------------------------------
# Model and registration
# ----------------------------------------------------------------------


def fit_model(tracking, config, prompter):
    prompter.info("\n== Model")
    distances = tuple(sorted(set(config.session_distances)))
    started = time.perf_counter()
    tracking.process_model_updates(session_distances=distances)
    tracking.fit_model(session_distances=distances)
    model = tracking.model
    if not model.fitted:
        raise RuntimeError("The model could not be fitted.")
    parameters = model.parameters if isinstance(model.parameters, dict) else {}
    bad = [name for name, value in parameters.items() if np.ndim(value) == 0 and not np.isfinite(value)]
    if bad:
        raise RuntimeError(f"The model fit gave invalid parameters: {', '.join(bad)}.")
    shown = ", ".join(
        f"{name} {float(value):.3g}" for name, value in parameters.items() if np.ndim(value) == 0
    )
    prompter.note(
        f"{model.kind} model fitted from session pairs {distances} apart "
        f"({time.perf_counter() - started:.1f} s): {shown}."
    )


def register_neurons(tracking, prompter):
    prompter.info("\n== Registering neurons")
    started = time.perf_counter()
    result = tracking.build_assignment_update()
    if result is None:
        raise RuntimeError("No sessions to register.")
    tracking.publish_assignment_update(result)

    ids = np.asarray(tracking.assignments.ids)
    present = ids >= 0
    per_neuron = Counter(present.sum(axis=1))
    n_sessions = ids.shape[1]
    prompter.note(
        f"{ids.shape[0]} neurons over {n_sessions} sessions "
        f"({time.perf_counter() - started:.1f} s); in all sessions: {per_neuron.get(n_sessions, 0)}, "
        f"in one only: {per_neuron.get(1, 0)}."
    )
    for index, session in enumerate(tracking.sessions):
        matched = int((present[:, index] & (present.sum(axis=1) > 1)).sum())
        prompter.note(
            f"  {session.name}: {int(present[:, index].sum())} footprints assigned, "
            f"{matched} matched to another session."
        )


# ----------------------------------------------------------------------
# The run
# ----------------------------------------------------------------------


def project_path(output_directory, config, paths):
    name = config.project_name or Path(output_directory).parent.name or "catan"
    if Path(output_directory).name != "catan" and not config.project_name:
        name = Path(output_directory).name
    return Path(output_directory) / f"{name}.catan.hdf5"


def run_matching(*, paths=None, plans=None, output_directory, config, prompter):
    """Match the sessions ``paths`` (detected loading) or ``plans`` (from a
    loading recipe); returns a MatchingResult. Raises prompts.Aborted when
    stopped, prompts.InputRequired when a decision needs a terminal."""
    from catan import Tracking
    from catan.core.structures.loading_recipe import recipe_document
    from catan.tracking.project import project_files, save_project

    project = project_path(output_directory, config, paths or [p.path for p in plans])
    main, sessions_file, model_file = project_files(project)
    recipe_file = main.with_name(main.name[: -len(".catan.hdf5")] + ".recipe.json")
    existing = [p for p in (main, sessions_file, model_file, recipe_file) if p.exists()]
    if existing and not config.overwrite:
        replace_files = prompter.confirm(
            f"{main.parent} already holds this project ({', '.join(p.name for p in existing)}). Replace it?",
            option="--overwrite (or --project-name / --output)",
        )
        if not replace_files:
            raise Aborted("The existing project is kept.")

    pixel_size = config.pixel_size
    if pixel_size is None:
        pixel_size = prompter.ask_number(
            "Pixel size in µm per pixel?",
            minimum=1e-6,
            option="--pixel-size",
            details=(
                "Field of view width (µm) / image width (px). The model's imaging-slab "
                "thickness (10 µm) is converted with it; distances stay in pixels."
            ),
        )
        config = replace(config, pixel_size=float(pixel_size))

    if plans is None:
        plans = plan_sessions(list(paths), config, prompter)
    if len(plans) < 2:
        raise Aborted(f"Matching needs at least two sessions; {len(plans)} usable.")

    tracking = Tracking(
        neighbor_distance=config.neighbor_distance,
        n_threads=config.n_threads,
        um_per_px=float(pixel_size),
    )
    loaded, skipped = load_sessions(tracking, plans, config, prompter)
    check_sessions(tracking, prompter)
    fit_model(tracking, config, prompter)
    register_neurons(tracking, prompter)

    prompter.info("\n== Saving")
    recipe = recipe_document([plan.recipe_entry() for plan in loaded])
    save_project(tracking, main, recipe=recipe)
    recipe_file.write_text(json.dumps(recipe, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    for written in (main, sessions_file, model_file, recipe_file):
        prompter.note(str(written))

    return MatchingResult(
        project=main,
        sessions_file=sessions_file,
        model_file=model_file,
        recipe_file=recipe_file,
        sessions=[plan.name for plan in loaded],
        skipped=[plan.name for plan in skipped],
        warnings=list(prompter.warnings),
    )
