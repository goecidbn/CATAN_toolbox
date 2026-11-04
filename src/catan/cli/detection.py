"""How to read the session files of ``catan match``.

Nothing is hard-coded: every public session preset (packaged and the user's
own, as in the GUI) is tested against the files' structure (metadata only);
the background comes from the preset, from an image next to the session
file or from the footprints; image dimensions and background orientation
are checked before loading. Whatever stays ambiguous is asked (prompts)."""

import re
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

# load_config before core.io.detection (their imports form a cycle).
from catan.core.structures.load_config import FieldSpec, LoadConfig
from catan.core.io.detection import IMAGE_SUFFIXES
from catan.core.io.inspection import check_fields_compatibility, inspect_file

from .prompts import Aborted

# Background images next to a session file, best first by name.
BACKGROUND_NAME_HINTS = ("cn", "corr", "mean", "avg", "max", "std", "background")
QUALITY_NAME_HINTS = {"snr_comp": "SNR", "cnn_preds": "CNN", "r_values": "r-values"}
FOOTPRINT_NAME_HINTS = ("a", "footprints", "spatial", "rois", "masks")


@dataclass
class SessionPlan:
    """How one session file is loaded (a loading recipe entry)."""

    path: Path
    name: str
    config: LoadConfig
    options: dict = field(default_factory=dict)

    def recipe_entry(self):
        return {
            "path": str(self.path),
            "name": self.name,
            "load_config": self.config.to_dict(),
            "load_options": dict(self.options),
            "active": True,
        }


# ----------------------------------------------------------------------
# Load presets
# ----------------------------------------------------------------------


def session_presets(manager):
    """Public session presets {name: LoadConfig}: packaged first, then the
    user's (as listed in the GUI)."""
    names = manager.builtin_names() + manager.user_names()
    return {
        name: manager.get(name)
        for name in names
        if manager.get(name).source_type == "session"
    }


def _required_and_optional(config):
    fields = config.get_fields_to_load()
    required, optional = {}, {}
    for group, specs in fields.items():
        for label, spec in specs.items():
            target = required if spec.required else optional
            target.setdefault(group, {})[label] = spec
    return required, optional


@dataclass
class PresetFit:
    name: str
    config: LoadConfig
    # path -> labels of required fields missing in it
    missing: dict = field(default_factory=dict)
    # path -> labels of optional fields found in it
    optional: dict = field(default_factory=dict)

    def fits(self, path):
        return not self.missing.get(path)

    @property
    def score(self):
        return sum(len(found) for found in self.optional.values())


def _available(path, fields):
    if not fields:
        return set(), set()
    report = check_fields_compatibility(path, fields)
    found = {f.label for f in report.fields if f.available}
    lost = {f.label for f in report.fields if not f.available}
    return found, lost


def fit_presets(paths, presets):
    """PresetFit per preset name, over all ``paths`` (file metadata only)."""
    fits = {name: PresetFit(name, config) for name, config in presets.items()}
    for path in paths:
        for name, config in presets.items():
            required, optional = _required_and_optional(config)
            if not required:
                fits[name].missing[path] = ["(no required fields)"]
                continue
            try:
                _, lost = _available(path, required)
                found, _ = _available(path, optional)
            except Exception as exc:
                fits[name].missing[path] = [f"unreadable: {type(exc).__name__}: {exc}"]
                continue
            fits[name].missing[path] = sorted(lost)
            fits[name].optional[path] = sorted(found)
    return fits


def _same_fields(a, b):
    def paths(config):
        return {
            (group, label, spec.path, spec.source, spec.attribute, spec.source_path)
            for group, specs in config.get_fields_to_load().items()
            for label, spec in specs.items()
        }

    return paths(a) == paths(b)


def choose_configs(paths, manager, prompter, *, requested=None):
    """{path: LoadConfig} for all ``paths`` (files nothing fits are dropped
    after asking). ``requested``: a preset name or a load-config JSON file
    (--load-config) — then only that one is tested."""
    if requested is not None:
        if Path(requested).suffix.lower() == ".json" and Path(requested).is_file():
            config = LoadConfig.load_json(requested)
            presets = {config.name or Path(requested).stem: config}
        else:
            presets = session_presets(manager)
            if requested not in presets:
                raise ValueError(
                    f"Unknown session preset {requested!r}. Known: "
                    + ", ".join(presets)
                    + ". Or give a load-config JSON file."
                )
            presets = {requested: presets[requested]}
    else:
        presets = session_presets(manager)
        if not presets:
            raise ValueError("No session load presets are available.")

    prompter.info(f"Testing {len(presets)} load preset(s) against {len(paths)} file(s)…")
    fits = fit_presets(paths, presets)

    # Best fitting preset per file: all required fields, most optional ones.
    order = list(presets)
    chosen = {}
    unfit = []
    for path in paths:
        candidates = [fit for fit in fits.values() if fit.fits(path)]
        if not candidates:
            unfit.append(path)
            continue
        best = max(len(fit.optional.get(path, ())) for fit in candidates)
        top = [fit for fit in candidates if len(fit.optional.get(path, ())) == best]
        chosen[path] = top

    # One decision for all files with the same set of best presets.
    decided = {}
    for path, top in chosen.items():
        key = tuple(fit.name for fit in top)
        if key not in decided:
            distinct = []
            for fit in top:
                if not any(_same_fields(fit.config, other.config) for other in distinct):
                    distinct.append(fit)
            if len(distinct) == 1:
                decided[key] = distinct[0]
            else:
                files = [p for p, t in chosen.items() if tuple(f.name for f in t) == key]
                choice = prompter.choose(
                    f"{len(distinct)} load presets read {len(files)} file(s) equally well "
                    f"(e.g. {files[0].name}). Which one should be used?",
                    [(fit.name, _describe_preset(fit, files)) for fit in distinct],
                    default=min(distinct, key=lambda fit: order.index(fit.name)).name,
                    option="--load-config NAME",
                )
                decided[key] = fits[choice]
        chosen[path] = decided[key]

    configs = {path: fit.config for path, fit in chosen.items()}

    used = {}
    for path, fit in chosen.items():
        used.setdefault(fit.name, []).append(path)
    for name, files in used.items():
        fit = fits[name]
        optional = sorted({label for p in files for label in fit.optional.get(p, ())})
        _, all_optional = _required_and_optional(fit.config)
        missing_optional = sorted(
            {label for specs in all_optional.values() for label in specs} - set(optional)
        )
        prompter.note(
            f"Preset '{name}' for {len(files)} file(s)"
            + (f"; optional fields found: {', '.join(optional)}" if optional else "")
            + (f"; not found: {', '.join(missing_optional)}" if missing_optional else "")
            + "."
        )

    for path in unfit:
        config = _config_for_unfit(path, fits, prompter, n_files=len(paths))
        if config is not None:
            configs[path] = config

    return {path: configs[path] for path in paths if path in configs}


def _describe_preset(fit, files):
    optional = sorted({label for p in files for label in fit.optional.get(p, ())})
    footprints = fit.config.groups["spatial"].fields.get("footprints")
    return (
        f"{fit.name} (footprints {footprints.path if footprints else '?'}"
        + (f"; optional: {', '.join(optional)}" if optional else "")
        + ")"
    )


def _config_for_unfit(path, fits, prompter, *, n_files):
    reasons = "\n".join(
        f"- {fit.name}: missing {', '.join(fit.missing.get(path) or ['?'])}"
        for fit in fits.values()
    )
    candidates = footprint_candidates(path)
    choices = [
        (("field", info.path), f"Read footprints from {info.path} (shape {info.shape}, {info.shape[1]} neurons)")
        for info in candidates
    ]
    choices += [("skip", "Leave this file out"), ("abort", "Stop")]
    choice = prompter.choose(
        f"No load preset fits {path}.",
        choices,
        # Without a terminal: one obvious footprint field is used.
        default=choices[0][0] if len(candidates) == 1 else None,
        option="--load-config (a preset name or a JSON file saved in the GUI)",
        details="Presets tested:\n" + reasons,
    )
    if choice == "abort":
        raise Aborted(f"No load configuration for {path}.")
    if choice == "skip":
        prompter.warn(f"{path} left out: no load preset fits.")
        return None
    return config_from_structure(path, choice[1], prompter)


# ----------------------------------------------------------------------
# A configuration from the file structure
# ----------------------------------------------------------------------


def footprint_candidates(path):
    """Fields that look like footprints (pixels × neurons), best first."""
    structure = inspect_file(path)
    out = []
    for info in structure.entries.values():
        shape = info.shape or ()
        if info.kind != "field" or len(shape) != 2:
            continue
        pixels, neurons = shape
        if pixels < 32 * 32 or not 0 < neurons < pixels:
            continue
        out.append(info)

    def rank(info):
        name = info.name.lower()
        hint = FOOTPRINT_NAME_HINTS.index(name) if name in FOOTPRINT_NAME_HINTS else len(FOOTPRINT_NAME_HINTS)
        return (hint, info.path)

    return sorted(out, key=rank)


def config_from_structure(path, footprint_path, prompter):
    """Session LoadConfig reading footprints from ``footprint_path``, with
    the quality fields and a background (pixels or image) found next to it."""
    structure = inspect_file(path)
    footprint = structure.entries[footprint_path]
    pixels, neurons = footprint.shape
    folder = footprint_path.rsplit("/", 1)[0] or "/"

    def spec(info):
        return FieldSpec(path=info.path, source=info.kind, required=False)

    quality = {}
    background = None
    for info in structure.entries.values():
        shape = tuple(info.shape or ())
        parent = info.path.rsplit("/", 1)[0] or "/"
        if parent != folder:
            continue
        label = QUALITY_NAME_HINTS.get(info.name.lower())
        if label and shape == (neurons,):
            quality[label] = spec(info)
        if background is None and len(shape) == 2 and int(np.prod(shape)) == pixels and (
            info.name.lower() in BACKGROUND_NAME_HINTS
        ):
            background = spec(info)

    groups = {
        "spatial": {
            "title": "Spatial data",
            "type": "static",
            "enabled": True,
            "fields": {
                "footprints": FieldSpec(path=footprint_path, source=footprint.kind, required=True).to_dict(),
                **({"background": background.to_dict()} if background else {}),
            },
        },
        "quality": {
            "title": "Quality data",
            "type": "dynamic",
            "enabled": True,
            "fields": {label: s.to_dict() for label, s in quality.items()},
        },
    }
    config = LoadConfig.from_dict(
        {
            "format_version": 1,
            "source_type": "session",
            "public": True,
            "native": False,
            "groups": groups,
            "subconfigs": {},
            "name": f"detected ({Path(path).name})",
        }
    )
    prompter.note(
        f"Configuration from the file: footprints {footprint_path}"
        + (f", background {background.path}" if background else "")
        + (f", quality {', '.join(quality)}" if quality else ", no quality fields")
        + "."
    )
    return config


# ----------------------------------------------------------------------
# Backgrounds
# ----------------------------------------------------------------------


def configured_background_available(path, config):
    background = config.groups.get("spatial") and config.groups["spatial"].fields.get("background")
    if background is None:
        return False
    try:
        found, _ = _available(path, {"spatial": {"background": background}})
    except Exception:
        return False
    return "background" in found


def image_files(path):
    folder = Path(path).parent
    return sorted(
        p for p in folder.iterdir()
        if p.is_file() and p.suffix.lower() in IMAGE_SUFFIXES
    )


def _name_pattern(name):
    """Pattern of a file name with its numbers as wildcards."""
    return re.sub(r"\d+", "*", name)


def _hint_rank(name):
    lower = name.lower()
    for index, hint in enumerate(BACKGROUND_NAME_HINTS):
        if re.search(rf"(^|[^a-z]){hint}([^a-z]|$)", lower):
            return index
    return len(BACKGROUND_NAME_HINTS)


def image_background(config, image):
    """Copy of ``config`` with the background read from ``image`` (next to
    the session file; relative path as in the GUI's loading recipes)."""
    config = config.copy_for_session()
    # Set directly: the spatial group is static (add_field refuses).
    config.groups["spatial"].fields["background"] = FieldSpec(
        path="/image", source="dataset", required=False, source_path=image.name
    )
    config.dimensions = {"mode": "image", "field": None}
    return config


def without_background(config, dims_spec):
    config = config.copy_for_session()
    config.groups["spatial"].fields.pop("background", None)
    config.dimensions = dims_spec
    return config


def plan_backgrounds(configs, prompter, *, requested=None):
    """{path: (config, background_mode)}: the configured background where
    the file has it, else an image next to the file, else the footprints.
    ``requested``: "configured", "footprints" or a file pattern (glob in the
    session folder), --background."""
    paths = list(configs)
    plans = {}
    missing = []
    for path in paths:
        if requested in (None, "configured") and configured_background_available(path, configs[path]):
            config = configs[path].copy_for_session()
            config.dimensions = {"mode": "image", "field": None}
            plans[path] = (config, "configured")
        elif requested == "configured":
            raise ValueError(f"{path} has no configured background (--background configured).")
        else:
            missing.append(path)

    if plans and missing and requested is None:
        prompter.note(
            f"{len(plans)} file(s) hold their background; {len(missing)} do not."
        )
    if not missing:
        if plans:
            prompter.note("Background: as configured in the preset.")
        return plans

    if requested == "footprints":
        for path in missing:
            plans[path] = _footprint_plan(path, configs[path], prompter)
        return plans

    images = {path: image_files(path) for path in missing}

    if requested not in (None, "configured", "footprints"):
        for path in missing:
            matches = sorted(Path(path).parent.glob(requested))
            if len(matches) != 1:
                raise ValueError(
                    f"--background {requested!r} matches {len(matches)} files next to {path}."
                )
            plans[path] = (image_background(configs[path], matches[0]), "configured")
        prompter.note(f"Background: {requested} next to each session file.")
        return plans

    # A name pattern found exactly once next to every remaining file.
    patterns = {}
    for path, files in images.items():
        for image in files:
            patterns.setdefault(_name_pattern(image.name), {}).setdefault(path, []).append(image)
    covering = sorted(
        (
            pattern for pattern, found in patterns.items()
            if len(found) == len(missing) and all(len(v) == 1 for v in found.values())
        ),
        key=lambda pattern: (_hint_rank(pattern), pattern),
    )

    if covering:
        options = [(pattern, f"Image {pattern} next to each session file") for pattern in covering]
        options.append(("footprints", "Construct the background from the footprints"))
        pattern = (
            covering[0]
            if len(covering) == 1
            else prompter.choose(
                f"Several images next to the {len(missing)} session file(s) could be their background.",
                options,
                default=covering[0],
                option="--background PATTERN",
                details=f"For example next to {missing[0]}: "
                + ", ".join(p.name for p in images[missing[0]]),
            )
        )
        if pattern == "footprints":
            for path in missing:
                plans[path] = _footprint_plan(path, configs[path], prompter)
            return plans
        prompter.note(
            f"Background: {pattern} next to each session file "
            f"(e.g. {patterns[pattern][missing[0]][0].name})."
        )
        for path in missing:
            plans[path] = (image_background(configs[path], patterns[pattern][path][0]), "configured")
        return plans

    # Nothing common: per file.
    for path in missing:
        files = images[path]
        if not files:
            prompter.note(f"{Path(path).name}: no background in the file and no image next to it.")
            plans[path] = _footprint_plan(path, configs[path], prompter)
            continue
        choices = [(str(image), f"Image {image.name}") for image in files]
        choices.append(("footprints", "Construct the background from the footprints"))
        choice = prompter.choose(
            f"Which background for {path}?",
            choices,
            default=choices[0][0] if len(files) == 1 else None,
            option="--background PATTERN|footprints",
        )
        if choice == "footprints":
            plans[path] = _footprint_plan(path, configs[path], prompter)
        else:
            plans[path] = (image_background(configs[path], Path(choice)), "configured")
    return plans


def _footprint_plan(path, config, prompter):
    return without_background(config, dims_source(path, config, prompter)), "footprints"


def dims_source(path, config, prompter):
    """Dimension source for footprints without a background: a two-value
    "dims" field whose product is the pixel count, else a square image,
    else asked."""
    footprints = config.groups["spatial"].fields["footprints"]
    structure = inspect_file(path)
    info = structure.entries.get(footprints.path)
    pixels = int(info.shape[0]) if info is not None and info.shape else None

    for entry in structure.entries.values():
        if entry.name.lower() in {"dims", "dimensions", "image_dims"} and tuple(entry.shape or ()) == (2,):
            return {
                "mode": "values",
                "field": FieldSpec(path=entry.path, source=entry.kind, required=True).to_dict(),
            }

    if pixels is not None:
        side = int(round(np.sqrt(pixels)))
        if side * side == pixels:
            prompter.note(f"{Path(path).name}: no dimensions stored; assuming a square {side} × {side} image.")
            return {"mode": "manual", "height": side, "width": side, "field": None}

    height = prompter.ask_number(
        f"{path} stores no image dimensions and {pixels} pixels are not a square. Image height (px)?",
        minimum=1,
        integer=True,
        option="a background (--background) or a preset with a dims field",
    )
    if pixels is not None and pixels % height:
        raise ValueError(f"{pixels} pixels cannot form an image of height {height}.")
    return {"mode": "manual", "height": height, "width": pixels // height, "field": None}


# ----------------------------------------------------------------------
# Orientation of the background
# ----------------------------------------------------------------------


def orientation_scores(path, config):
    """{orientation: background_coherence} for the orientations whose shape
    fits the footprints (footprint_frame: does the background resemble the
    footprints?)."""
    from scipy import sparse

    from catan.core import footprint_frame
    from catan.core.io.api import load_fields_from_sources
    from catan.core.structures.session import SessionData

    spatial = config.groups["spatial"].fields
    data = load_fields_from_sources(
        path,
        {"spatial": {"footprints": spatial["footprints"], "background": spatial["background"]}},
    )["spatial"]
    footprints = sparse.csc_matrix(data["footprints"])
    background = SessionData._background_as_grayscale(data["background"])
    projection = np.asarray(footprints.sum(axis=1), dtype=np.float32).ravel()

    scores = {}
    for orientation, image in (("as_stored", background), ("transpose", background.T)):
        if image.size != projection.size:
            continue
        scores[orientation] = footprint_frame.background_coherence(
            image, projection.reshape(image.shape)
        )
    return scores


def choose_orientation(path, config, prompter):
    """Background orientation for the sessions: the one whose background
    matches the footprints; asked if none or both do."""
    scores = orientation_scores(path, config)
    if len(scores) == 1:
        orientation = next(iter(scores))
        prompter.note(f"Background orientation: {orientation.replace('_', ' ')} (only shape that fits).")
        return orientation

    def describe(name):
        score = scores[name]
        return (
            f"{name.replace('_', ' ')}: "
            + ("matches" if score.get("coherent") else "does not match")
            + f" the footprints (z {score.get('zscore', float('nan')):.1f}, "
            f"correlation {score.get('correlation', float('nan')):.2f})"
        )

    coherent = [name for name, score in scores.items() if score.get("coherent")]
    if len(coherent) == 1:
        prompter.note(f"Background orientation: {describe(coherent[0])}.")
        return coherent[0]

    best = max(scores, key=lambda name: scores[name].get("zscore") or -np.inf)
    return prompter.choose(
        f"The background orientation of {Path(path).name} is unclear "
        + ("(both orientations match the footprints)." if coherent else "(neither orientation matches the footprints)."),
        [(name, describe(name)) for name in scores]
        + [("auto", "Decide per session while loading")],
        default=None,
        option="--orientation as_stored|transpose",
        details=(
            "If neither matches, the background may belong to another field of view or "
            "need another image; 'footprints' (--background footprints) avoids it."
            if not coherent
            else f"Best score: {best.replace('_', ' ')}."
        ),
    )


# ----------------------------------------------------------------------
# Image geometry
# ----------------------------------------------------------------------


def check_geometry(path, config, orientation, expected_dims=None):
    """(ok, dims, message) from the file metadata (spatial_geometry)."""
    from catan.core.spatial_geometry import inspect_dimensions

    spatial = {
        label: spec
        for label, spec in config.groups["spatial"].fields.items()
        if label in ("footprints", "background")
    }
    result = inspect_dimensions(
        path, {"spatial": spatial}, config.dimensions, orientation, expected_dims
    )
    return result["ok"], result["dims"], result["message"]
