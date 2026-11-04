"""Headless matching (catan match) on a copy of the demo dataset."""

import io
import json
import shutil
from pathlib import Path

import numpy as np
import pytest

from catan.cli.detection import (
    choose_configs,
    choose_orientation,
    plan_backgrounds,
)
from catan.cli.matching import MatchingConfiguration, config_manager, run_matching
from catan.cli.pipeline import resolve_session_paths
from catan.cli.prompts import InputRequired, Prompter
from catan.core.structures.loading_recipe import parse_loading_recipe

DEMO = Path(__file__).resolve().parents[2] / "demo" / "catan_demo_ID29"


@pytest.fixture
def demo(tmp_path):
    target = tmp_path / "demo"
    shutil.copytree(DEMO, target, ignore=shutil.ignore_patterns("catan"))
    return target


def _paths(root):
    return resolve_session_paths(
        explicit_paths=[], root=root, glob_pattern="*/neuron_detection_*.hdf5", path_file=None
    )


def _quiet(**kwargs):
    return Prompter(stdout=io.StringIO(), **kwargs)


def test_prompter_without_terminal():
    prompter = _quiet(interactive=False)
    assert prompter.choose("Which?", [("a", "A"), ("b", "B")], default="b") == "b"
    with pytest.raises(InputRequired, match="--thing"):
        prompter.choose("Which?", [("a", "A"), ("b", "B")], option="--thing")


def test_prompter_in_terminal():
    prompter = _quiet(interactive=True, stdin=io.StringIO("x\n2\n\n0.5\n"))
    assert prompter.choose("Which?", [("a", "A"), ("b", "B")]) == "b"
    assert prompter.ask_number("Size?", default=1.0) == 1.0
    assert prompter.ask_number("Size?", maximum=1) == 0.5


def test_detection_on_demo(demo):
    paths = _paths(demo)
    prompter = _quiet(interactive=False)
    configs = choose_configs(paths, config_manager(), prompter)
    assert {config.name for config in configs.values()} == {"CaImAn"}

    plans = plan_backgrounds(configs, prompter)
    for path, (config, mode) in plans.items():
        background = config.groups["spatial"].fields["background"]
        assert mode == "configured"
        assert background.source_path == f"ID29_{path.parent.name}_Cn.tiff"

    first = paths[0]
    # The demo recipe says "transpose" (found by background coherence).
    assert choose_orientation(first, plans[first][0], prompter) == "transpose"


def test_unclear_background_needs_an_answer(demo):
    for folder in demo.glob("2024*"):
        image = next(folder.glob("*_Cn.tiff"))
        shutil.copy(image, folder / f"ID29_{folder.name}_mean.tiff")
    paths = _paths(demo)
    prompter = _quiet(interactive=False)
    configs = choose_configs(paths, config_manager(), prompter)
    # Several patterns cover all sessions: a default exists (by name), so
    # no question remains without a terminal.
    plans = plan_backgrounds(configs, prompter)
    assert all(
        config.groups["spatial"].fields["background"].source_path.endswith("_Cn.tiff")
        for config, _ in plans.values()
    )
    asked = _quiet(interactive=True, stdin=io.StringIO("2\n"))
    plans = plan_backgrounds(configs, asked)
    assert all(
        config.groups["spatial"].fields["background"].source_path.endswith("_mean.tiff")
        for config, _ in plans.values()
    )


def test_match_demo_writes_project(demo):
    prompter = _quiet(interactive=False)
    result = run_matching(
        paths=_paths(demo),
        output_directory=demo / "catan",
        config=MatchingConfiguration(pixel_size=1.0, n_threads=2),
        prompter=prompter,
    )
    assert result.project == demo / "catan" / "demo.catan.hdf5"
    for path in (result.project, result.sessions_file, result.model_file, result.recipe_file):
        assert path.is_file()
    assert result.sessions == ["20240719", "20240720", "20240721", "20240722"]

    from catan.tracking.structures.assignments import Assignments

    ids = np.asarray(Assignments._from_file(path=str(result.project)).ids)
    # 349 here; the model fit differs slightly with the BLAS kernels (CPU),
    # and S3 footprint 100 lies at the competitor threshold (p ≈ 0.30 vs
    # 0.3), so it is dropped on some CI runners (348).
    assert ids.shape[1] == 4
    assert abs(ids.shape[0] - 349) <= 3

    # The recipe reproduces the loading (background next to each file).
    recipe = json.loads(result.recipe_file.read_text())
    sessions = parse_loading_recipe(recipe, result.recipe_file.parent)
    assert [s.name for s in sessions] == result.sessions
    assert all(s._recipe_options["background_orientation"] == "transpose" for s in sessions)

    # An existing project is not replaced without asking.
    with pytest.raises(InputRequired, match="--overwrite"):
        run_matching(
            paths=_paths(demo),
            output_directory=demo / "catan",
            config=MatchingConfiguration(pixel_size=1.0),
            prompter=_quiet(interactive=False),
        )


def test_alignment_problem_needs_an_answer(demo):
    with pytest.raises(InputRequired, match="--on-issue"):
        run_matching(
            paths=_paths(demo),
            output_directory=demo / "catan",
            config=MatchingConfiguration(pixel_size=1.0, max_shift=5.0),
            prompter=_quiet(interactive=False),
        )
