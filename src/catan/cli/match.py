from __future__ import annotations

import argparse
import logging
import sys
from collections.abc import Sequence
from pathlib import Path



def add_match_parser(
    subparsers: argparse._SubParsersAction,
) -> argparse.ArgumentParser:
    parser = subparsers.add_parser(
        "match",
        help="Match neurons across multiple imaging sessions.",
        description=(
            "Load calcium-imaging sessions, align them, fit the CATAN model, "
            "register the neurons and save a CATAN project (<name>.catan.hdf5 "
            "with its sessions, model and loading recipe) that the GUI opens. "
            "How the files are read is detected; unclear points are asked "
            "(or, without a terminal, answered by the options below)."
        ),
    )

    parser.add_argument(
        "sessions",
        nargs="*",
        type=Path,
        metavar="SESSION",
        help=(
            "Explicit session paths. This cannot be combined with "
            "--root or --path-file."
        ),
    )

    parser.add_argument(
        "--root",
        type=Path,
        help=(
            "Root directory under which session files or directories "
            "are discovered."
        ),
    )

    parser.add_argument(
        "--glob",
        dest="glob_pattern",
        help=(
            "Glob pattern evaluated relative to --root, for example "
            "'Session*/results.hdf5'."
        ),
    )

    parser.add_argument(
        "--path-file",
        type=Path,
        help=(
            "Text or CSV file containing session paths. Relative paths "
            "are interpreted relative to the path file."
        ),
    )

    parser.add_argument(
        "--output",
        type=Path,
        help=(
            "Directory in which model and registration results are saved. "
            "Defaults to a 'matching' directory beneath --root or the "
            "common parent of all session paths."
        ),
    )

    parser.add_argument(
        "--recipe",
        type=Path,
        help=(
            "Loading recipe (.json, saved by the GUI or by an earlier run): "
            "sessions with their load configuration; skips the detection."
        ),
    )

    parser.add_argument(
        "--pixel-size",
        type=float,
        default=None,
        help="Pixel size in µm per pixel (asked if not given).",
    )

    parser.add_argument(
        "--load-config",
        metavar="NAME|JSON",
        help=(
            "Load preset (name as in the GUI) or load-config JSON file. "
            "Default: detected from the files."
        ),
    )

    parser.add_argument(
        "--background",
        metavar="configured|footprints|PATTERN",
        help=(
            "Background: from the preset, constructed from the footprints, "
            "or an image next to each session file (glob, e.g. '*_Cn.tiff'). "
            "Default: detected."
        ),
    )

    parser.add_argument(
        "--orientation",
        choices=("auto", "as_stored", "transpose"),
        help="Background orientation. Default: detected from the first session.",
    )

    parser.add_argument(
        "--max-shift",
        type=float,
        default=50.0,
        help="Largest accepted shift between sessions (px). Default: %(default)s",
    )

    parser.add_argument(
        "--min-flow-correlation",
        type=float,
        default=0.10,
        help="Background correlation a flow alignment needs. Default: %(default)s",
    )

    parser.add_argument(
        "--correct-rotation",
        action="store_true",
        help="Also correct rotations between sessions.",
    )

    parser.add_argument(
        "--refine-with-footprints",
        action="store_true",
        help="Correct each alignment by the footprint positions.",
    )

    parser.add_argument(
        "--session-distances",
        type=int,
        nargs="+",
        default=[1],
        metavar="N",
        help="Model evidence from session pairs N apart. Default: 1",
    )

    parser.add_argument(
        "--neighbor-distance",
        type=float,
        default=25.0,
        help="Maximum neighbour distance in pixels. Default: %(default)s",
    )

    parser.add_argument(
        "--threads",
        type=int,
        default=1,
        help="Number of worker threads. Default: %(default)s",
    )

    parser.add_argument(
        "--on-issue",
        choices=("ask", "accept", "skip", "abort"),
        default="ask",
        help=(
            "Sessions with problems (alignment failed or deviating, file not "
            "loadable): ask (needs a terminal), accept deviating alignments "
            "(others are left out), skip (leave out) or abort. Default: ask."
        ),
    )

    parser.add_argument(
        "--non-interactive",
        action="store_true",
        help=(
            "Never ask: decisions without a default stop the run with a "
            "message naming the option that answers them."
        ),
    )

    parser.add_argument(
        "--project-name",
        help="Project name (<name>.catan.hdf5). Default: the data folder's name.",
    )

    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace an existing project of the same name.",
    )

    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print resolved paths and settings without running matching.",
    )

    parser.add_argument(
        "-v",
        "--verbose",
        action="count",
        default=0,
        help="Increase logging verbosity. May be supplied more than once.",
    )

    parser.set_defaults(handler=run_match_command)

    return parser


def run_match_command(args: argparse.Namespace) -> int:
    configure_logging(args.verbose)

    # Lazy import: `catan match --help` does not need to import NumPy,
    # HDF5, the tracking model, or any GUI-related modules.
    from .matching import MatchingConfiguration, plans_from_recipe, run_matching
    from .pipeline import resolve_output_directory, resolve_session_paths
    from .prompts import Aborted, InputRequired, Prompter

    plans = None
    try:
        if args.recipe is not None:
            if args.sessions or args.root or args.path_file:
                raise ValueError("--recipe cannot be combined with other session inputs.")
            plans = plans_from_recipe(args.recipe)
            paths = [plan.path for plan in plans]
        else:
            paths = resolve_session_paths(
                explicit_paths=args.sessions,
                root=args.root,
                glob_pattern=args.glob_pattern,
                path_file=args.path_file,
            )

        output_directory = resolve_output_directory(
            paths=paths,
            explicit_output=args.output,
            root=args.root,
        )
    except (FileNotFoundError, NotADirectoryError, ValueError, KeyError) as exc:
        raise SystemExit(f"catan match: error: {exc}") from exc

    try:
        config = MatchingConfiguration(
            pixel_size=args.pixel_size,
            n_threads=args.threads,
            neighbor_distance=args.neighbor_distance,
            max_shift=args.max_shift,
            min_flow_correlation=args.min_flow_correlation,
            correct_rotation=args.correct_rotation,
            refine_with_footprints=args.refine_with_footprints,
            session_distances=tuple(args.session_distances),
            load_config=args.load_config,
            background=args.background,
            orientation=args.orientation,
            on_issue=args.on_issue,
            project_name=args.project_name,
            overwrite=args.overwrite,
        )
    except ValueError as exc:
        raise SystemExit(f"catan match: error: {exc}") from exc

    print_configuration(
        paths=paths,
        output_directory=output_directory,
        config=config,
        recipe=args.recipe,
    )

    if args.dry_run:
        return 0

    prompter = Prompter(interactive=False if args.non_interactive else None)
    try:
        result = run_matching(
            paths=None if plans is not None else paths,
            plans=plans,
            output_directory=output_directory,
            config=config,
            prompter=prompter,
        )
    except InputRequired as exc:
        print(f"\ncatan match: a decision is needed:\n{exc}", file=sys.stderr)
        return 2
    except Aborted as exc:
        print(f"\ncatan match: stopped: {exc}", file=sys.stderr)
        return 1

    print()
    print(f"Project saved: {result.project}")
    print("Open it in the GUI with File ▸ Open project.")
    if result.skipped:
        print(f"Left out: {', '.join(result.skipped)}")
    if result.warnings:
        print(f"{len(result.warnings)} warning(s):")
        for warning in result.warnings:
            print("  - " + warning.splitlines()[0])
    return 0


def configure_logging(verbosity: int) -> None:
    if verbosity >= 2:
        level = logging.DEBUG
    elif verbosity == 1:
        level = logging.INFO
    else:
        level = logging.WARNING

    logging.basicConfig(
        level=level,
        format="%(levelname)s: %(message)s",
    )


def print_configuration(*, paths, output_directory, config, recipe=None) -> None:
    print(f"Found {len(paths)} sessions:")
    for index, path in enumerate(paths, start=1):
        print(f"  {index:>3}: {path}")
    print()
    print(f"Output directory: {output_directory}")
    print()
    print("Matching configuration:")
    detected = f"from recipe {recipe.name}" if recipe is not None else "detected"
    print(f"  pixel size:        {config.pixel_size if config.pixel_size else 'asked'} µm/px")
    print(f"  load preset:       {config.load_config or detected}")
    print(f"  background:        {config.background or detected}")
    print(f"  orientation:       {config.orientation or detected}")
    print(f"  max shift:         {config.max_shift} px")
    print(f"  min flow corr.:    {config.min_flow_correlation}")
    print(f"  correct rotation:  {config.correct_rotation}")
    print(f"  refine (fp):       {config.refine_with_footprints}")
    print(f"  session distances: {', '.join(map(str, config.session_distances))}")
    print(f"  neighbor distance: {config.neighbor_distance} px")
    print(f"  threads:           {config.n_threads}")
    print(f"  on issue:          {config.on_issue}")


def standalone_main(
    argv: Sequence[str] | None = None,
) -> int:
    from .main import main

    arguments = ["match"]

    if argv is not None:
        arguments.extend(argv)

    return main(arguments)