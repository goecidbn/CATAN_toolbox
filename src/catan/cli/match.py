from __future__ import annotations

import argparse
import logging
from collections.abc import Sequence
from pathlib import Path

from .pipeline import MatchingConfiguration


def add_match_parser(
    subparsers: argparse._SubParsersAction,
) -> argparse.ArgumentParser:
    parser = subparsers.add_parser(
        "match",
        help="Match neurons across multiple imaging sessions.",
        description=(
            "Load multiple calcium-imaging sessions, fit the CATAN "
            "matching model, register neurons, and save the results."
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
        "--neighbor-distance",
        type=float,
        default=25.0,
        help="Maximum neighbour distance in micrometres. Default: %(default)s",
    )

    parser.add_argument(
        "--bins",
        type=int,
        default=64,
        help="Number of model bins. Default: %(default)s",
    )

    parser.add_argument(
        "--threads",
        type=int,
        default=1,
        help="Number of worker threads. Default: %(default)s",
    )

    parser.add_argument(
        "--pixel-size",
        type=float,
        default=536.0 / 512.0,
        help=(
            "Pixel size in micrometres per pixel. "
            "Default: %(default).8g"
        ),
    )

    parser.add_argument(
        "--kde",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Enable or disable KDE model fitting. Default: disabled.",
    )

    parser.add_argument(
        "--align",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Align sessions while loading. Default: enabled.",
    )

    parser.add_argument(
        "--use-cdf",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Use the CDF during model fitting. Default: enabled.",
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
    from .pipeline import (
        MatchingConfiguration,
        resolve_output_directory,
        resolve_session_paths,
        run_matching,
    )

    try:
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
    except (FileNotFoundError, NotADirectoryError, ValueError) as exc:
        raise SystemExit(f"catan match: error: {exc}") from exc

    config = MatchingConfiguration(
        neighbor_distance=args.neighbor_distance,
        bins=args.bins,
        n_threads=args.threads,
        use_kde=args.kde,
        pixel_size=args.pixel_size,
        align=args.align,
        use_cdf=args.use_cdf,
    )

    print_configuration(
        paths=paths,
        output_directory=output_directory,
        config=config,
    )

    if args.dry_run:
        return 0

    run_matching(
        paths=paths,
        output_directory=output_directory,
        config=config,
    )

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


def print_configuration(
    *,
    paths: Sequence[Path],
    output_directory: Path,
    config: MatchingConfiguration,
) -> None:
    print(f"Found {len(paths)} sessions:")

    for index, path in enumerate(paths, start=1):
        print(f"  {index:>3}: {path}")

    print()
    print(f"Output directory: {output_directory}")
    print()
    print("Matching configuration:")
    print(f"  neighbor distance: {config.neighbor_distance}")
    print(f"  bins:              {config.bins}")
    print(f"  threads:           {config.n_threads}")
    print(f"  KDE:               {config.use_kde}")
    print(f"  pixel size:        {config.pixel_size}")
    print(f"  alignment:         {config.align}")
    print(f"  use CDF:           {config.use_cdf}")


def standalone_main(
    argv: Sequence[str] | None = None,
) -> int:
    from .main import main

    arguments = ["match"]

    if argv is not None:
        arguments.extend(argv)

    return main(arguments)