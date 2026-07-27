from __future__ import annotations

import argparse
from collections.abc import Sequence

from catan import __version__

from .match import add_match_parser


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="catan",
        description=(
            "CATAN: matching and curation tools for longitudinal "
            "calcium-imaging data."
        ),
    )

    parser.add_argument(
        "--version",
        action="version",
        version=f"%(prog)s {__version__}",
    )

    subparsers = parser.add_subparsers(
        dest="command",
        metavar="COMMAND",
    )

    add_match_parser(subparsers)

    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    handler = getattr(args, "handler", None)

    if handler is None:
        parser.print_help()
        return 0

    return int(handler(args))


if __name__ == "__main__":
    raise SystemExit(main())