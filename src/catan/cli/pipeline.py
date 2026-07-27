from __future__ import annotations

import csv
import logging
import os
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

from tqdm.auto import tqdm


from catan import Tracking

logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class MatchingConfiguration:
    neighbor_distance: float = 25.0
    bins: int = 64
    n_threads: int = 1
    use_kde: bool = False
    pixel_size: float = 536.0 / 512.0
    align: bool = True
    use_cdf: bool = True

    def __post_init__(self) -> None:
        if self.neighbor_distance <= 0:
            raise ValueError("neighbor_distance must be greater than zero.")

        if self.bins < 1:
            raise ValueError("bins must be at least 1.")

        if self.n_threads < 1:
            raise ValueError("n_threads must be at least 1.")

        if self.pixel_size <= 0:
            raise ValueError("pixel_size must be greater than zero.")


def resolve_session_paths(
    *,
    explicit_paths: Sequence[Path],
    root: Path | None,
    glob_pattern: str | None,
    path_file: Path | None,
) -> list[Path]:
    """Resolve session paths from exactly one input mode."""

    uses_explicit_paths = bool(explicit_paths)
    uses_root = root is not None or glob_pattern is not None
    uses_path_file = path_file is not None

    mode_count = sum(
        (
            uses_explicit_paths,
            uses_root,
            uses_path_file,
        )
    )

    if mode_count == 0:
        raise ValueError(
            "No sessions were supplied. Provide explicit SESSION paths, "
            "--root together with --glob, or --path-file."
        )

    if mode_count > 1:
        raise ValueError(
            "Session input modes cannot be combined. Use either explicit "
            "SESSION paths, --root/--glob, or --path-file."
        )

    if uses_root:
        if root is None or glob_pattern is None:
            raise ValueError("--root and --glob must be supplied together.")

        root = root.expanduser().resolve()

        if not root.exists():
            raise FileNotFoundError(f"Root path does not exist: {root}")

        if not root.is_dir():
            raise NotADirectoryError(f"Root path is not a directory: {root}")

        paths = sorted(root.glob(glob_pattern))

    elif uses_path_file:
        paths = read_path_file(path_file)

    else:
        paths = list(explicit_paths)

    paths = [
        path.expanduser().resolve()
        for path in paths
    ]

    if not paths:
        if root is not None and glob_pattern is not None:
            raise ValueError(
                f"No session paths matched {glob_pattern!r} beneath {root}."
            )

        raise ValueError("No session paths were found.")

    duplicate_paths = find_duplicates(paths)

    if duplicate_paths:
        duplicates = ", ".join(str(path) for path in duplicate_paths)
        raise ValueError(f"Duplicate session paths were supplied: {duplicates}")

    missing_paths = [path for path in paths if not path.exists()]

    if missing_paths:
        formatted = "\n".join(f"  - {path}" for path in missing_paths)
        raise FileNotFoundError(
            f"The following session paths do not exist:\n{formatted}"
        )

    return paths


def read_path_file(path_file: Path) -> list[Path]:
    """Read session paths from a plain-text or CSV file."""

    path_file = path_file.expanduser().resolve()

    if not path_file.exists():
        raise FileNotFoundError(f"Path file does not exist: {path_file}")

    if not path_file.is_file():
        raise ValueError(f"Path file is not a regular file: {path_file}")

    suffix = path_file.suffix.lower()

    if suffix == ".csv":
        raw_paths = read_csv_paths(path_file)
    else:
        raw_paths = read_text_paths(path_file)

    resolved_paths: list[Path] = []

    for raw_path in raw_paths:
        candidate = Path(raw_path).expanduser()

        if not candidate.is_absolute():
            candidate = path_file.parent / candidate

        resolved_paths.append(candidate)

    return resolved_paths


def read_text_paths(path_file: Path) -> list[str]:
    paths: list[str] = []

    with path_file.open("r", encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            value = line.strip()

            if not value or value.startswith("#"):
                continue

            paths.append(value)

    return paths


def read_csv_paths(path_file: Path) -> list[str]:
    with path_file.open(
        "r",
        encoding="utf-8-sig",
        newline="",
    ) as file:
        rows = list(csv.reader(file))

    rows = [
        row
        for row in rows
        if row and any(cell.strip() for cell in row)
    ]

    if not rows:
        return []

    header = [cell.strip().lower() for cell in rows[0]]

    if "path" in header:
        path_column = header.index("path")
        data_rows = rows[1:]
    else:
        path_column = 0
        data_rows = rows

    paths: list[str] = []

    for row_number, row in enumerate(data_rows, start=2):
        if len(row) <= path_column:
            raise ValueError(
                f"CSV row {row_number} has no path column."
            )

        value = row[path_column].strip()

        if not value or value.startswith("#"):
            continue

        paths.append(value)

    return paths


def find_duplicates(paths: Sequence[Path]) -> list[Path]:
    seen: set[Path] = set()
    duplicates: list[Path] = []

    for path in paths:
        if path in seen and path not in duplicates:
            duplicates.append(path)

        seen.add(path)

    return duplicates


def resolve_output_directory(
    *,
    paths: Sequence[Path],
    explicit_output: Path | None,
    root: Path | None,
) -> Path:
    if explicit_output is not None:
        return explicit_output.expanduser().resolve()

    if root is not None:
        return root.expanduser().resolve() / "matching"

    common_parent = determine_common_parent(paths)
    return common_parent / "matching"


def determine_common_parent(paths: Sequence[Path]) -> Path:
    if not paths:
        raise ValueError(
            "At least one path is needed to determine an output directory."
        )

    # For files, use the parent. For session directories, use the
    # directory itself. This also gives a useful result for a single file.
    candidates = [
        path if path.is_dir() else path.parent
        for path in paths
    ]

    try:
        common_path = os.path.commonpath(
            [str(path) for path in candidates]
        )
    except ValueError as exc:
        # This can occur on Windows when paths are on different drives.
        raise ValueError(
            "The supplied sessions do not have a common parent path. "
            "Please provide --output explicitly."
        ) from exc

    return Path(common_path)


def run_matching(
    *,
    paths: Sequence[Path],
    output_directory: Path,
    config: MatchingConfiguration,
) -> None:
    """Run CATAN's complete matching pipeline."""

    # Import here rather than at module import time so path resolution and
    # `--dry-run` remain lightweight.

    output_directory.mkdir(
        parents=True,
        exist_ok=True,
    )

    logger.info("Writing matching results to %s", output_directory)

    tracking = Tracking(
        neighbor_distance=config.neighbor_distance,
        bins=config.bins,
        n_threads=config.n_threads,
        use_kde=config.use_kde,
        pxtomu=config.pixel_size,
    )

    path_iteration = tqdm(
        paths,
        desc="Registering sessions",
        unit="session",
    )

    for index, path in enumerate(path_iteration, start=1):
        path_iteration.set_description(
            f"Registering session {index}/{len(paths)}: {path.name}"
        )

        tracking.register_session(
            from_file=path,
            load_content=[
                "quality",
                "spatial",
                "temporal",
            ],
            align=config.align,
        )

    tracking.reset_model()
    for session_id in tqdm(
        range(len(tracking.sessions)),
        desc="Updating matching model",
        unit="session",
    ):
        tracking.update_model_with_data(
            from_session_id=session_id,
        )

    tracking.fit_to_model(
        use_cdf=config.use_cdf,
    )

    tracking.reset_registration()
    for session_id in tqdm(
        range(len(tracking.sessions)),
        desc="Registering neurons",
        unit="session",
    ):
        tracking.register_neurons(
            from_session_id=session_id,
        )

    save_matching_results(
        tracking=tracking,
        output_directory=output_directory,
    )

def save_matching_results(
    *,
    tracking: Tracking,
    output_directory: Path,
) -> None:
    """Save model and registration files to one result directory."""

    tracking.save_model(
        output_directory=output_directory,
    )

    tracking.save_registration(
        output_directory=output_directory,
    )