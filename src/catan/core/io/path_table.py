"""Session paths listed in a CSV table (one table, optional header row)."""

import csv
import os
from pathlib import Path, PureWindowsPath


def looks_like_path(text):
    text = text.strip()
    return bool(text) and (
        "/" in text or "\\" in text or bool(Path(text).suffix)
    )


def read_path_table(path):
    """{"header": column names or None, "rows": [[cell, ...], ...]}.

    The first row is a header if none of its cells looks like a path
    while a later row has one.
    """
    with open(path, "r", encoding="utf-8-sig", newline="") as stream:
        sample = stream.read(65536)
        stream.seek(0)

        try:
            dialect = csv.Sniffer().sniff(sample, delimiters=",;\t")
        except csv.Error:
            dialect = csv.excel

        rows = [
            [cell.strip() for cell in row]
            for row in csv.reader(stream, dialect)
            if any(cell.strip() for cell in row)
        ]

    if not rows:
        raise ValueError("The table contains no rows.")

    header = None
    if not any(looks_like_path(cell) for cell in rows[0]) and any(
        looks_like_path(cell) for row in rows[1:] for cell in row
    ):
        header, rows = rows[0], rows[1:]

    if not rows:
        raise ValueError("The table contains no path rows.")

    return {"header": header, "rows": rows}


def column_paths(table, column=0):
    return [
        row[column]
        for row in table["rows"]
        if column < len(row) and row[column].strip()
    ]


def _flavour(text):
    # Tables written on Windows use backslashes.
    return PureWindowsPath if "\\" in text and "/" not in text else Path


def _parts(text):
    return _flavour(text)(text).parts


def resolve_table_paths(paths, root):
    """Map table paths onto the data root.

    Relative paths resolve against ``root``. Absolute paths that do not
    exist (e.g. written on another machine) lose the leading folders that
    the first findable of them needs to be found under ``root``; the same
    number of folders is removed from the others.

    Returns {"paths": found, "missing": table entries not found,
    "removed": the removed leading folder or None}.
    """
    root = Path(os.path.abspath(os.path.expanduser(str(root))))
    resolved = [None] * len(paths)
    unresolved = []

    for index, text in enumerate(paths):
        parts = _parts(text)
        candidate = Path(*parts)

        if not candidate.is_absolute() and not PureWindowsPath(text).drive:
            candidate = root / candidate
            if candidate.is_file():
                resolved[index] = str(candidate)
        elif candidate.is_file():
            resolved[index] = str(candidate)
        else:
            unresolved.append((index, text, parts))

    strip = None
    removed = None

    # The first entry found under the root sets the folders to remove.
    for _, text, parts in unresolved:
        # parts[0] is the anchor ("/" or a drive): keep at least one name.
        for count in range(1, len(parts)):
            if (root / Path(*parts[count:])).is_file():
                strip = count
                removed = str(_flavour(text)(*parts[:count]))
                break

        if strip is not None:
            break

    for index, _, parts in unresolved:
        if strip is not None and strip < len(parts):
            candidate = root / Path(*parts[strip:])
            if candidate.is_file():
                resolved[index] = str(candidate)

    return {
        # Table order, without duplicates.
        "paths": list(dict.fromkeys(p for p in resolved if p is not None)),
        "missing": [text for text, p in zip(paths, resolved) if p is None],
        "removed": removed,
    }
