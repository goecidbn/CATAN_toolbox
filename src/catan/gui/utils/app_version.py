"""CATAN version for display.

An editable install writes ``catan/_version.py`` only when it is installed,
so in a git checkout that file falls behind every later tag and commit. The
version is then taken from ``git describe`` (as setuptools_scm would number
it: the next patch version as dev release after a tag); otherwise, or if git
fails, from the installed package.
"""

import re
import subprocess
from pathlib import Path

GIT_TIMEOUT_S = 2.0

_DESCRIBE = re.compile(r"^v?(?P<tag>\d+(?:\.\d+)*)-(?P<distance>\d+)-g(?P<node>[0-9a-f]+)(?P<dirty>-dirty)?$")


def _installed_version():
    import catan

    return str(getattr(catan, "__version__", "") or "unknown")


def version_from_describe(described):
    """``git describe --tags --long --dirty`` output → version string, or None.

    "v0.1.15-0-g0e5a311" → "0.1.15"; "v0.1.15-3-g0e5a311-dirty" →
    "0.1.16.dev3+g0e5a311.dirty".
    """
    match = _DESCRIBE.match(str(described).strip())
    if match is None:
        return None
    tag, distance = match["tag"], int(match["distance"])
    dirty = bool(match["dirty"])
    if distance == 0 and not dirty:
        return tag
    if distance == 0:
        return f"{tag}+g{match['node']}.dirty"
    parts = tag.split(".")
    parts[-1] = str(int(parts[-1]) + 1)
    local = f"g{match['node']}" + (".dirty" if dirty else "")
    return f"{'.'.join(parts)}.dev{distance}+{local}"


def _checkout_root():
    """Repository root if CATAN runs from its own git checkout (src layout)."""
    import catan

    root = Path(catan.__file__).resolve().parent.parent.parent
    if (root / ".git").exists() and (root / "pyproject.toml").is_file():
        return root
    return None


def catan_version():
    root = _checkout_root()
    if root is not None:
        try:
            described = subprocess.run(
                ["git", "describe", "--tags", "--long", "--dirty", "--match", "v[0-9]*"],
                cwd=root,
                capture_output=True,
                text=True,
                timeout=GIT_TIMEOUT_S,
                check=True,
            ).stdout
        except (OSError, subprocess.SubprocessError):
            described = ""
        version = version_from_describe(described)
        if version is not None:
            return version
    return _installed_version()
