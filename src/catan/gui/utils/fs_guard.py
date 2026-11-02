"""File-system checks from the GUI thread that cannot freeze the GUI.

On an unresponsive network mount (e.g. sshfs whose server side hangs), a
single stat() blocks uninterruptibly until the server answers — possibly
forever. Such calls run here in a daemon thread and are abandoned after
TIMEOUT_S; the mount is then remembered as unresponsive for
UNRESPONSIVE_HOLD_S, so further checks on it fail immediately (no pile-up
of stuck threads). Daemon threads do not block quitting the application.
"""

import os
import threading
import time
from pathlib import Path

TIMEOUT_S = 5.0
UNRESPONSIVE_HOLD_S = 30.0

# mount point -> time it was found unresponsive
_unresponsive = {}
_lock = threading.Lock()


def _absolute(path):
    # Lexical only: no file-system access.
    return os.path.normpath(os.path.abspath(os.path.expanduser(os.fspath(path))))


def lexical_path(path):
    """Absolute, normalized path without touching the file system (instead of
    Path.resolve(), which follows symlinks and can block on a dead mount)."""
    return Path(_absolute(path))


# File systems whose unresponsiveness is remembered (network / FUSE);
# a slow moment of a local disk is not.
_REMOTE_TYPES = ("nfs", "nfs4", "cifs", "smb3", "smbfs", "9p", "afs", "ceph", "sshfs")


def _mounts():
    """(mount point, fs type) from /proc/mounts (Linux), longest first."""
    try:
        with open("/proc/mounts", encoding="utf-8") as handle:
            entries = {}
            for line in handle:
                parts = line.split()
                if len(parts) >= 3:
                    entries[parts[1].replace("\\040", " ")] = parts[2]
    except OSError:
        return []
    return sorted(entries.items(), key=lambda item: len(item[0]), reverse=True)


def _mount(path):
    path = _absolute(path)
    for point, kind in _mounts():
        if path == point or path.startswith(point.rstrip("/") + "/"):
            return point, kind
    return None, None


def mount_point(path):
    """Mount point containing ``path`` (lexically), or None if unknown."""
    return _mount(path)[0]


def _remote(kind):
    return bool(kind) and (kind.startswith("fuse") or kind in _REMOTE_TYPES)


def unresponsive_mount(path):
    """Mount point of ``path`` if it was unresponsive recently, else None."""
    point = mount_point(path)
    if point is None:
        return None
    with _lock:
        since = _unresponsive.get(point)
        if since is None:
            return None
        if time.monotonic() - since > UNRESPONSIVE_HOLD_S:
            _unresponsive.pop(point, None)
            return None
    return point


def probe(fn, path, *, default=None, timeout=TIMEOUT_S):
    """``fn(path)`` with a time limit. Returns ``default`` if it does not
    finish in time, raises OSError, or the path's mount was unresponsive
    recently. ``probe.timed_out`` tells whether the last call timed out."""
    if unresponsive_mount(path) is not None:
        probe.timed_out = True
        return default

    result = {}

    def run():
        try:
            result["value"] = fn(path)
        except OSError:
            result["value"] = default

    thread = threading.Thread(target=run, name="fs-probe", daemon=True)
    thread.start()
    thread.join(timeout)

    if thread.is_alive():
        point, kind = _mount(path)
        if point is not None and _remote(kind):
            with _lock:
                _unresponsive[point] = time.monotonic()
        probe.timed_out = True
        return default

    probe.timed_out = False
    return result.get("value", default)


probe.timed_out = False


def is_dir(path, default=False):
    return probe(lambda p: Path(p).is_dir(), path, default=default)


def is_file(path, default=False):
    return probe(lambda p: Path(p).is_file(), path, default=default)


def exists(path, default=False):
    return probe(lambda p: Path(p).exists(), path, default=default)


def responsive_folder(*candidates):
    """First candidate folder that exists and answers in time, else the
    home folder. Returns (folder, unresponsive candidates)."""
    skipped = []
    for candidate in candidates:
        if not candidate:
            continue
        if is_dir(candidate):
            return str(candidate), skipped
        if probe.timed_out:
            skipped.append(str(candidate))
    return str(Path.home()), skipped


def warn_unresponsive(state, folders):
    """Notice (message) that ``folders`` did not respond."""
    if not folders:
        return
    state.issue(
        "warning",
        "Folder not responding",
        "No answer within "
        f"{TIMEOUT_S:g} s from:\n"
        + "\n".join(f"  {folder}" for folder in folders)
        + "\n\nA network mount (e.g. sshfs) may be hanging. The file dialog "
        "starts elsewhere; navigating into that folder may stay empty. "
        "Remount it (e.g. fusermount3 -uz <mount>, then mount again).",
    )
