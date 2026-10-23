"""Returning freed memory to the operating system (glibc only; no-ops elsewhere).

NumPy work in many worker threads makes glibc create one heap ("arena") per
thread (up to 8 x cores). Freed memory stays reserved in these fragmented
heaps, so the process size (RSS) remains high after processing although the
data is gone. Limiting the arenas and trimming after heavy tasks keeps RSS
close to the memory actually in use.
"""

import ctypes
import os
import sys

# mallopt parameter (malloc.h)
M_ARENA_MAX = -8

_libc = None


def _glibc():
    global _libc

    if _libc is None:
        _libc = False

        if sys.platform.startswith("linux"):
            try:
                libc = ctypes.CDLL("libc.so.6")
                libc.gnu_get_libc_version  # glibc only
                _libc = libc
            except (OSError, AttributeError):
                pass

    return _libc or None


def limit_malloc_arenas(max_arenas=4):
    """Limit glibc heaps; call before worker threads start. Respects an
    explicit MALLOC_ARENA_MAX environment setting."""
    libc = _glibc()

    if libc is None or "MALLOC_ARENA_MAX" in os.environ:
        return False

    return bool(libc.mallopt(M_ARENA_MAX, int(max_arenas)))


def release_free_memory():
    """Return free heap memory of all arenas to the operating system."""
    libc = _glibc()

    if libc is None:
        return False

    return bool(libc.malloc_trim(0))
