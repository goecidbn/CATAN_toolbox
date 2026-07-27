"""Selection of a working Qt graphics backend for CATAN."""

from __future__ import annotations

from dataclasses import dataclass
import logging
import os
import platform
import subprocess
import sys
from collections.abc import Mapping

logger = logging.getLogger(__name__)

_PROBE_MODULE = "catan.gui.utils._graphics_probe"
_PROBE_TIMEOUT_SECONDS = 20


@dataclass(frozen=True)
class ProbeResult:
    """Result of an isolated graphics probe."""

    success: bool
    returncode: int | None
    stdout: str
    stderr: str
    error: str | None = None


def configure_graphics_backend() -> None:
    """Select a usable Qt graphics backend before Qt is imported.

    On Linux Wayland sessions, CATAN first tests the native Qt backend.
    If native VisPy rendering fails but XCB rendering works, CATAN uses
    the XCB platform plugin for the current process.

    Windows, macOS, X11 sessions, and explicit user selections are left
    unchanged.
    """

    if platform.system() != "Linux":
        return

    # Permit users and automated tests to bypass the probe.
    if os.environ.get("CATAN_SKIP_GRAPHICS_PROBE") == "1":
        logger.debug("Skipping CATAN graphics probe.")
        return

    # Respect an explicit Qt configuration made by the user, system
    # administrator, test environment, or packaging system.
    if "QT_QPA_PLATFORM" in os.environ:
        logger.debug(
            "Respecting explicit QT_QPA_PLATFORM=%s",
            os.environ["QT_QPA_PLATFORM"],
        )
        return

    if not _is_wayland_session():
        return

    native_result = _run_probe()

    if native_result.success:
        logger.debug("Native Wayland VisPy probe succeeded.")
        return

    logger.info("Native Wayland VisPy rendering failed; testing the Qt XCB backend.")

    xcb_result = _run_probe(
        {
            "QT_QPA_PLATFORM": "xcb",
        }
    )

    if xcb_result.success:
        # This must be set before QApplication or any Qt window is created.
        os.environ["QT_QPA_PLATFORM"] = "xcb"

        logger.warning(
            "Native Wayland rendering is incompatible with the current "
            "VisPy/OpenGL setup. CATAN is using the Qt XCB backend."
        )
        return

    raise RuntimeError(
        _format_failure_message(
            native_result=native_result,
            xcb_result=xcb_result,
        )
    )


def _is_wayland_session() -> bool:
    session_type = os.environ.get("XDG_SESSION_TYPE", "").lower()

    return session_type == "wayland" or bool(os.environ.get("WAYLAND_DISPLAY"))


def _run_probe(
    environment_updates: Mapping[str, str] | None = None,
) -> ProbeResult:
    env = os.environ.copy()

    if environment_updates:
        env.update(environment_updates)

    try:
        completed = subprocess.run(
            [
                sys.executable,
                "-m",
                _PROBE_MODULE,
            ],
            env=env,
            capture_output=True,
            text=True,
            timeout=_PROBE_TIMEOUT_SECONDS,
            check=False,
        )

    except subprocess.TimeoutExpired as exc:
        return ProbeResult(
            success=False,
            returncode=None,
            stdout=exc.stdout or "",
            stderr=exc.stderr or "",
            error=(
                "Graphics probe timed out after " f"{_PROBE_TIMEOUT_SECONDS} seconds."
            ),
        )

    except OSError as exc:
        return ProbeResult(
            success=False,
            returncode=None,
            stdout="",
            stderr="",
            error=f"Could not start graphics probe: {exc}",
        )

    return ProbeResult(
        success=completed.returncode == 0,
        returncode=completed.returncode,
        stdout=completed.stdout,
        stderr=completed.stderr,
    )


def _format_failure_message(
    *,
    native_result: ProbeResult,
    xcb_result: ProbeResult,
) -> str:
    return (
        "CATAN could not initialize a working VisPy graphics backend.\n\n"
        "The native Wayland probe failed, and the Qt XCB fallback also "
        "failed.\n\n"
        "Native Wayland probe:\n"
        f"{_summarize_probe(native_result)}\n\n"
        "XCB probe:\n"
        f"{_summarize_probe(xcb_result)}\n\n"
        "You can override automatic backend selection by setting "
        "QT_QPA_PLATFORM explicitly. To bypass the probe entirely, set "
        "CATAN_SKIP_GRAPHICS_PROBE=1."
    )


def _summarize_probe(result: ProbeResult) -> str:
    details: list[str] = []

    if result.error:
        details.append(result.error)

    if result.returncode is not None:
        details.append(f"Exit code: {result.returncode}")

    output = result.stderr.strip() or result.stdout.strip()

    if output:
        # Keep error reports useful without dumping thousands of lines.
        details.append(output[-4000:])

    return "\n".join(details) or "No diagnostic output was produced."
