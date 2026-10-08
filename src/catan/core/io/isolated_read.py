from __future__ import annotations

import os
import pickle
import shutil
import subprocess
import sys
import tempfile
import threading
import time
import traceback
from uuid import uuid4
from pathlib import Path

# Includes processes whose termination is still pending.
# Repeated retries must not create unlimited stuck readers.
_READER_SLOTS = threading.BoundedSemaphore(2)


class ReadFailure(RuntimeError):
    def __init__(self, summary: str, recovery: str):
        super().__init__(summary)
        self.failure_info = {
            "summary": summary,
            "recovery": recovery,
        }


def _check_cancelled(ctx):
    if ctx is not None:
        ctx.check_cancelled()


def _message(ctx, text):
    if ctx is not None:
        ctx.message(text)


def _cleanup(directory):
    shutil.rmtree(directory, ignore_errors=True)


def _reap_later(process, directory):
    """Retain the capacity slot until the child actually exits."""
    try:
        while process.poll() is None:
            time.sleep(0.25)
    finally:
        _cleanup(directory)
        _READER_SLOTS.release()


def _recovery_for(exception):
    if isinstance(exception, FileNotFoundError):
        return (
            "The source file or one of its directories was not found. "
            "Restore access to the original location, or correct the "
            "configured field source, then use Load again."
        )

    if isinstance(exception, PermissionError):
        return (
            "CATAN could not access the source with the current permissions. "
            "Restore access outside CATAN, then use Load again."
        )

    if isinstance(exception, KeyError):
        return (
            "A required field or object could not be read. Check the "
            "selected load configuration and field paths, then use Load "
            "again. Reconnecting alone will not fix an incorrect mapping."
        )

    if isinstance(exception, MemoryError):
        return (
            "There was not enough memory to read the selected data. "
            "Unload unused data or select fewer fields, then use Load again."
        )

    if isinstance(exception, OSError):
        return (
            "The operating system or file backend could not read the source. "
            "If it is on a mounted drive, restore access outside CATAN. "
            "Otherwise check that the file is readable and valid. "
            "Then use Load again."
        )

    return (
        "Check the source and load configuration. Open Details for the "
        "underlying error before retrying with Load."
    )


def _acquire_reader_slot(ctx, *, timeout=60.0):
    _check_cancelled(ctx)

    if _READER_SLOTS.acquire(blocking=False):
        return

    _message(
        ctx,
        "Waiting for a file reader. Another operation may still be "
        "finishing or stopping. You can cancel this wait.",
    )

    deadline = time.monotonic() + timeout

    while True:
        _check_cancelled(ctx)

        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise ReadFailure(
                "No file reader became available within " f"{timeout:g} seconds.",
                (
                    "Other file operations may still be running, or a "
                    "cancelled reader may be waiting for the operating "
                    "system to release an unavailable mounted drive. "
                    "Restore access to that drive outside CATAN if needed, "
                    "then click Retry. Alternatively, cancel this batch "
                    "and select its unfinished sessions again later. "
                    "Previously completed sessions are retained."
                ),
            )

        if _READER_SLOTS.acquire(timeout=min(0.1, remaining)):
            return


def read_fields(
    path,
    fields_to_load,
    *,
    root="/",
    ctx=None,
    warn_after=15.0,
    timeout=300.0,
    operation="fields",
    parameters=None,
    activity="Reading",
):
    """
    Run selected-field I/O outside the CATAN process.

    Call from a background worker, never directly from the GUI thread.
    The timeout covers the child process, including result serialization.
    """
    path = os.path.abspath(os.path.expanduser(os.fspath(path)))

    _acquire_reader_slot(ctx)

    directory = None
    process = None
    deferred_cleanup = False

    try:
        _check_cancelled(ctx)

        directory = Path(
            tempfile.mkdtemp(
                prefix="catan-read-",
                dir=os.environ.get("CATAN_IO_TEMP_DIR") or None,
            )
        )

        request_path = directory / "request.pkl"
        result_path = directory / "result.pkl"
        log_path = directory / "reader.log"

        with request_path.open("wb") as stream:
            pickle.dump(
                {
                    "path": path,
                    "fields": fields_to_load,
                    "root": root,
                    "operation": operation,
                    "parameters": parameters or {},
                },
                stream,
                protocol=pickle.HIGHEST_PROTOCOL,
            )

        _check_cancelled(ctx)

        if getattr(sys, "frozen", False):
            command = [
                sys.executable,
                "--catan-read-worker",
                str(directory),
            ]
        else:
            package_root = Path(__file__).resolve().parents[3]

            bootstrap = (
                "import sys; "
                "sys.path.insert(0, sys.argv[1]); "
                "from catan.core.io.isolated_read import _child_main; "
                "_child_main(sys.argv[2])"
            )

            command = [
                sys.executable,
                "-c",
                bootstrap,
                str(package_root),
                str(directory),
            ]

        options = {}
        if os.name == "nt":
            options["creationflags"] = subprocess.CREATE_NO_WINDOW

        # Use a local working directory, independent of the remote source.
        with log_path.open("wb") as log:
            process = subprocess.Popen(
                command,
                cwd=str(directory),
                stdin=subprocess.DEVNULL,
                stdout=log,
                stderr=log,
                **options,
            )

        started = time.monotonic()
        warned = False

        _message(ctx, f"{activity} {path}")

        while process.poll() is None:
            _check_cancelled(ctx)

            elapsed = time.monotonic() - started

            if not warned and elapsed >= warn_after:
                warned = True
                _message(
                    ctx,
                    (
                        f"{activity} {path} is taking longer than expected. "
                        "The location may be slow or unavailable. "
                        "You can cancel this operation."
                    ),
                )

            if elapsed >= timeout:
                raise ReadFailure(
                    f"{activity} {path} exceeded {timeout:g} seconds.",
                    (
                        "No result was returned by this read operation. "
                        "The source may be unavailable or the read may simply "
                        "need longer. For a mounted drive, restore access "
                        "outside CATAN before using Load again."
                    ),
                )

            try:
                process.wait(timeout=0.05)
            except subprocess.TimeoutExpired:
                pass

        _check_cancelled(ctx)

        if process.returncode != 0 or not result_path.is_file():
            details = log_path.read_text(errors="replace")
            raise ReadFailure(
                f"The file reader stopped unexpectedly while reading {path}.",
                (
                    "No result was returned by this read operation. "
                    "Check source access and available memory, then use "
                    "Load again.\n\n"
                    f"Reader exit code: {process.returncode}\n"
                    f"{details[-4000:]}"
                ),
            )

        with result_path.open("rb") as stream:
            result = pickle.load(stream)

        _check_cancelled(ctx)

        if not result["ok"]:
            error = ReadFailure(
                result["summary"],
                (
                    "No result was returned by this read operation.\n\n"
                    + result["recovery"]
                ),
            )
            # Include the child traceback in the ordinary worker traceback.
            raise error from RuntimeError(result["traceback"])

        return result["data"]

    finally:
        if process is not None and process.poll() is None:
            try:
                process.kill()
            except OSError:
                pass

            # Do not wait here: kernel I/O can delay actual termination.
            if process.poll() is None:
                deferred_cleanup = True
                threading.Thread(
                    target=_reap_later,
                    args=(process, directory),
                    daemon=True,
                    name="catan-reader-cleanup",
                ).start()

        if not deferred_cleanup:
            if directory is not None:
                _cleanup(directory)
            _READER_SLOTS.release()


def read_operation(
    operation,
    path,
    *,
    fields_to_load=None,
    root="/",
    ctx=None,
    timeout=60.0,
    retry_hint=None,
    **parameters,
):
    try:
        return read_fields(
            path,
            fields_to_load,
            root=root,
            ctx=ctx,
            timeout=timeout,
            operation=operation,
            parameters=parameters,
        )
    except ReadFailure as exc:
        if retry_hint is None:
            raise

        raise ReadFailure(
            str(exc),
            f"{exc.failure_info['recovery']}\n\n{retry_hint}",
        ) from exc


def _execute_request(request):
    operation = request.get("operation", "fields")
    path = request["path"]
    root = request.get("root", "/")

    if operation == "fields":
        from catan.core.io.api import load_fields_from_sources

        return load_fields_from_sources(
            path,
            request["fields"],
            root=root,
        )

    if operation == "inspect":
        from catan.core.io.api import get_backend

        return get_backend(path).inspect_file(path, root=root)

    if operation in {
        "dimensions",
        "session_compatibility",
        "session_fields",
    }:
        from catan.core.spatial_geometry import inspect_dimensions
        from catan.core.io.api import load_fields_from_sources
        from catan.core.io.inspection import check_fields_compatibility

        parameters = request["parameters"]

        geometry = inspect_dimensions(
            path,
            request["fields"],
            parameters["dimensions"],
            parameters.get("orientation", "auto"),
            parameters.get("expected_dims"),
        )

        if operation == "dimensions":
            return geometry

        if operation == "session_compatibility":
            return {
                "fields": check_fields_compatibility(
                    path,
                    request["fields"],
                    refresh=True,
                    raise_source_errors=True,
                ),
                "geometry": geometry,
            }

        if not geometry["ok"]:
            raise ValueError(geometry["message"])

        data = load_fields_from_sources(
            path,
            request["fields"],
            root=root,
        )
        data["spatial"]["dims"] = geometry["dims"]
        return data

    if operation == "compatibility":
        from catan.core.io.inspection import check_fields_compatibility

        return check_fields_compatibility(
            path,
            request["fields"],
            root=root,
            refresh=True,
            raise_source_errors=True,
        )

    if operation == "glob":
        pattern = request["parameters"]["pattern"]
        directory = Path(path)

        # Check the root explicitly: glob can otherwise report an
        # inaccessible/nonexistent directory as simply having no matches.
        with os.scandir(directory) as entries:
            next(entries, None)

        return [str(match) for match in directory.glob(pattern) if match.is_file()]

    if operation == "json":
        import json

        with open(path, "r", encoding="utf-8") as stream:
            return json.load(stream)

    if operation == "registration":
        from catan.core.io.api import get_backend

        backend = get_backend(path)
        with backend.open_read(path) as ref:
            object_type = backend.get_attribute(ref, "/", "object_type")

        if isinstance(object_type, bytes):
            object_type = object_type.decode("utf-8")

        native = object_type in {"SessionData", "SessionList"}

        if native:
            # Run the existing native-session loader entirely inside
            # this child. Do not open the native file again in the GUI process.
            from catan import Tracking

            sessions = Tracking().load_session_data(path)
        else:
            sessions = None

        return {
            "restored_from_catan": native,
            "sessions": sessions,
        }

    if operation == "format":
        from catan.core.io.detection import detect_file_format

        file_format = detect_file_format(path)
        if file_format is None:
            raise ValueError(f"Unsupported file format: {path}")

        return file_format

    if operation in {"assignments_preflight", "assignments_fields"}:
        from catan.core.io.api import read_assignments_source

        return read_assignments_source(
            path,
            request["fields"],
            load_data=(operation == "assignments_fields"),
        )

    if operation in {"browse_directory", "select_path"}:
        import stat

        path = os.path.abspath(os.path.expanduser(os.fspath(path)))
        parameters = request.get("parameters") or {}

        if operation == "select_path":
            pick_dir = parameters.get("pick_dir", False)
            only_existing = parameters.get("only_existing", True)

            suffix = parameters.get("default_suffix", "").lstrip(".")

            if not pick_dir and not only_existing and suffix and not Path(path).suffix:
                # Existing directories must remain navigable.
                try:
                    is_directory = stat.S_ISDIR(os.stat(path).st_mode)
                except FileNotFoundError:
                    is_directory = False

                if not is_directory:
                    path += "." + suffix

            try:
                mode = os.stat(path).st_mode
            except FileNotFoundError:
                if pick_dir or only_existing:
                    raise

                # For saving, validate the parent without creating anything.
                parent = os.path.dirname(path)
                if not stat.S_ISDIR(os.stat(parent).st_mode):
                    raise NotADirectoryError(parent)

                return {
                    "path": path,
                    "directory": False,
                    "exists": False,
                }

            is_directory = stat.S_ISDIR(mode)

            if pick_dir and not is_directory:
                raise NotADirectoryError(path)

            if not is_directory and not stat.S_ISREG(mode):
                raise ValueError(f"Not a regular file: {path}")

            return {
                "path": path,
                "directory": is_directory,
                "exists": True,
            }

        # A supplied file path opens its containing directory and preselects it.
        selected = None

        try:
            mode = os.stat(path).st_mode
        except FileNotFoundError:
            if not parameters.get("allow_missing_file", False):
                raise

            selected = os.path.basename(path)
            path = os.path.dirname(path)

            if not stat.S_ISDIR(os.stat(path).st_mode):
                raise NotADirectoryError(path)
        else:
            if not stat.S_ISDIR(mode):
                selected = os.path.basename(path)
                path = os.path.dirname(path)

        entries = []

        with os.scandir(path) as iterator:
            for entry in iterator:
                # Do not follow links while listing. Resolve only when opened.
                if entry.is_symlink():
                    kind = "link"
                elif entry.is_dir(follow_symlinks=False):
                    kind = "directory"
                elif entry.is_file(follow_symlinks=False):
                    kind = "file"
                else:
                    continue

                entries.append((entry.name, kind))

        entries.sort(
            key=lambda entry: (
                entry[1] != "directory",
                entry[0].casefold(),
            )
        )

        return {
            "path": path,
            "selected": selected,
            "entries": entries,
        }

    if operation == "write_bytes":
        parameters = request["parameters"]
        return _write_bytes_atomic(
            path,
            parameters["temporary"],
            parameters["payload"],
        )

    if operation == "write_session_snapshots":
        from catan.core.io.api import get_backend
        from catan.core.structures.session_snapshot import (
            write_prepared_session_snapshot,
        )

        parameters = request["parameters"]
        temporary = Path(parameters["temporary"])
        snapshots = parameters["snapshots"]

        if not snapshots:
            raise ValueError("No session snapshots to save.")

        # Detection may inspect an existing MATLAB destination.
        # It runs here, inside the guarded process.
        backend = get_backend(
            path,
            for_write=True,
            mat_version=parameters["mat_version"],
        )

        # Reserve the temporary filename without replacing an existing file.
        with temporary.open("xb"):
            pass

        with backend.open_write(temporary) as ref:
            backend.set_attribute(ref, "/", "object_type", parameters["object_type"])
            backend.set_attribute(ref, "/", "format_version", 2)

            for index, snapshot in enumerate(snapshots):
                write_prepared_session_snapshot(
                    backend,
                    ref,
                    snapshot,
                    root=f"/session_{index:03d}",
                )

        # All backend handles are closed before replacement.
        with temporary.open("rb+") as stream:
            os.fsync(stream.fileno())

        os.replace(temporary, path)
        return str(path)

    if operation == "write_prepared_file":
        from catan.core.io.api import get_backend

        parameters = request["parameters"]
        prepared = parameters["prepared"]
        temporary = Path(parameters["temporary"])

        backend = get_backend(
            path,
            for_write=True,
            mat_version=parameters["mat_version"],
        )

        with temporary.open("xb"):
            pass

        with backend.open_write(temporary) as ref:
            for name, value in prepared["attributes"].items():
                backend.set_attribute(ref, "/", name, value)

            backend.write(
                ref,
                prepared["data"],
                prepared["fields"],
                root="/",
            )

        with temporary.open("rb+") as stream:
            os.fsync(stream.fileno())

        os.replace(temporary, path)
        return str(path)

    if operation == "model":
        from catan.core.io import load_file
        from catan.tracking.structures.model import NATIVE_MODEL_CONFIG

        return load_file(
            path,
            config_name=NATIVE_MODEL_CONFIG,
            root="/",
        )

    if operation == "create_directory":
        name = request["parameters"]["name"]

        # Accept a single directory name, not a relative/absolute path.
        if (
            not name
            or name in {".", ".."}
            or any(character in name for character in ("/", "\\", "\0"))
            or os.path.splitdrive(name)[0]
        ):
            raise ValueError("Enter a single folder name.")

        directory = Path(path) / name

        # Makes Retry safe if creation succeeded but its response was lost.
        # An existing non-directory still raises an error.
        directory.mkdir(exist_ok=True)

        return {"path": str(directory)}

    raise ValueError(f"Unknown read operation: {operation!r}")


def _child_main(directory):
    directory = Path(directory)

    try:
        with (directory / "request.pkl").open("rb") as stream:
            request = pickle.load(stream)

        data = _execute_request(request)

        result = {
            "ok": True,
            "data": data,
        }

    except Exception as exc:
        result = {
            "ok": False,
            "summary": f"Could not read the source: {exc}",
            "recovery": _recovery_for(exc),
            "traceback": traceback.format_exc(),
        }

    with (directory / "result.pkl").open("wb") as stream:
        pickle.dump(result, stream, protocol=pickle.HIGHEST_PROTOCOL)


if __name__ == "__main__":
    # File location:
    # <package_root>/catan/core/io/isolated_read.py
    #
    # Make this same CATAN package available to the child, including
    # when the parent was launched directly from a source checkout.
    package_root = Path(__file__).resolve().parents[3]
    sys.path.insert(0, str(package_root))

    _child_main(sys.argv[1])


def _write_operation(
    operation,
    path,
    parameters,
    *,
    ctx=None,
    timeout=300.0,
):
    target = Path(os.path.abspath(os.path.expanduser(os.fspath(path))))

    # Keep the real extension last: some backends append it otherwise.
    temporary = target.with_name(
        f".{target.stem}.catan-{uuid4().hex}.tmp{target.suffix}"
    )

    try:
        return read_fields(
            target,
            None,
            ctx=ctx,
            timeout=timeout,
            activity="Saving",
            operation=operation,
            parameters={
                **parameters,
                "temporary": str(temporary),
            },
        )
    except Exception as exc:
        info = {
            "summary": f"Save was not confirmed: {target}",
            "recovery": (
                "Restore access to the destination before retrying. "
                "If interruption occurred during final replacement, "
                "the destination may already contain the completed save. "
                "Check it before saving again, or choose another filename.\n\n"
                "An interrupted save may leave this temporary file:\n"
                f"{temporary}\n"
                "Remove it only after the save process has stopped and "
                "you have checked the destination."
            ),
        }

        if isinstance(exc, ReadFailure):
            raise ReadFailure(info["summary"], info["recovery"]) from exc

        exc.failure_info = info
        raise


def write_bytes(path, payload, *, ctx=None, timeout=300.0):
    return _write_operation(
        "write_bytes",
        path,
        {"payload": payload},
        ctx=ctx,
        timeout=timeout,
    )


def write_session_snapshots(
    path,
    snapshots,
    *,
    mat_version="7.3",
    object_type="SessionList",
    ctx=None,
    timeout=300.0,
):
    if Path(path).suffix.lower() not in {
        ".h5",
        ".hdf5",
        ".mat",
        ".npz",
    }:
        raise ValueError(
            "Guarded session saving supports .h5, .hdf5, .mat, "
            "and .npz files. Choose one of these formats."
        )

    return _write_operation(
        "write_session_snapshots",
        path,
        {
            "snapshots": snapshots,
            "mat_version": mat_version,
            "object_type": object_type,
        },
        ctx=ctx,
        timeout=timeout,
    )


def _write_bytes_atomic(target, temporary, payload):
    target = Path(target)
    temporary = Path(temporary)

    # Exclusive creation avoids modifying any pre-existing temp file.
    # Leave temporary files on failure: cleanup against an unavailable
    # filesystem could itself stall or obscure the original exception.
    with temporary.open("xb") as stream:
        stream.write(payload)
        stream.flush()
        os.fsync(stream.fileno())

    # Both paths are in the same directory.
    os.replace(temporary, target)
    return str(target)


def write_prepared_file(
    path,
    prepared,
    *,
    mat_version="7.3",
    ctx=None,
    timeout=300.0,
):
    if Path(path).suffix.lower() not in {
        ".h5",
        ".hdf5",
        ".mat",
        ".npz",
    }:
        raise ValueError("Choose a .h5, .hdf5, .mat, or .npz destination.")

    return _write_operation(
        "write_prepared_file",
        path,
        {
            "prepared": prepared,
            "mat_version": mat_version,
        },
        ctx=ctx,
        timeout=timeout,
    )
