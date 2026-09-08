from pathlib import Path
from typing import Literal
import h5py

from .types import FileFormat
from .matlab import is_mat73

SUPPORTED_SUFFIXES = (".h5", ".hdf5", ".mat", ".npz", ".zarr")

def detect_file_format(
    path: str | Path,
    *,
    for_write: bool = False,
    mat_version: Literal["pre73", "7.3"] = "7.3",
) -> FileFormat:
    path = Path(path)
    suffix = path.suffix.lower()

    if suffix in (".h5", ".hdf5"):
        return FileFormat.HDF5
    if suffix == ".mat":
        if for_write and not path.exists():
            return FileFormat.MAT73 if mat_version == "7.3" else FileFormat.MAT_PRE73
        if for_write and path.exists():
            return FileFormat.MAT73 if is_mat73(path) else FileFormat.MAT_PRE73
        return FileFormat.MAT73 if is_mat73(path) else FileFormat.MAT_PRE73
    if suffix == ".npz":
        return FileFormat.NPZ
    if suffix == ".zarr" or (path.is_dir() and path.name.lower().endswith(".zarr")):
        return FileFormat.ZARR

    return None
    # raise ValueError(
    #     f"Unsupported file type {suffix!r}. Supported: {', '.join(SUPPORTED_SUFFIXES)}"
    # )
