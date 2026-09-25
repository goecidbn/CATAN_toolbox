from __future__ import annotations

from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator

import cv2
import numpy as np
import tifffile

from catan.core.structures.load_config import FieldSpec

from .base import IOBackend
from .types import (
    FieldInfo,
    FileFormat,
    FileStructure,
    SaveEntry,
    normalize_path,
    normalized_source,
)


class ImageBackend(IOBackend):

    file_format = FileFormat.IMAGE

    @contextmanager
    def open_read(self, path: str | Path) -> Iterator[np.ndarray]:

        path = Path(path)

        if path.suffix.lower() in {".tif", ".tiff"}:
            image = np.asarray(tifffile.imread(path))

        else:
            image = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)

            if image is None:
                raise ValueError(f"Could not read image file {path}")

            # OpenCV returns ordinary color images as BGR/BGRA.
            if image.ndim == 3:
                if image.shape[-1] == 3:
                    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

                elif image.shape[-1] == 4:
                    image = cv2.cvtColor(image, cv2.COLOR_BGRA2RGBA)

        yield image

    @contextmanager
    def open_write(self, path):
        raise NotImplementedError("CATAN image sources are currently read-only.")
        yield None

    def read_field(
        self, ref: np.ndarray, spec: FieldSpec, *, key: str | None, root: str
    ) -> dict[str, Any]:

        if normalized_source(spec.source) != "field":
            if spec.required:
                raise ValueError("Raster images do not provide attributes.")
            return {}

        path = normalize_path(spec.path)

        if path not in {"/", "/image"}:
            if spec.required:
                raise KeyError(f"Image source has no field {path!r}")
            return {}

        return {key or "image": ref}

    def load_all(self, ref: np.ndarray, *, root: str = "/") -> dict[str, Any]:
        return {"image": ref}

    def inspect(self, ref: np.ndarray, *, root: str = "/") -> FileStructure:

        structure = FileStructure(self.file_format, root="/")

        structure.add(
            FieldInfo(
                name="image",
                path="/image",
                kind="field",
                shape=tuple(ref.shape),
                dtype=str(ref.dtype),
            )
        )

        return structure

    def write_entry(self, ref, entry: SaveEntry, *, root: str) -> None:
        raise NotImplementedError("CATAN image sources are currently read-only.")

    def get_attribute(self, ref, path: str, name: str, *, root="/", default=None):
        return default

    def set_attribute(self, ref, path: str, name: str, value, *, root="/") -> None:
        raise NotImplementedError("CATAN image sources do not support attributes.")

    def list_groups(self, ref, *, root="/") -> list[str]:
        return []

    def exists(self, ref, path: str) -> bool:
        return normalize_path(path) in {"/", "/image"}
