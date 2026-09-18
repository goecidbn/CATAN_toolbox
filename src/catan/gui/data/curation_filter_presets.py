from dataclasses import dataclass
from importlib.resources import files
import json
from pathlib import Path

from PySide6.QtCore import (
    QSettings,
    QStandardPaths,
)

from catan.gui.data.curation_filter import (
    CurationFilterGroup,
    curation_filter_from_dict,
    curation_filter_to_dict,
)


@dataclass(frozen=True, slots=True)
class CurationFilterPresetInfo:

    key: str
    name: str
    source: str  # "builtin" | "user"

    @property
    def writable(self) -> bool:
        return self.source == "user"


class CurationFilterPresetStore:

    DEFAULT_PRESET_SETTING = "CurationFilter/default_preset"

    FALLBACK_DEFAULT_KEY = "builtin:default"

    def __init__(
        self,
        *,
        settings: QSettings,
    ):

        self.settings = settings

        self._working_filter = None
        self._working_preset_key = None
        self._working_dirty = False

        self._initialize_working_filter()

    @property
    def user_directory(self) -> Path:

        path = (
            Path(
                QStandardPaths.writableLocation(
                    QStandardPaths.StandardLocation.AppConfigLocation
                )
            )
            / "curation_filters"
        )

        path.mkdir(
            parents=True,
            exist_ok=True,
        )

        return path

    def _builtin_directory(self):

        return files("catan.resources.curation_filters")

    def presets(
        self,
    ) -> list[CurationFilterPresetInfo]:

        presets = []

        builtin_dir = self._builtin_directory()

        for resource in builtin_dir.iterdir():

            if not resource.name.endswith(".json"):
                continue

            try:
                text = resource.read_text(encoding="utf-8").strip()

                if not text:
                    raise ValueError("preset file is empty")

                data = json.loads(text)

            except (
                OSError,
                ValueError,
                json.JSONDecodeError,
            ) as exc:
                print(
                    f"Skipping invalid builtin curation "
                    f"preset {resource.name!r}: {exc}"
                )
                continue

            stem = resource.name.removesuffix(".json")

            presets.append(
                CurationFilterPresetInfo(
                    key=f"builtin:{stem}",
                    name=data.get("name", stem),
                    source="builtin",
                )
            )

        for path in self.user_directory.glob("*.json"):

            try:
                data = json.loads(path.read_text(encoding="utf-8"))
            except Exception:
                continue

            presets.append(
                CurationFilterPresetInfo(
                    key=f"user:{path.stem}",
                    name=data.get("name", path.stem),
                    source="user",
                )
            )

        return presets

    def preset_info(
        self,
        key: str,
    ) -> CurationFilterPresetInfo | None:

        return next(
            (preset for preset in self.presets() if preset.key == key),
            None,
        )

    def preset_exists(
        self,
        key: str,
    ) -> bool:

        return self.preset_info(key) is not None

    def _load_document(
        self,
        key: str,
    ) -> dict:

        source, name = self._split_key(key)

        if source == "builtin":

            resource = self._builtin_directory().joinpath(f"{name}.json")

            text = resource.read_text(encoding="utf-8")

        else:

            path = self.user_directory / f"{name}.json"

            text = path.read_text(encoding="utf-8")

        return json.loads(text)

    def load_preset(
        self,
        key: str,
    ) -> CurationFilterGroup:

        if not self.preset_exists(key):
            raise KeyError(f"Unknown curation preset: {key}")

        return curation_filter_from_dict(self._load_document(key))

    def save_user_preset(
        self,
        *,
        name: str,
        root: CurationFilterGroup,
        overwrite: bool = False,
    ) -> str:

        slug = self._slug(name)

        path = self.user_directory / f"{slug}.json"

        if path.exists() and not overwrite:
            raise FileExistsError(path)

        document = curation_filter_to_dict(
            root,
            name=name,
        )

        path.write_text(
            json.dumps(
                document,
                indent=2,
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )

        return f"user:{slug}"

    @staticmethod
    def _slug(
        name: str,
    ) -> str:

        slug = "".join(char.lower() if char.isalnum() else "_" for char in name.strip())

        slug = "_".join(part for part in slug.split("_") if part)

        if not slug:
            raise ValueError("Preset name must not be empty.")

        return slug

    @property
    def default_preset_key(
        self,
    ) -> str | None:

        requested = self.settings.value(
            self.DEFAULT_PRESET_SETTING,
            self.FALLBACK_DEFAULT_KEY,
            type=str,
        )

        if self.preset_exists(requested):
            return requested

        if self.preset_exists(self.FALLBACK_DEFAULT_KEY):
            return self.FALLBACK_DEFAULT_KEY

        presets = self.presets()

        if presets:
            return presets[0].key

        return None

    @staticmethod
    def _split_key(
        key: str,
    ) -> tuple[str, str]:

        try:
            source, name = key.split(
                ":",
                1,
            )
        except ValueError:
            raise ValueError(f"Invalid preset key: {key!r}")

        if source not in (
            "builtin",
            "user",
        ):
            raise ValueError(f"Invalid preset source: " f"{source!r}")

        return source, name

    def set_default_preset(
        self,
        key: str,
    ):

        if not self.preset_exists(key):
            raise KeyError(f"Unknown preset {key!r}")

        self.settings.setValue(
            self.DEFAULT_PRESET_SETTING,
            key,
        )

        self.settings.sync()

    @property
    def working_filter(self) -> CurationFilterGroup:
        return self._working_filter

    @property
    def working_preset_key(self) -> str | None:
        return self._working_preset_key

    @property
    def working_dirty(self) -> bool:
        return self._working_dirty

    def _initialize_working_filter(self):

        key = self.default_preset_key

        try:
            root = self.load_preset(key)

        except Exception:

            # Last-resort fallback. We should normally
            # always ship builtin:default.
            root = CurationFilterGroup(
                operator="and",
                match_level="footprint",
            )

            key = None

        self._working_filter = root
        self._working_preset_key = key
        self._working_dirty = False

    def set_working_preset(self, key: str):

        self._working_filter = self.load_preset(key)

        self._working_preset_key = key
        self._working_dirty = False

    def mark_working_dirty(
        self,
    ):

        self._working_dirty = True

    def save_working(self):

        key = self._working_preset_key

        if key is None:
            raise ValueError("Working filter is not associated " "with a preset.")

        info = self.preset_info(key)

        if info is None:
            raise KeyError(key)

        if not info.writable:
            raise PermissionError("Built-in presets cannot be overwritten.")

        self.save_user_preset(
            name=info.name,
            root=self._working_filter,
            overwrite=True,
        )

        self._working_dirty = False

    def save_working_as(
        self,
        name: str,
        *,
        overwrite: bool = False,
    ) -> str:

        key = self.save_user_preset(
            name=name,
            root=self._working_filter,
            overwrite=overwrite,
        )

        self._working_preset_key = key
        self._working_dirty = False

        return key

    def rename_user_preset(
        self,
        key: str,
        new_name: str,
    ) -> str:

        info = self.preset_info(key)

        if info is None:
            raise KeyError(key)

        if not info.writable:
            raise PermissionError("Built-in presets cannot be renamed.")

        source, old_slug = self._split_key(key)

        old_path = self.user_directory / f"{old_slug}.json"

        document = self._load_document(key)
        document["name"] = new_name

        new_slug = self._slug(new_name)
        new_path = self.user_directory / f"{new_slug}.json"

        if new_path != old_path and new_path.exists():
            raise FileExistsError(new_path)

        new_path.write_text(
            json.dumps(
                document,
                indent=2,
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )

        if new_path != old_path:
            old_path.unlink()

        new_key = f"user:{new_slug}"

        if self._working_preset_key == key:
            self._working_preset_key = new_key

        if self.default_preset_key == key:
            self.set_default_preset(new_key)

        return new_key

    def delete_user_preset(
        self,
        key: str,
    ):

        info = self.preset_info(key)

        if info is None:
            raise KeyError(key)

        if not info.writable:
            raise PermissionError("Built-in presets cannot be deleted.")

        source, slug = self._split_key(key)

        path = self.user_directory / f"{slug}.json"

        path.unlink()

        if self._working_preset_key == key:
            # Keep the current tree alive, but it is
            # no longer backed by a saved preset.
            self._working_preset_key = None
            self._working_dirty = True

        if self.default_preset_key == key:
            self.settings.remove(self.DEFAULT_PRESET_SETTING)
            self.settings.sync()
