from dataclasses import dataclass
from importlib.resources import files
import json
from pathlib import Path
from importlib.metadata import distribution, PackageNotFoundError

from PySide6.QtCore import (
    QSettings,
    QStandardPaths,
)

from catan.gui.data.curation_filter import (
    CurationFilterGroup,
    curation_filter_from_dict,
    curation_filter_to_dict,
)


# Curation stages, in the recommended order of curation. Presets without
# a (known) category are listed last.
PRESET_CATEGORIES = {
    "footprints": "1. Footprint curation",
    "neurons": "2. Neuron filtering",
    "assignments": "3. Assignment correction",
    "review": "4. Review status",
}

PRESET_CATEGORY_DESCRIPTIONS = {
    "footprints": "Split, merge or remove session footprints.",
    "neurons": "Exclude neurons, e.g. noise, singular or border components.",
    "assignments": "Merge neurons, reassign or move footprints.",
    "review": "Mark neurons as reviewed or uncertain.",
}

UNCATEGORIZED_LABEL = "Other"

# Category suggested for a preset from its manipulations.
_MANIPULATION_CATEGORIES = {
    "split": "footprints",
    "merge": "footprints",
    "exclude_footprint": "footprints",
    "exclude_neuron": "neurons",
    "neuron_merge": "assignments",
    "reassign": "assignments",
    "mark_reviewed": "review",
    "mark_uncertain": "review",
}


def category_rank(category):
    order = list(PRESET_CATEGORIES)
    return order.index(category) if category in order else len(order)


def suggest_category(root) -> str | None:
    """Category of the first manipulation in the tree (root first)."""
    pending = [root]

    while pending:
        node = pending.pop(0)
        manipulation = getattr(node, "manipulation", None)
        kind = getattr(manipulation, "kind", None)

        if kind in _MANIPULATION_CATEGORIES:
            return _MANIPULATION_CATEGORIES[kind]

        pending.extend(getattr(node, "children", ()))

    return None


@dataclass(frozen=True, slots=True)
class CurationFilterPresetInfo:

    key: str
    name: str
    source: str  # "builtin" | "user"
    category: str | None = None

    @property
    def writable(self) -> bool:
        return self.source == "user"


def _known_category(document):
    category = document.get("category")
    return category if category in PRESET_CATEGORIES else None


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
                    category=_known_category(data),
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
                    category=_known_category(data),
                )
            )

        # Curation stage, then alphabetically.
        return sorted(
            presets,
            key=lambda preset: (
                category_rank(preset.category),
                preset.name.casefold(),
                preset.key,
            ),
        )

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
        category: str | None = None,
    ) -> str:

        slug = self._slug(name)

        path = self.user_directory / f"{slug}.json"

        if path.exists() and not overwrite:
            raise FileExistsError(path)

        document = curation_filter_to_dict(
            root,
            name=name,
            category=category,
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

    def new_working_preset(self):
        """Start an empty, unsaved filter."""
        self._working_filter = CurationFilterGroup(
            operator="and",
            match_level="footprint",
        )
        self._working_preset_key = None
        self._working_dirty = False

    def can_edit_category(self, key: str | None) -> bool:
        info = None if key is None else self.preset_info(key)

        return info is not None and (info.writable or self.can_save_builtin())

    def set_preset_category(self, key: str, category: str | None):
        """Assign a saved preset to a curation stage (None: uncategorized)."""
        if category is not None and category not in PRESET_CATEGORIES:
            raise ValueError(f"Unknown preset category: {category!r}")

        if not self.can_edit_category(key):
            raise PermissionError("This preset cannot be edited.")

        source, slug = self._split_key(key)
        document = self._load_document(key)
        document.pop("category", None)

        if category is not None:
            # Keep the key order of curation_filter_to_dict().
            root = document.pop("root")
            document["category"] = category
            document["root"] = root

        if source == "builtin":
            path = Path(str(self._builtin_directory())) / f"{slug}.json"
        else:
            path = self.user_directory / f"{slug}.json"

        path.write_text(
            json.dumps(document, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )

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

        if info.source == "builtin":
            if not self.can_save_builtin():
                raise PermissionError(
                    "Built-in presets can only be saved in an editable installation."
                )

            # Same file, name and category.
            source, slug = self._split_key(key)
            document = curation_filter_to_dict(
                self._working_filter,
                name=info.name,
                category=info.category,
            )
            path = Path(str(self._builtin_directory())) / f"{slug}.json"
            path.write_text(
                json.dumps(document, indent=2, ensure_ascii=False) + "\n",
                encoding="utf-8",
            )

        else:
            self.save_user_preset(
                name=info.name,
                root=self._working_filter,
                overwrite=True,
                category=info.category,
            )

        self._working_dirty = False

    def name_conflict(self, name, *, ignore_key=None):
        """Existing preset whose name equals ``name`` (ignoring case) or
        whose file name it would reuse; None if the name is free."""
        name = name.strip()
        folded = name.casefold()

        try:
            slug = self._slug(name)
        except ValueError:
            slug = None

        for preset in self.presets():
            if preset.key == ignore_key:
                continue

            stem = self._split_key(preset.key)[1]

            if preset.name.strip().casefold() == folded or stem == slug:
                return preset

        return None

    def save_working_as(
        self,
        name: str,
        *,
        overwrite: bool = False,
        category: str | None = None,
    ) -> str:

        # Saving under the current user preset's own name equals "Save".
        current = self._working_preset_key
        ignore = current if current is not None and current.startswith("user:") else None

        if self.name_conflict(name, ignore_key=ignore) is not None:
            raise FileExistsError(f"A preset named {name!r} already exists.")

        key = self.save_user_preset(
            name=name,
            root=self._working_filter,
            overwrite=overwrite,
            category=category,
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

        if self.name_conflict(new_name, ignore_key=key) is not None:
            raise FileExistsError(f"A preset named {new_name!r} already exists.")

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

        if not self.can_edit_category(key):
            raise PermissionError(
                "Built-in presets can only be deleted in an editable installation."
            )

        source, slug = self._split_key(key)

        if source == "builtin":
            path = Path(str(self._builtin_directory())) / f"{slug}.json"
        else:
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

    @staticmethod
    def can_save_builtin() -> bool:
        try:
            text = distribution("catan-toolbox").read_text("direct_url.json")
            metadata = json.loads(text or "{}")
            return metadata.get("dir_info", {}).get("editable") is True
        except (PackageNotFoundError, OSError, ValueError):
            return False

    def save_working_as_builtin(self, name: str, *, overwrite=False, category=None):
        if not self.can_save_builtin():
            raise PermissionError(
                "Saving CATAN presets requires an editable installation."
            )

        name = name.strip()
        slug = self._slug(name)
        new_key = f"builtin:{slug}"

        # The working preset itself is replaced (a user preset moves here).
        if not overwrite and self.name_conflict(
            name, ignore_key=self._working_preset_key
        ) is not None:
            raise FileExistsError(f"A preset named {name!r} already exists.")
        destination = Path(str(self._builtin_directory())) / f"{slug}.json"

        old_key = self._working_preset_key
        old_user_path = None

        if old_key is not None:
            source, old_slug = self._split_key(old_key)
            if source == "user":
                old_user_path = self.user_directory / f"{old_slug}.json"

        # Read before deleting the old preset.
        was_default = old_key is not None and self.default_preset_key == old_key

        document = curation_filter_to_dict(
            self._working_filter,
            name=name,
            category=category,
        )
        text = json.dumps(document, indent=2, ensure_ascii=False) + "\n"

        # Finish writing the native preset before removing the user version.
        with destination.open(
            "w" if overwrite else "x",
            encoding="utf-8",
        ) as stream:
            stream.write(text)

        if old_user_path is not None:
            old_user_path.unlink(missing_ok=True)

        self._working_preset_key = new_key
        self._working_dirty = False

        if was_default:
            self.settings.setValue(
                self.DEFAULT_PRESET_SETTING,
                new_key,
            )
            self.settings.sync()

        return new_key
