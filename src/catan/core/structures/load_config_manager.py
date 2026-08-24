from __future__ import annotations

from copy import deepcopy
from importlib.resources import files
from pathlib import Path

from catan.core.structures import LoadConfig


class LoadConfigManager:
    """
    Manage built-in and user-defined LoadConfig presets.

    Built-in configs are shipped inside the CATAN package under:

        catan/resources/load_configs/

    User configs are stored in a normal writable filesystem directory.

    The currently selected config is always represented by an editable
    working copy. Editing it therefore never modifies the stored preset
    until save_current() / save_current_as() is called.
    """

    BUILTIN_PACKAGE = "catan"
    BUILTIN_RESOURCE_PATH = (
        "resources",
        "load_configs",
    )

    def __init__(
        self,
        user_dir: str | Path,
        default_config: str = "CaImAn",
    ):
        self.user_dir = Path(user_dir)

        self.user_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

        # Stored presets
        self.configs: dict[str, LoadConfig] = {}

        # "builtin" or "user"
        self.config_sources: dict[str, str] = {}

        # For user configs, remember their actual file.
        self.config_paths: dict[str, Path] = {}

        # Currently selected preset
        self.current_name: str | None = None

        # Editable working copy
        self.current: LoadConfig | None = None

        self.reload_configs()

        if default_config in self.configs:
            self.select(default_config)

        elif self.configs:
            # Graceful fallback if requested default is unavailable
            self.select(next(iter(self.configs)))

        else:
            raise RuntimeError(
                "No load configurations could be found."
            )

    # ================================================================
    # Discovery / loading
    # ================================================================

    def reload_configs(self) -> None:
        """
        Reload all built-in and user-defined configs.

        User configs are loaded after built-ins and therefore override
        built-in configs with the same name.
        """

        previous_name = self.current_name

        self.configs.clear()
        self.config_sources.clear()
        self.config_paths.clear()

        self._load_builtin_configs()
        self._load_user_configs()

        # Restore the currently selected preset if possible.
        if (
            previous_name is not None
            and previous_name in self.configs
        ):
            self.select(previous_name)

    def _load_builtin_configs(self) -> None:
        """
        Load JSON presets shipped inside the installed CATAN package.
        """

        resource_dir = files(
            self.BUILTIN_PACKAGE
        )

        for part in self.BUILTIN_RESOURCE_PATH:
            resource_dir = resource_dir / part

        for resource in resource_dir.iterdir():

            if (
                not resource.is_file()
                or not resource.name.lower().endswith(".json")
            ):
                continue

            config = LoadConfig.load_resource(
                resource.name
            )
            if config.public:

                self.configs[config.name] = config
                self.config_sources[config.name] = "builtin"

    def _load_user_configs(self) -> None:
        """
        Load user-defined configs from the writable user config folder.
        """

        if not self.user_dir.exists():
            return

        for path in sorted(
            self.user_dir.glob("*.json")
        ):
            try:
                config = LoadConfig.load_json(
                    path
                )

            except Exception as exc:
                print(
                    f"Could not load config "
                    f"{path}: {exc}"
                )
                continue

            # User configs intentionally override a built-in preset
            # with the same internal name.
            self.configs[config.name] = config
            self.config_sources[config.name] = "user"
            self.config_paths[config.name] = path

    def load_fields_from_config(
        self,
        name: str,
        groups: list[str] | None = None,
    ):
        """
        Load a stored config by name and return its fields_to_load
        without changing the currently selected config.
        """

        if name not in self.configs:
            raise KeyError(
                f"Unknown load config {name!r}"
            )

        config = self.configs[name]

        return config.get_fields_to_load(
            groups=groups
        )

    # ================================================================
    # Inspection
    # ================================================================

    def names(self) -> list[str]:
        return sorted(
            self.configs.keys(),
            key=str.lower,
        )

    def builtin_names(self) -> list[str]:
        return sorted(
            (
                name
                for name, source
                in self.config_sources.items()
                if source == "builtin"
            ),
            key=str.lower,
        )

    def user_names(self) -> list[str]:
        return sorted(
            (
                name
                for name, source
                in self.config_sources.items()
                if source == "user"
            ),
            key=str.lower,
        )

    def get(
        self,
        name: str,
    ) -> LoadConfig:
        return self.configs[name]

    def source(
        self,
        name: str | None = None,
    ) -> str | None:

        if name is None:
            name = self.current_name

        if name is None:
            return None

        return self.config_sources.get(name)

    def is_builtin(
        self,
        name: str | None = None,
    ) -> bool:
        return self.source(name) == "builtin"

    def is_user(
        self,
        name: str | None = None,
    ) -> bool:
        return self.source(name) == "user"


    # ================================================================
    # Selection / working copy
    # ================================================================

    def select(
        self,
        name: str,
    ) -> LoadConfig:
        """
        Select a preset and create an editable working copy.
        """

        if name not in self.configs:
            raise KeyError(
                f"Unknown load config {name!r}"
            )

        self.current_name = name

        self.current = deepcopy(
            self.configs[name]
        )

        return self.current

    def reset_current(self) -> LoadConfig:
        """
        Discard modifications and restore the selected preset.
        """

        if self.current_name is None:
            raise RuntimeError(
                "No load config is currently selected."
            )

        return self.select(
            self.current_name
        )

    @property
    def modified(self) -> bool:
        """
        True if the editable working copy differs from the stored preset.
        """

        if (
            self.current is None
            or self.current_name is None
            or self.current_name not in self.configs
        ):
            return False

        return (
            self.current.to_dict()
            != self.configs[
                self.current_name
            ].to_dict()
        )

    # ================================================================
    # Saving
    # ================================================================

    def save_current(
        self,
    ) -> Path:
        """
        Save changes to the currently selected user preset.

        Built-in presets cannot be overwritten.
        """

        if self.current is None:
            raise RuntimeError(
                "No active load configuration."
            )

        if self.current_name is None:
            raise RuntimeError(
                "Current load configuration has no name."
            )

        if self.is_builtin():
            raise ValueError(
                "Built-in load configurations cannot "
                "be overwritten. Use save_current_as()."
            )

        path = self.config_paths.get(
            self.current_name
        )

        if path is None:
            path = (
                self.user_dir
                / f"{self._safe_filename(self.current_name)}.json"
            )

        self.current.name = self.current_name
        self.current.save_json(path)

        # Update stored preset to match saved version.
        self.configs[self.current_name] = deepcopy(
            self.current
        )

        self.config_sources[
            self.current_name
        ] = "user"

        self.config_paths[
            self.current_name
        ] = path

        return path

    def save_current_as(
        self,
        name: str,
        *,
        overwrite: bool = False,
    ) -> Path:
        """
        Save the current working configuration as a new user preset.
        """

        if self.current is None:
            raise RuntimeError(
                "No active load configuration."
            )

        name = name.strip()

        if not name:
            raise ValueError(
                "Config name must not be empty."
            )

        if (
            name in self.configs
            and not overwrite
        ):
            raise FileExistsError(
                f"A load configuration named "
                f"{name!r} already exists."
            )

        # Never overwrite the packaged built-in resource itself.
        # Saving with the same name creates/replaces a user preset.
        config = deepcopy(
            self.current
        )

        config.name = name

        path = (
            self.user_dir
            / f"{self._safe_filename(name)}.json"
        )

        if (
            path.exists()
            and not overwrite
        ):
            raise FileExistsError(
                f"Config file already exists: {path}"
            )

        config.save_json(path)

        self.configs[name] = deepcopy(
            config
        )

        self.config_sources[name] = "user"
        self.config_paths[name] = path

        self.current_name = name
        self.current = deepcopy(
            config
        )

        return path

    # ================================================================
    # Removing user presets
    # ================================================================

    def delete(
        self,
        name: str,
    ) -> None:
        """
        Delete a user-defined preset.

        Built-in presets cannot be deleted.
        """

        if name not in self.configs:
            raise KeyError(
                f"Unknown load config {name!r}"
            )

        if self.is_builtin(name):
            raise ValueError(
                "Built-in load configurations cannot "
                "be deleted."
            )

        path = self.config_paths.get(name)

        if (
            path is not None
            and path.exists()
        ):
            path.unlink()

        self.configs.pop(
            name,
            None,
        )

        self.config_sources.pop(
            name,
            None,
        )

        self.config_paths.pop(
            name,
            None,
        )

        if self.current_name == name:
            self.current_name = None
            self.current = None

    # ================================================================
    # Helpers
    # ================================================================

    @staticmethod
    def _safe_filename(
        name: str,
    ) -> str:
        """
        Convert a display name into a simple portable filename.
        """

        filename = "".join(
            char
            if (
                char.isalnum()
                or char in "-_"
            )
            else "_"
            for char in name.strip()
        )

        filename = filename.strip("_")

        if not filename:
            filename = "load_config"

        return filename