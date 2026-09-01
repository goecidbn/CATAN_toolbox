from __future__ import annotations

from importlib.resources import files
from pathlib import Path
import json

from .config import LoadConfig
from catan.core.io.types import FileFormat
from catan.core.io.detection import detect_file_format


class LoadConfigManager:
    """Manage packaged presets, editable working copies and user configs.

    Packaged configs are immutable on disk, but non-native packaged configs can
    be edited and saved under the same name: this creates a user override in the
    writable config directory. Deleting a non-native config persists as a
    lightweight "hidden builtin" tombstone rather than modifying the installed
    package. CATAN-native schemas remain protected.
    """

    BUILTIN_PACKAGE = "catan"
    BUILTIN_RESOURCE_PATH = ("resources", "load_configs")
    HIDDEN_BUILTINS_FILE = ".hidden_builtins.json"

    default_fallback: str = "catan-caiman-session"
    default_by_format: dict[FileFormat, str]

    def __init__(self, user_dir: str | Path):
        self.user_dir = Path(user_dir)
        self.user_dir.mkdir(parents=True, exist_ok=True)

        self.configs: dict[str, LoadConfig] = {}
        self.config_sources: dict[str, str] = {}  # builtin | user
        self.config_paths: dict[str, Path] = {}
        self._packaged_configs: dict[str, LoadConfig] = {}
        self._hidden_builtins: set[str] = self._load_hidden_builtins()

        self._inspector = None

        self.defaults_file = (
            self.user_dir.parent
            / "load_config_defaults.json"
        )

        self.reload_configs()
        self._load_defaults()
                
        self.last_used_by_format: dict[FileFormat, LoadConfig] = {}

    # ------------------------------------------------------------------
    # Discovery
    # ------------------------------------------------------------------
    @property
    def _hidden_path(self) -> Path:
        return self.user_dir / self.HIDDEN_BUILTINS_FILE

    def _load_hidden_builtins(self) -> set[str]:
        path = self.user_dir / self.HIDDEN_BUILTINS_FILE
        if not path.exists():
            return set()
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            return set(data.get("hidden", [])) if isinstance(data, dict) else set()
        except Exception:
            return set()

    def _save_hidden_builtins(self) -> None:
        self._hidden_path.write_text(
            json.dumps({"hidden": sorted(self._hidden_builtins)}, indent=2) + "\n",
            encoding="utf-8",
        )

    def reload_configs(self) -> None:
        self.configs.clear()
        self.config_sources.clear()
        self.config_paths.clear()
        self._packaged_configs.clear()
        self._hidden_builtins = self._load_hidden_builtins()
        self._load_builtin_configs()
        self._load_user_configs()

    def _builtin_dir(self):
        resource = files(self.BUILTIN_PACKAGE)
        for part in self.BUILTIN_RESOURCE_PATH:
            resource = resource / part
        return resource

    def _load_builtin_configs(self) -> None:
        for resource in self._builtin_dir().iterdir():
            if not resource.is_file() or not resource.name.lower().endswith(".json"):
                continue
            config = LoadConfig.load_resource(resource.name)
            self._packaged_configs[config.name] = config
            if config.name in self._hidden_builtins and not config.native:
                continue
            self.configs[config.name] = config
            self.config_sources[config.name] = "builtin"

    def _load_user_configs(self) -> None:
        for path in sorted(self.user_dir.glob("*.json")):
            if path.name == self.HIDDEN_BUILTINS_FILE:
                continue
            try:
                config = LoadConfig.load_json(path)
            except Exception as exc:
                print(f"Could not load config {path}: {exc}")
                continue

            packaged = self._packaged_configs.get(config.name)
            if packaged is not None and packaged.native:
                print(
                    f"Ignoring user config {path}: name {config.name!r} is reserved "
                    "by a CATAN-native config."
                )
                continue

            # A user override explicitly brings a hidden non-native builtin name
            # back into the active set, with the user config taking precedence.
            self.configs[config.name] = config
            self.config_sources[config.name] = "user"
            self.config_paths[config.name] = path

    # ------------------------------------------------------------------
    # Lookup / selection
    # ------------------------------------------------------------------
    def names(
        self,
        *,
        public_only: bool = True,
        source_type: str | None = None,
    ) -> list[str]:
        names = []
        for name, config in self.configs.items():
            if public_only and not config.public:
                continue
            if source_type is not None and config.source_type != source_type:
                continue
            names.append(name)
        return sorted(names, key=str.lower)

    def builtin_names(self, *, public_only: bool = True) -> list[str]:
        return [
            name for name in self.names(public_only=public_only)
            if self.config_sources.get(name) == "builtin"
        ]

    def user_names(self, *, public_only: bool = True) -> list[str]:
        return [
            name for name in self.names(public_only=public_only)
            if self.config_sources.get(name) == "user"
        ]
    

    def get(self, name: str, *, copy: bool = False) -> LoadConfig:
        config = self.configs[name]
        return config.copy() if copy else config

    def get_by_uid(
        self,
        uid: str,
    ) -> LoadConfig:

        for config in self.configs.values():
            if config.uid == uid:
                return config

        raise KeyError(
            f"No load config with uid {uid!r}"
        )

    def _validate_unique_uids(self) -> None:
        seen = {}

        for name, config in self.configs.items():
            if config.uid in seen:
                raise ValueError(
                    f"Duplicate load-config uid "
                    f"{config.uid!r}: "
                    f"{seen[config.uid]!r} and {name!r}"
                )

            seen[config.uid] = name

    def select(self, name: str) -> LoadConfig:
        if name not in self.configs:
            raise KeyError(f"Unknown load config {name!r}")
        
        return self.configs[name].copy_for_session()

    def suggest_config_for(
        self,
        path: str | Path,
    ) -> LoadConfig | None:

        fmt = detect_file_format(path)

        # 1. Previous config for this format
        previous = self.last_used_by_format.get(
            fmt
        )

        if previous is not None:
            return previous.copy(
                new_identity=True
            )

        # 2. Default preset
        default = self.default_for(fmt)

        if default is not None:
            return default.copy_for_session()

        return None
    
    def default_for(
        self,
        file_format: FileFormat,
    ) -> LoadConfig | None:

        uid = self.default_by_format.get(file_format)
        if uid is None:
            return self.get_by_uid(self.default_fallback)

        return self.get_by_uid(uid)

    def remember_used_config(
        self,
        file_format: FileFormat,
        config: LoadConfig,
    ) -> None:
        self.last_used_by_format[file_format] = config.copy(new_identity=False)

    def set_default_for_format(
        self,
        path: str | Path,
        config: LoadConfig | str,
    ) -> None:
        """
        Set a stored config as the default for a file format.

        `config` may either be a LoadConfig or a config UID.
        """

        file_format = detect_file_format(path)
        if isinstance(config, LoadConfig):
            uid = config.preset_uid or config.uid
        else:
            uid = config

        # Make sure this UID actually belongs to a registered preset.
        self.get_by_uid(uid)
        if not self.is_registered(uid):
            raise ValueError(
                "Only stored load-config presets can be "
                "used as persistent defaults."
            )

        self.default_by_format[file_format] = uid
        self._save_defaults()

    def clear_default_for_format(
        self,
        file_format: FileFormat,
    ) -> None:

        self.default_by_format.pop(
            file_format,
            None,
        )

        self._save_defaults()

    def _save_defaults(self) -> None:

        self.defaults_file.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        data = {
            file_format.value: uid
            for file_format, uid
            in self.default_by_format.items()
        }
        with self.defaults_file.open(
            "w",
            encoding="utf-8",
        ) as fh:
            json.dump(
                data,
                fh,
                indent=2,
            )

    def _load_defaults(self) -> None:

        if not self.defaults_file.exists():
            self.default_by_format = {}
            return

        try:
            with self.defaults_file.open(
                "r",
                encoding="utf-8",
            ) as fh:
                data = json.load(fh)

        except (OSError, json.JSONDecodeError):
            return

        defaults = {}

        for format_name, uid in data.items():

            try:
                file_format = FileFormat(format_name)

            except ValueError:
                # Unknown/obsolete file format.
                continue

            try:
                # Ignore defaults whose config no longer exists.
                self.get_by_uid(uid)

            except KeyError:
                continue

            defaults[file_format] = uid

        self.default_by_format = defaults

    def is_registered(
        self,
        config: LoadConfig | str,
    ) -> bool:

        uid = (
            config.uid
            if isinstance(config, LoadConfig)
            else config
        )

        try:
            self.get_by_uid(uid)
            return True
        except KeyError:
            return False

    def is_default(
        self,
        path: str | Path | FileFormat,
        config: LoadConfig | str,
    ) -> bool:
        if isinstance(path, (str, Path)):
            file_format = detect_file_format(path)
        else:
            file_format = path

        uid = (
            config.preset_uid
            if isinstance(config, LoadConfig)
            else config
        )

        return (
            self.default_by_format.get(file_format)
            == uid
        )

    def modified(self, config: LoadConfig) -> bool:
        original_config = self.get_by_uid(config.preset_uid) if config.preset_uid else None
        
        if original_config is None:
            return True
        return config.to_dict(for_compare=True) != original_config.to_dict(for_compare=True)

    def source(self, name: str) -> str | None:
        return self.config_sources.get(name)

    def is_builtin(self, name: str) -> bool:
        return self.source(name) == "builtin"

    def is_user(self, name: str) -> bool:
        return self.source(name) == "user"

    def load_fields_from_config(
        self,
        name: str,
        groups: list[str] | None = None,
        *,
        enabled_only: bool = True,
    ):
        return self.configs[name].get_fields_to_load(
            groups,
            enabled_only=enabled_only,
        )

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save_config(
        self,
        config: LoadConfig,
        name: str,
    ) -> Path:
        if not isinstance(config, LoadConfig):
            raise TypeError("config must be an instance of LoadConfig")
        name = name.strip()
        if not name:
            raise ValueError("Config name must not be empty")

        packaged = self._packaged_configs.get(name)
        if packaged is not None and not packaged.public:
            raise ValueError(f"{name!r} is reserved by a CATAN-native config")
        # if name in self.configs:
        #     raise FileExistsError(f"A load configuration named {name!r} already exists")

        config.name = name
        config.public = True
        config.native = False
        
        # if config.native is None:

        # if config.public and not config.native:
        path = self.user_dir / f"{self._safe_filename(name)}.json"
        # else:
            # path = self._builtin_dir() / f"{self._safe_filename(name)}.json"

        # if path.exists() and not overwrite:
        #     raise FileExistsError(f"Config file already exists: {path}")
        config.save_json(path)

        self._hidden_builtins.discard(name)
        self._save_hidden_builtins()
        self.configs[name] = config.copy()
        self.config_sources[name] = "user"
        self.config_paths[name] = path
        return path

    def delete(self, name: str) -> None:
        """Delete/hide a non-native config.

        User JSON files are removed. Packaged non-native configs are hidden by a
        persistent tombstone in ``user_dir`` because installed package resources
        themselves are read-only.
        """
        if name not in self.configs and name not in self._packaged_configs:
            raise KeyError(f"Unknown load config {name!r}")

        config = self.configs.get(name) or self._packaged_configs[name]
        if config.native:
            raise ValueError("CATAN-native configs cannot be deleted")

        user_path = self.config_paths.get(name)
        if user_path is not None and user_path.exists():
            user_path.unlink()

        if name in self._packaged_configs:
            self._hidden_builtins.add(name)
            self._save_hidden_builtins()

        self.configs.pop(name, None)
        self.config_sources.pop(name, None)
        self.config_paths.pop(name, None)

    def restore_builtin(self, name: str) -> LoadConfig:
        """Restore a previously hidden non-native packaged config."""
        if name not in self._packaged_configs:
            raise KeyError(f"No packaged config named {name!r}")
        self._hidden_builtins.discard(name)
        self._save_hidden_builtins()

        # Remove an identically named user override so "restore" really means
        # return to the packaged schema.
        user_path = self.config_paths.get(name)
        if user_path is not None and user_path.exists():
            user_path.unlink()
        self.reload_configs()
        return self.select(name)

    # ------------------------------------------------------------------
    # IO/inspection conveniences for the GUI
    # ------------------------------------------------------------------
    @property
    def inspector(self):
        if self._inspector is None:
            from catan.core.io.inspection import FileInspector
            self._inspector = FileInspector()
        return self._inspector

    def browse_file_fields(
        self,
        path: str | Path,
        *,
        root: str = "/",
        subpath: str = "/",
        selected: str | None = None,
        refresh: bool = False,
    ):
        return self.inspector.browse(
            path,
            root=root,
            subpath=subpath,
            selected=selected,
            refresh=refresh,
        )

    def check_compatibility(
        self,
        path: str | Path,
        *,
        config: LoadConfig | None = None,
        root: str = "/",
        refresh: bool = False,
    ):
        if config is None:
            raise RuntimeError("No load config selected")
        return self.inspector.check_compatibility(
            path,
            config.get_fields_to_load(),
            root=root,
            refresh=refresh,
        )

    @staticmethod
    def _safe_filename(name: str) -> str:
        filename = "".join(
            char if (char.isalnum() or char in "-_") else "_"
            for char in name.strip()
        ).strip("_")
        return filename or "load_config"
