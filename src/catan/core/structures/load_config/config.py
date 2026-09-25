from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, field
from importlib.resources import files
from pathlib import Path
from typing import Literal
import json
from uuid import uuid4

from catan.core.io.types import SourceTypes, FieldSource, FieldSpec, FieldGroupSpec


@dataclass
class SubConfigSpec:
    path: str
    config: str
    required: bool = False

    def to_dict(self) -> dict:
        return {
            "path": self.path,
            "config": self.config,
            "required": self.required,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "SubConfigSpec":
        return cls(
            path=data["path"],
            config=data["config"],
            required=data.get("required", False),
        )


@dataclass
class LoadConfig:
    name: str | None = None
    uid: str = field(default_factory=lambda: str(uuid4()))
    preset_uid: str | None = None

    source_type: SourceTypes = "session"

    public: bool = False
    native: bool = False

    groups: dict[str, FieldGroupSpec] = field(default_factory=dict)

    subconfigs: dict[str, SubConfigSpec] = field(default_factory=dict)

    FORMAT_VERSION = 1

    def get_group(self, group: str) -> FieldGroupSpec:
        try:
            return self.groups[group]
        except KeyError as exc:
            raise KeyError(f"Unknown field group {group!r}") from exc

    def set_group_enabled(self, group: str, enabled: bool) -> None:
        self.get_group(group).enabled = enabled

    def set_field_path(self, group: str, field_name: str, path: str) -> None:
        self.get_group(group).set_field_path(field_name, path)

    def add_field(
        self,
        group: str,
        field_name: str,
        path: str,
        *,
        source: FieldSource = "dataset",
        attribute: str | None = None,
        required: bool = False,
        source_path: str | None = None,
    ) -> None:
        self.get_group(group).add_field(
            field_name,
            FieldSpec(
                path=path,
                source=source,
                attribute=attribute,
                required=required,
                source_path=source_path,
            ),
        )

    def remove_field(self, group: str, field_name: str) -> None:
        self.get_group(group).remove_field(field_name)

    def rename_field(self, group: str, old_name: str, new_name: str) -> None:
        self.get_group(group).rename_field(old_name, new_name)

    def update_field(self, group: str, field_name: str, **changes) -> None:
        spec = self.get_group(group).fields[field_name]
        valid = {"path", "source", "attribute", "required", "source_path"}
        unknown = set(changes) - valid
        if unknown:
            raise ValueError(f"Unknown FieldSpec properties: {sorted(unknown)}")
        for key, value in changes.items():
            setattr(spec, key, value)

    def get_fields_to_load(
        self,
        groups: list[str] | None = None,
        *,
        enabled_only: bool = True,
    ) -> dict[str, dict[str, FieldSpec]]:
        if groups is None:
            groups = [
                name
                for name, group in self.groups.items()
                if (group.enabled or not enabled_only)
            ]
        return {
            group_name: dict(self.groups[group_name].fields) for group_name in groups
        }

    def get_fields_to_save(
        self,
        groups: list[str] | None = None,
        *,
        enabled_only: bool = True,
    ) -> dict[str, dict[str, FieldSpec]]:
        # Same structure, different semantic use. Keeping one canonical mapping
        # guarantees CATAN-native read/write round trips use identical paths.
        return self.get_fields_to_load(groups, enabled_only=enabled_only)

    def copy(
        self,
        *,
        new_identity: bool = True,
    ) -> "LoadConfig":

        config = deepcopy(self)

        if new_identity:
            config.uid = str(uuid4())

        return config

    def copy_for_session(self) -> "LoadConfig":
        config = deepcopy(self)

        config.preset_uid = self.uid
        config.uid = str(uuid4())

        return config

    def to_dict(self, for_compare=False) -> dict:

        out = {
            "format_version": self.FORMAT_VERSION,
            "source_type": self.source_type,
            "public": self.public,
            "native": self.native,
            "groups": {name: group.to_dict() for name, group in self.groups.items()},
            "subconfigs": {
                name: spec.to_dict() for name, spec in self.subconfigs.items()
            },
        }

        if for_compare:
            return out
        else:
            return out | {
                "uid": self.uid,
                "name": self.name,
            }

    @classmethod
    def from_dict(
        cls,
        data: dict,
    ) -> "LoadConfig":

        version = data.get(
            "format_version",
            1,
        )

        if version != cls.FORMAT_VERSION:
            raise ValueError(
                f"Unsupported load-config format version {version}; "
                f"expected {cls.FORMAT_VERSION}"
            )

        return cls(
            uid=data.get(
                "uid",
                str(uuid4()),
            ),
            name=data.get(
                "name",
                "Unnamed config",
            ),
            source_type=data.get(
                "source_type",
                "session",
            ),
            public=data.get(
                "public",
                False,
            ),
            native=data.get(
                "native",
                False,
            ),
            groups={
                name: FieldGroupSpec.from_dict(group)
                for name, group in data.get("groups", {}).items()
            },
            subconfigs={
                name: SubConfigSpec.from_dict(spec)
                for name, spec in data.get("subconfigs", {}).items()
            },
        )

    @classmethod
    def load_resource(cls, filename: str) -> "LoadConfig":
        resource = files("catan") / "resources" / "load_configs" / filename
        with resource.open("r", encoding="utf-8") as fh:
            return cls.from_dict(json.load(fh))

    @classmethod
    def fields_from_resource(
        cls,
        filename: str,
        groups: list[str] | None = None,
        *,
        enabled_only: bool = True,
    ) -> dict[str, dict[str, FieldSpec]]:
        return cls.load_resource(filename).get_fields_to_load(
            groups,
            enabled_only=enabled_only,
        )

    @classmethod
    def fields_from_json(
        cls,
        path: str | Path,
        groups: list[str] | None = None,
        *,
        enabled_only: bool = True,
    ) -> dict[str, dict[str, FieldSpec]]:
        return cls.load_json(path).get_fields_to_load(
            groups,
            enabled_only=enabled_only,
        )

    def save_json(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as fh:
            json.dump(self.to_dict(), fh, indent=2, ensure_ascii=False)

    @classmethod
    def load_json(cls, path: str | Path) -> "LoadConfig":
        path = Path(path)
        with path.open("r", encoding="utf-8") as fh:
            data = json.load(fh)
        if not isinstance(data, dict):
            raise ValueError(
                f"Invalid load config in {path}: top-level JSON must be an object"
            )
        return cls.from_dict(data)

    def get_subconfig(
        self,
        name: str,
    ) -> SubConfigSpec:
        try:
            return self.subconfigs[name]
        except KeyError as exc:
            raise KeyError(f"Unknown subconfig {name!r}") from exc

    def add_subconfig(
        self,
        name: str,
        path: str,
        config: str,
        *,
        required: bool = False,
    ) -> None:

        if name in self.subconfigs:
            raise KeyError(f"Subconfig {name!r} already exists")

        self.subconfigs[name] = SubConfigSpec(
            path=path,
            config=config,
            required=required,
        )

    def remove_subconfig(
        self,
        name: str,
    ) -> None:
        if name not in self.subconfigs:
            raise KeyError(f"Unknown subconfig {name!r}")

        del self.subconfigs[name]

    def update_subconfig(
        self,
        name: str,
        **changes,
    ) -> None:

        spec = self.get_subconfig(name)

        valid = {
            "path",
            "config",
            "required",
        }

        unknown = set(changes) - valid

        if unknown:
            raise ValueError(f"Unknown SubConfigSpec properties: " f"{sorted(unknown)}")

        for key, value in changes.items():
            setattr(
                spec,
                key,
                value,
            )
