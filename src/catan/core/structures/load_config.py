from __future__ import annotations

from typing import Literal
from dataclasses import dataclass, field
from pathlib import Path
from importlib.resources import files
import json



@dataclass
class FieldSpec:
    path: str
    source: Literal["dataset", "attribute"] = "dataset"
    attribute: str | None = None
    required: bool = False

    def to_dict(self) -> dict:
        return {
            "path": self.path,
            "source": self.source,
            "attribute": self.attribute,
            "required": self.required,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "FieldSpec":
        return cls(
            path=data["path"],
            source=data.get("source", "dataset"),
            attribute=data.get("attribute"),
            required=data.get("required", False),
        )


@dataclass
class FieldGroupSpec:
    title: str
    type: Literal["static", "dynamic"]
    fields: dict[str, FieldSpec] = field(default_factory=dict)
    enabled: bool = True

    def add_field(
        self,
        name: str,
        spec: FieldSpec,
    ) -> None:
        if self.type != "dynamic":
            raise ValueError(
                f"Cannot add fields to static group {self.title!r}."
            )

        if name in self.fields:
            raise KeyError(
                f"Field {name!r} already exists."
            )

        self.fields[name] = spec

    def remove_field(
        self,
        name: str,
    ) -> None:
        if self.type != "dynamic":
            raise ValueError(
                f"Cannot remove fields from static group {self.title!r}."
            )

        if name not in self.fields:
            raise KeyError(
                f"Unknown field {name!r}."
            )

        self.fields.pop(name)

    def rename_field(
        self,
        old_name: str,
        new_name: str,
    ) -> None:
        if self.type != "dynamic":
            raise ValueError(
                f"Cannot rename fields in static group {self.title!r}."
            )

        if old_name not in self.fields:
            raise KeyError(
                f"Unknown field {old_name!r}."
            )

        if new_name in self.fields:
            raise KeyError(
                f"Field {new_name!r} already exists."
            )

        self.fields[new_name] = self.fields.pop(
            old_name
        )

    def set_field_path(
        self,
        name: str,
        path: str,
    ) -> None:
        if name not in self.fields:
            raise KeyError(
                f"Unknown field {name!r}."
            )

        self.fields[name].path = path

    def to_dict(self) -> dict:
        return {
            "title": self.title,
            "type": self.type,
            "enabled": self.enabled,
            "fields": {
                key: spec.to_dict()
                for key, spec in self.fields.items()
            },
        }

    @classmethod
    def from_dict(cls, data: dict) -> "FieldGroupSpec":
        return cls(
            title=data["title"],
            type=data["type"],
            enabled=data.get("enabled", True),
            fields={
                key: FieldSpec.from_dict(spec)
                for key, spec in data.get("fields", {}).items()
            },
        )


@dataclass
class LoadConfig:
    name: str
    source_type: str = "session"
    public: bool = False
    groups: dict[str, FieldGroupSpec] = field(default_factory=dict)

    FORMAT_VERSION = 1

    def get_group(
        self,
        group: str,
    ) -> FieldGroupSpec:
        if group not in self.groups:
            raise KeyError(
                f"Unknown field group {group!r}."
            )

        return self.groups[group]

    def set_group_enabled(
        self,
        group: str,
        enabled: bool,
    ) -> None:
        self.get_group(group).enabled = enabled

    def set_field_path(
        self,
        group: str,
        field_name: str,
        path: str,
    ) -> None:
        self.get_group(group).set_field_path(
            field_name,
            path,
        )

    def add_field(
        self,
        group: str,
        field_name: str,
        path: str,
        *,
        source: Literal[
            "dataset",
            "attribute",
        ] = "dataset",
        attribute: str | None = None,
        required: bool = False,
    ) -> None:
        spec = FieldSpec(
            path=path,
            source=source,
            attribute=attribute,
            required=required,
        )

        self.get_group(group).add_field(
            field_name,
            spec,
        )

    def remove_field(
        self,
        group: str,
        field_name: str,
    ) -> None:
        self.get_group(group).remove_field(
            field_name
        )

    def rename_field(
        self,
        group: str,
        old_name: str,
        new_name: str,
    ) -> None:
        self.get_group(group).rename_field(
            old_name,
            new_name,
        )
        
    def get_fields_to_load(
        self,
        groups: list[str] | None = None,
    ) -> dict[str, dict[str, FieldSpec]]:
        if groups is None:
            groups = [
                name
                for name, group in self.groups.items()
                if group.enabled
            ]

        return {
            group_name: dict(
                self.groups[group_name].fields
            )
            for group_name in groups
        }
    
    def update_field(
        self,
        group: str,
        field_name: str,
        **changes,
    ) -> None:
        spec = self.get_group(group).fields[field_name]

        valid = {
            "path",
            "source",
            "attribute",
            "required",
        }

        unknown = set(changes) - valid

        if unknown:
            raise ValueError(
                f"Unknown FieldSpec properties: {unknown}"
            )

        for key, value in changes.items():
            setattr(spec, key, value)


    def to_dict(self) -> dict:
        return {
            "format_version": self.FORMAT_VERSION,
            "name": self.name,
            "source_type": self.source_type,
            "groups": {
                key: group.to_dict()
                for key, group in self.groups.items()
            },
        }

    @classmethod
    def from_dict(cls, data: dict) -> "LoadConfig":
        version = data.get("format_version", 1)

        if version != cls.FORMAT_VERSION:
            raise ValueError(
                f"Unsupported load-config format version {version}. "
                f"Expected {cls.FORMAT_VERSION}."
            )

        return cls(
            name=data.get("name", "Unnamed config"),
            source_type=data.get("source_type", "session"),
            public=data.get("public", False),
            groups={
                key: FieldGroupSpec.from_dict(group)
                for key, group in data.get("groups", {}).items()
            },
        )

    @classmethod
    def load_resource(
        cls,
        filename: str,
    ) -> "LoadConfig":

        resource = (
            files("catan")
            / "resources"
            / "load_configs"
            / filename
        )

        with resource.open(
            "r",
            encoding="utf-8",
        ) as f:
            data = json.load(f)

        return cls.from_dict(data)

    @classmethod
    def fields_from_resource(
        cls,
        filename: str,
        groups: list[str] | None = None,
    ):
        config = cls.load_resource(filename)

        return config.get_fields_to_load(
            groups=groups
        )

    @classmethod
    def fields_from_json(
        cls,
        path: str | Path,
        groups: list[str] | None = None,
    ):
        config = cls.load_json(path)

        return config.get_fields_to_load(
            groups=groups
        )

    def save_json(
        self,
        path: str | Path,
    ) -> None:
        path = Path(path)

        path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        with path.open(
            "w",
            encoding="utf-8",
        ) as file:
            json.dump(
                self.to_dict(),
                file,
                indent=2,
                ensure_ascii=False,
            )

    @classmethod
    def load_json(
        cls,
        path: str | Path,
    ) -> "LoadConfig":
        path = Path(path)

        with path.open(
            "r",
            encoding="utf-8",
        ) as file:
            data = json.load(file)

        if not isinstance(data, dict):
            raise ValueError(
                f"Invalid load config in {path}: "
                "top-level JSON object must be a dictionary."
            )

        return cls.from_dict(data)