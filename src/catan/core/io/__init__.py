from catan._lazy import install_exports as _install_exports

_groups = {
    ".api": (
        "detect_file_format",
        "get_backend",
        "load_file",
        "load_from_config",
        "save_file",
        "save_from_config",
        "NATIVE_SESSION_OBJECT_TYPES",
        "NATIVE_SESSION_CONFIG",
        "NATIVE_REMAP_CONFIG",
        "NATIVE_MODEL_CONFIG",
        "NATIVE_ASSIGNMENTS_CONFIG",
        "load_fields_from_sources",
        "resolve_source_path",
    ),
    ".inspection": (
        "FileInspector",
        "inspect_file",
        "browse_file_fields",
        "check_fields_compatibility",
        "evaluate_fields_compatibility",
        "check_file_compatibility",
        "evaluate_file_compatibility",
        "clear_inspection_cache",
    ),
    ".types": (
        "CompatibilityReport",
        "MultiSourceCompatibilityReport",
        "FieldCompatibility",
        "FieldInfo",
        "FileFormat",
        "FileStructure",
    ),
}

_install_exports(
    globals(),
    __name__,
    {name: (module, name) for module, names in _groups.items() for name in names},
)
