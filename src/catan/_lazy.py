from importlib import import_module


def install_exports(namespace, package, exports):
    def get_attribute(name):
        target = exports.get(name)

        if target is None:
            raise AttributeError(f"module {package!r} has no attribute {name!r}")

        module_name, attribute = target
        value = getattr(
            import_module(module_name, package),
            attribute,
        )

        namespace[name] = value
        return value

    namespace["__all__"] = list(exports)
    namespace["__getattr__"] = get_attribute
    namespace["__dir__"] = lambda: sorted(set(namespace) | set(exports))
