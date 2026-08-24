
from PySide6.QtGui import QIcon
import qtawesome as qta

def get_fa_icon(
    name: str,
    color="white",
    *,
    variants: tuple[str, ...] = ("s", "r", "b"),
) -> QIcon:
    """
    Find a Font Awesome icon by name.

    Searches Font Awesome versions in this order:
        7 -> 6 -> 5

    Within each version, searches:
        solid -> regular -> brands

    Examples
    --------
    get_fa_icon("gear")
    get_fa_icon("folder-open")
    """

    prefixes = []

    for version in (7, 6, 5):
        for variant in variants:
            prefixes.append(
                f"fa{version}{variant}"
            )

    for prefix in prefixes:
        icon_name = f"{prefix}.{name}"

        try:
            return qta.icon(icon_name,color=color)
        except Exception:
            pass

    raise ValueError(
        f"Could not find Font Awesome icon {name!r} "
        f"in FA7, FA6 or FA5."
    )
