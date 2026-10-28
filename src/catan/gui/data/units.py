"""Display units of distances and areas.

All data, statistics, model parameters and evidence are in px. Shown
values are converted by the pixel size (µm/px) when distances are
displayed in µm; switching the unit or the pixel size only rescales.
"""

from dataclasses import dataclass

# value_kind (StatisticDefinition.value_kind): power of the length unit.
UNIT_POWERS = {"distance": 1, "area": 2}

UNIT_LABELS = {
    ("distance", False): "px",
    ("distance", True): "µm",
    ("area", False): "px²",
    ("area", True): "µm²",
}


@dataclass(frozen=True)
class DisplayUnits:
    um_per_px: float = 1.0
    show_um: bool = False

    def factor(self, kind):
        """px value × factor = displayed value (1 for other kinds)."""
        power = UNIT_POWERS.get(kind)
        if power is None or not self.show_um:
            return 1.0
        return float(self.um_per_px) ** power

    def label(self, kind):
        return UNIT_LABELS.get((kind, self.show_um), "")

    def to_display(self, value, kind):
        return value * self.factor(kind)

    def from_display(self, value, kind):
        return value / self.factor(kind)

    def convert(self, value, kind, unit):
        """``value`` given in ``unit`` ("px"/"µm" or "px²"/"µm²") as a
        displayed value."""
        power = UNIT_POWERS.get(kind)
        if power is None or unit is None:
            return value
        px = value / float(self.um_per_px) ** power if "µm" in unit else value
        return self.to_display(px, kind)
