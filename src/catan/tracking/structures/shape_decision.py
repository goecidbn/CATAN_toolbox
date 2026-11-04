"""Record of a shape decision ("two cells" / "one cell"); the logic is in
catan.gui.data.shape_decisions."""

from dataclasses import dataclass

import numpy as np

KINDS = ("split", "merge")
LABELS = {"split": "two cells", "merge": "one cell"}


@dataclass(frozen=True)
class ShapeDecision:
    kind: str  # "split" (two cells) or "merge" (one cell)
    whole: tuple  # (session ID, footprint ID)
    pieces: tuple  # two (session ID, footprint ID)

    def __post_init__(self):
        if self.kind not in KINDS:
            raise ValueError(f"Unknown shape decision {self.kind!r}.")
        if len(self.pieces) != 2:
            raise ValueError("A shape decision needs exactly two pieces.")

    @property
    def label(self):
        return LABELS[self.kind]

    def to_record(self):
        return {
            "kind": self.kind,
            "refs": np.asarray([self.whole, *self.pieces], dtype=np.int64),
        }

    @classmethod
    def from_record(cls, record):
        refs = np.asarray(record["refs"], dtype=np.int64).reshape(3, 2)
        kind = record["kind"]
        if isinstance(kind, bytes):
            kind = kind.decode()
        return cls(
            str(kind),
            tuple(int(v) for v in refs[0]),
            tuple(tuple(int(v) for v in ref) for ref in refs[1:]),
        )
