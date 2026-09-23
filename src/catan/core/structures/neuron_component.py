from dataclasses import dataclass
from typing import Tuple


@dataclass(frozen=True, slots=True)
class NeuronComponent:
    neuron_id: int
    session_id: int | None

    @property
    def id(self) -> Tuple[int, int | None]:
        # return self.session_id, self.neuron_id
        return int(self.neuron_id), (
            int(self.session_id) if self.session_id is not None else None
        )

    def __iter__(self):
        yield int(self.neuron_id)
        yield int(self.session_id) if self.session_id is not None else None
