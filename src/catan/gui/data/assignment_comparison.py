from collections import Counter
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class AssignmentPartner:
    neuron_id: int
    shared: int
    jaccard: float
    distance: float


class AssignmentComparison:
    """Correspondence between assignments using the same loaded sessions."""

    def __init__(self, source, target, sessions):
        self.source = source
        self.target = target

        self.source_rows, _ = self._index(source, sessions)
        self.target_rows, self.target_owners = self._index(target, sessions)

    @staticmethod
    def _index(assignments, sessions):
        ids = np.asarray(assignments.ids)

        if ids.ndim != 2 or ids.shape[1] > len(sessions):
            raise ValueError("Assignments do not match the loaded session columns.")

        included = getattr(assignments.union, "included", None)
        keep = (
            np.ones(len(ids), dtype=bool)
            if included is None
            else np.asarray(included, dtype=bool)
        )

        if keep.shape != (len(ids),):
            raise ValueError("Assignment inclusion has the wrong length.")

        rows = [set() for _ in ids]
        owners = {}

        for sid in range(ids.shape[1]):
            session = sessions[sid]
            if session is None:
                continue

            mask = session.included

            for neuron in np.flatnonzero(keep):
                footprint = int(ids[neuron, sid])

                if footprint < 0:
                    continue

                if footprint >= session.n_neurons:
                    raise ValueError(f"Invalid footprint {footprint} in session {sid}.")

                if mask is not None and not mask[footprint]:
                    continue

                key = (sid, footprint)

                if key in owners:
                    raise ValueError(
                        f"Footprint {footprint} has multiple owners "
                        f"in session {sid}."
                    )

                owners[key] = int(neuron)
                rows[neuron].add(key)

        return rows, owners

    def partners(self, neuron_id):
        """Return all overlapping partners, strongest first."""
        if not 0 <= neuron_id < len(self.source_rows):
            return []

        row = self.source_rows[neuron_id]

        counts = Counter(
            self.target_owners[key] for key in row if key in self.target_owners
        )

        result = []

        for other, shared in counts.items():
            union_size = len(row) + len(self.target_rows[other]) - shared
            distance = float("inf")

            source_centroids = getattr(self.source.union, "centroids", None)
            target_centroids = getattr(self.target.union, "centroids", None)

            if (
                source_centroids is not None
                and target_centroids is not None
                and neuron_id < len(source_centroids)
                and other < len(target_centroids)
            ):
                value = float(
                    np.linalg.norm(
                        source_centroids[neuron_id] - target_centroids[other]
                    )
                )
                if np.isfinite(value):
                    distance = value

            result.append(
                AssignmentPartner(
                    neuron_id=other,
                    shared=shared,
                    jaccard=shared / union_size,
                    distance=distance,
                )
            )

        return sorted(
            result,
            key=lambda partner: (
                -partner.shared,
                -partner.jaccard,
                partner.distance,
                partner.neuron_id,
            ),
        )

    def translate(self, neuron_id, session_id=None):
        """Return (neuron, session), or None when no partner exists."""
        if not 0 <= neuron_id < len(self.source_rows):
            return None

        # A concrete footprint has an exact owner in the other set.
        if session_id is not None:
            for sid, footprint in self.source_rows[neuron_id]:
                if sid == session_id:
                    owner = self.target_owners.get((sid, footprint))
                    return None if owner is None else (owner, session_id)

            return None

        # A neuron-level selection follows its strongest partner.
        partners = self.partners(neuron_id)
        return None if not partners else (partners[0].neuron_id, None)
