"""Do the other sessions see a footprint region as one cell or as two?

A footprint split and a footprint merge describe the same geometry: a
large footprint ("whole", one session) covering two smaller ones
("pieces", other sessions). Which reading is right is decided by the other
sessions. Per session (not the whole's): its included footprints lying
mostly inside the whole vote

- "joint" when one of them covers both pieces,
- "separate" when one covers the first piece and another the second.

A split (the whole is two merged cells) needs more separate than joint
votes, a merge (the pieces are fragments of one cell) more joint than
separate votes, so a configuration can never be proposed both ways.
"""

from dataclasses import dataclass

import numpy as np

# Footprint support: pixels at or above this fraction of its peak.
SUPPORT_FRACTION = 0.2

# A session's footprint takes part when this fraction of its support lies
# inside the whole.
INSIDE_FRACTION = 0.5

# A footprint covers a piece with this fraction of the piece's support.
COVER_FRACTION = 0.5

JOINT, SEPARATE, OTHER, EMPTY = "J", "S", "-", "."


@dataclass(frozen=True)
class SessionVotes:
    # One letter per other session in session order: J joint, S separate,
    # "-" footprints there but neither, "." none inside the whole.
    votes: str

    @property
    def joint(self):
        return self.votes.count(JOINT)

    @property
    def separate(self):
        return self.votes.count(SEPARATE)

    def margin(self, kind):
        """Votes for the manipulation minus votes against it."""
        if kind == "split":
            return self.separate - self.joint
        return self.joint - self.separate

    def supports(self, kind):
        return self.margin(kind) > 0

    def describe(self):
        return f"{self.separate} separate, {self.joint} joint"


class SessionVoter:
    """Votes for configurations of one data state (supports cached)."""

    def __init__(self, sessions, ids):
        self.sessions = sessions
        self.ids = np.asarray(ids)
        self._supports = {}
        self._cache = {}
        # Per whole: each other session's footprints inside it (their
        # supports as columns).
        self._members = {}
        # Per (whole, piece): per session, which members cover the piece.
        self._coverage = {}

    def _session_supports(self, session_id):
        supports = self._supports.get(session_id)

        if supports is None:
            footprints = self.sessions[session_id].footprints.tocsc()
            peaks = np.asarray(footprints.max(axis=0).todense()).ravel()
            supports = footprints.copy()
            supports.data = (
                supports.data
                >= SUPPORT_FRACTION * np.repeat(peaks, np.diff(supports.indptr))
            ).astype(np.float32)
            supports.eliminate_zeros()
            sizes = np.asarray(supports.sum(axis=0)).ravel()
            included = getattr(self.sessions[session_id], "included", None)
            included = (
                np.ones(len(sizes), dtype=bool)
                if included is None
                else np.asarray(included, dtype=bool)[: len(sizes)]
            )
            supports = (supports, sizes, included)
            self._supports[session_id] = supports

        return supports

    def _support(self, component):
        """Dense 0/1 support of a footprint (from its CSC column: the
        supports are binary without explicit zeros)."""
        supports, _, _ = self._session_supports(component.session_id)
        footprint = int(self.ids[component.neuron_id, component.session_id])
        dense = np.zeros(supports.shape[0], dtype=np.float32)
        start, stop = supports.indptr[footprint], supports.indptr[footprint + 1]
        dense[supports.indices[start:stop]] = 1.0
        return dense


    def _inside_members(self, whole):
        """Per session (None for the whole's own): the supports of the
        included footprints lying mostly inside ``whole`` (one row each),
        or an empty tuple without any. The same for every pair of pieces."""
        members_by_session = self._members.get(whole)
        if members_by_session is not None:
            return members_by_session

        whole_support = self._support(whole)
        members_by_session = []
        for session_id in range(self.ids.shape[1]):
            if session_id == whole.session_id:
                members_by_session.append(None)
                continue

            supports, sizes, included = self._session_supports(session_id)
            inside = np.asarray(supports.T @ whole_support).ravel()
            members = np.flatnonzero(
                included & (sizes > 0) & (inside >= INSIDE_FRACTION * sizes)
            )
            # Columns = members (CSC: small; a CSR copy would hold one
            # index entry per image pixel).
            members_by_session.append(supports[:, members] if len(members) else ())

        self._members[whole] = members_by_session
        return members_by_session

    def _covers(self, whole, piece):
        """Per session (None for the whole's own, () without members): for
        each member of ``whole`` there, whether it covers ``piece``.
        Computed once per piece, shared by all pairs it belongs to."""
        key = (whole, piece)
        covers = self._coverage.get(key)
        if covers is not None:
            return covers

        support = self._support(piece)
        size = max(float(support.sum()), 1.0)
        # members.T (CSR view of the CSC columns) @ dense: cost of the
        # members' few nonzeros only.
        covers = [
            members
            if members is None or isinstance(members, tuple)
            else np.asarray(members.T @ support).ravel() / size >= COVER_FRACTION
            for members in self._inside_members(whole)
        ]
        self._coverage[key] = covers
        return covers

    def votes(self, whole, pieces):
        """SessionVotes of ``whole`` (a footprint) split into the two
        footprints ``pieces``."""
        key = (whole, frozenset(pieces))
        cached = self._cache.get(key)
        if cached is not None:
            return cached

        first, second = (self._covers(whole, piece) for piece in pieces)

        votes = []
        for a, b in zip(first, second):
            if a is None:
                continue

            if isinstance(a, tuple):
                votes.append(EMPTY)
                continue

            covered = np.column_stack([a, b])

            if np.any(covered.all(axis=1)):
                votes.append(JOINT)
            elif covered[:, 0].any() and covered[:, 1].any():
                votes.append(SEPARATE)
            else:
                votes.append(OTHER)

        result = SessionVotes("".join(votes))
        self._cache[key] = result
        return result
