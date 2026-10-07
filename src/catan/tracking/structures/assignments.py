from dataclasses import dataclass
from typing import Any, Dict, Optional, Literal, List
from pathlib import Path
from copy import deepcopy

from catan.core.io import load_file, save_file, NATIVE_ASSIGNMENTS_CONFIG
from catan.core.structures.load_config import LoadConfig, FieldSpec

from catan.core.utils import pad_axis
from catan.core.structures.session import SessionData

import numpy as np
from scipy import sparse
from enum import IntEnum


class ReviewStatus(IntEnum):
    PENDING = 0
    REVIEWED = 1
    UNCERTAIN = 2

    @property
    def label(self) -> str:
        return self.name.replace("_", " ").capitalize()

    @property
    def shortcut(self) -> str:
        return {
            ReviewStatus.PENDING: "P",
            ReviewStatus.REVIEWED: "R",
            ReviewStatus.UNCERTAIN: "U",
        }[self]

    @property
    def menu_label(self) -> str:
        """
        Label with Qt mnemonic marker (&) on the
        shortcut letter.
        """
        label = self.label

        idx = label.lower().find(self.shortcut.lower())

        if idx < 0:
            return label

        return label[:idx] + "&" + label[idx:]


@dataclass(frozen=True, slots=True)
class FootprintRef:
    session_id: int
    footprint_id: int


@dataclass(slots=True)
class DetectionCorrection:
    operation: Literal["merge", "split"]

    # Components in the session that is to be re-detected.
    sources: tuple[FootprintRef, ...]

    # Footprints chosen as templates/seeds from other sessions.
    seeds: tuple[FootprintRef, ...]


class Assignments:
    """
    Class to store and manage neuron assignments across sessions.
    """

    HDF5_VERSION = "1.0"

    source_type: str = "assignments"
    source_config: LoadConfig | None = None

    union: Optional[SessionData]

    def __init__(self):

        self.path = ""
        self.source_config = None

        self.status: dict[str, bool] = {"loaded": False}

        self.reset()

    def reset(self):
        """
        Reset the assignments to an empty state.
        """
        self.ids: np.ndarray = np.zeros((0, 0), int)  # nNeurons x nSessions

        # Initialize all neurons as pending
        self.review_status = np.full(0, ReviewStatus.PENDING, dtype=np.uint8)

        self.stats: Dict[str, np.ndarray] = {
            "p_matched": np.zeros((0, 0, 2), float),
            "shifts": np.zeros((0, 0, 2), float),  # nNeurons x nSessions x 2 (x,y)
            "fp_corr": np.zeros((0, 0), float),
        }
        self.stats_default_value = {
            "p_matched": (1.0, np.nan),
            "shifts": 0.0,
            "fp_corr": 1.0,
        }

        self.matched_status: np.ndarray = np.array([], dtype=bool)

        self.manipulations: dict[int, dict] = {}
        # Latest manipulation affecting each tracked neuron.
        self.manipulation_id: dict[int, int] = {}
        self.next_manipulation_id = 0

        self.union = SessionData(name="union")

    def copy(self) -> "Assignments":
        """
        Create a deep copy of the Assignments instance.
        """
        new_instance = Assignments()
        new_instance.ids = self.ids.copy()
        new_instance.review_status = self.review_status.copy()
        new_instance.stats = {k: v.copy() for k, v in self.stats.items()}
        new_instance.stats_default_value = self.stats_default_value.copy()
        # new_instance.union = self.union.copy() if self.union is not None else None
        return new_instance

    def copy_prefix(self, from_session_id: int):
        """Copy assignments before from_session_id, preserving session columns."""
        n_sessions = self.ids.shape[1]

        if not 0 <= from_session_id <= n_sessions:
            raise IndexError(f"Invalid session_id {from_session_id}.")

        keep = np.any(
            self.ids[:, :from_session_id] >= 0,
            axis=1,
        )

        neuron_id_map = {
            int(old_id): new_id for new_id, old_id in enumerate(np.flatnonzero(keep))
        }

        result = Assignments()
        result.path = self.path
        result.source_config = self.source_config
        result.status = self.status.copy()

        result.ids = self.ids[keep].copy()
        result.ids[:, from_session_id:] = -1

        result.stats = {key: values[keep].copy() for key, values in self.stats.items()}
        for values in result.stats.values():
            values[:, from_session_id:, ...] = np.nan

        result.stats_default_value = deepcopy(self.stats_default_value)
        result.review_status = self.review_status[keep].copy()

        result.matched_status = self.matched_status.copy()
        result.matched_status[from_session_id:] = False

        # Retain history; remap the lookup for surviving neuron identities.
        result.manipulations = deepcopy(self.manipulations)
        result.next_manipulation_id = self.next_manipulation_id
        result.manipulation_id = {
            neuron_id_map[old_id]: manipulation_id
            for old_id, manipulation_id in self.manipulation_id.items()
            if old_id in neuron_id_map
        }

        # result.union is deliberately empty.
        # Rebuild it from the retained sessions before registering the suffix.
        return result, neuron_id_map

    def pad_empty(self, n_neurons: int, n_sessions: int, mode="new"):
        """
        Pad the assignments and stats arrays to accommodate new neurons and sessions.
        """
        self.ids = pad_axis(self.ids, (n_neurons, n_sessions), -1)

        for key in self.stats.keys():
            dims = [n_neurons, n_sessions]
            for _ in range(self.stats[key].ndim - 2):
                dims.append(0)

            self.stats[key] = pad_axis(
                self.stats[key],
                tuple(dims),
                self.stats_default_value[key] if mode == "new" else np.nan,
            )

        self.matched_status = pad_axis(self.matched_status, (n_sessions,), False)

        self.review_status = pad_axis(
            self.review_status, (n_neurons,), ReviewStatus.PENDING
        )

    def move_session(self, session_id: int, new_session_id: int):
        """
        need to move code over
        """

        if new_session_id >= 0:
            # Move the session's data in assignments and tracking
            self.ids = move_single_row(self.ids, session_id, new_session_id)
            for key in self.stats:
                self.stats[key] = move_single_row(
                    self.stats[key], session_id, new_session_id
                )
            self.matched_status = move_single_row(
                self.matched_status, session_id, new_session_id
            )
        else:
            self.ids = np.delete(self.ids, session_id, axis=1)
            for key in self.stats:
                self.stats[key] = np.delete(self.stats[key], session_id, axis=1)

            self.matched_status = np.delete(self.matched_status, session_id, axis=0)

        self.updating_neuron_presence()  # Update neuron presence and clean union data

    def unassign_neurons(self, session_id: int):
        """
        Removes a session's neurons from the union data and updates assignments and tracking accordingly.
        """
        if session_id < 0 or session_id >= self.ids.shape[1]:
            raise ValueError(
                "Invalid session_id. It must be within the range of existing sessions."
            )

        # Mark the session as unmatched

        # print(f"Session {session_id} has been unregistered. Updating union data...")

        # Mark assignments and tracking stats in this session as unassigned
        self.ids[:, session_id] = -1

        for key in self.stats.keys():
            self.stats[key][:, session_id, ...] = np.nan

        self.matched_status[session_id] = False

        self.updating_neuron_presence()  # Update neuron presence and clean union data

    def updating_neuron_presence(self):

        ## remove neurons that are no longer present in any session
        neuron_presence = (self.ids >= 0).sum(axis=1) > 0
        if np.all(neuron_presence):
            return

        print(neuron_presence.shape)

        self.ids = self.ids[neuron_presence, :]

        self.review_status = self.review_status[neuron_presence]
        for key in self.stats.keys():
            self.stats[key] = self.stats[key][neuron_presence, ...]

        ## could just rebuild it entirely from the remaining sessions, but for now just remove the columns of empty neurons
        if self.union is None:
            return

        footprints_cleaned = sparse.hstack(
            [
                self.union.footprints[:, i]
                for i in range(self.union.n_neurons)
                if neuron_presence[i]
            ],
            format="csc",
        )
        self.union.update_footprints(
            footprints=footprints_cleaned,
            mode="replace",
            included_values=self.union.included[neuron_presence],
            synthetic_values=self.union.synthetic[neuron_presence],
        )

    def register_manipulation(
        self,
        *,
        manipulation_type: str,
        origin: list[dict],
        sources: list[dict],
        results: list[dict],
        affected_neurons,
    ) -> int:

        manipulation_id = self.next_manipulation_id
        self.next_manipulation_id += 1

        self.manipulations[manipulation_id] = {
            "type": manipulation_type,
            "origin": origin,
            "sources": sources,
            "results": results,
        }

        for neuron_id in affected_neurons:
            self.manipulation_id[int(neuron_id)] = manipulation_id

        return manipulation_id

    @staticmethod
    def _from_file(
        path: str | Path,
        fields_to_load: dict[str, dict[str, FieldSpec]] | None = None,
    ) -> "Assignments":
        data = load_file(
            path, fields_to_load, config_name=NATIVE_ASSIGNMENTS_CONFIG, root="/"
        )
        return Assignments._from_dict(data)

    @staticmethod
    def _from_dict(data: dict) -> "Assignments":
        assignments = Assignments()
        assignments.register_data(**data)
        return assignments

    def load(self):
        from catan.core.io.api import read_assignments_source

        if self.source_config is None:
            raise ValueError("Source config is not set.")

        result = read_assignments_source(
            self.path,
            self.source_config.get_fields_to_load(),
            load_data=True,
        )

        if result["problems"]:
            raise ValueError("\n\n".join(result["problems"]))

        self.register_data(**result["data"])

    def register_data(self, **data):
        ids = data.get("assignments", {}).get("ids")

        if ids is None:
            raise ValueError("The required assignments.ids field was not loaded.")

        ids = np.asarray(ids)

        if ids.ndim != 2:
            raise ValueError(
                "Assignments IDs must be a two-dimensional array "
                f"(neurons × sessions); received shape {ids.shape}."
            )

        if ids.dtype.kind not in "iuf":
            raise ValueError(
                f"Assignments IDs must be numeric; received dtype {ids.dtype}."
            )

        if np.isinf(ids).any():
            raise ValueError(
                "Assignments IDs contain infinite values. "
                "Use NaN or -1 for unmatched entries."
            )

        finite = np.isfinite(ids)
        values = ids[finite]

        if np.any(values != np.floor(values)) or np.any(values < -1):
            raise ValueError(
                "Assignments IDs must be non-negative integers. "
                "Use NaN or -1 for unmatched entries. "
                "Check the selected field and its indexing convention."
            )

        # CATAN stores IDs as signed integers; reserve -1 for unmatched entries.
        if values.size and np.any(values >= float(2**63)):
            raise ValueError("Assignments IDs exceed the supported integer range.")

        self.ids = np.where(finite, ids, -1).astype(np.int64)
        n_neurons, n_sessions = self.ids.shape

        stats = data.get("stats") or {}
        if not isinstance(stats, dict):
            raise ValueError("Assignment statistics must be a dictionary.")

        # Supply the normal defaults when an external file contains only IDs.
        self.stats = {
            "p_matched": np.broadcast_to(
                np.array([1.0, np.nan]),
                (n_neurons, n_sessions, 2),
            ).copy(),
            "shifts": np.zeros((n_neurons, n_sessions, 2), dtype=float),
            "fp_corr": np.ones((n_neurons, n_sessions), dtype=float),
        }

        for name, values in stats.items():
            if values is None:
                continue

            values = np.asarray(values)
            if values.ndim < 2 or values.shape[:2] != self.ids.shape:
                raise ValueError(
                    f"Statistic {name!r} has shape {values.shape}; "
                    f"its first two dimensions must match IDs {self.ids.shape}."
                )

            self.stats[name] = values

        self.matched_status = np.any(self.ids >= 0, axis=0)

        curation = data.get("curation") or {}
        review_status = curation.get("review_status")

        if review_status is None:
            review_status = np.full(n_neurons, ReviewStatus.PENDING, dtype=np.uint8)
        else:
            review_status = np.asarray(review_status)
            if review_status.shape != (n_neurons,):
                raise ValueError(
                    "Saved review status does not match the assignments: "
                    f"shape {review_status.shape}, expected {(n_neurons,)}."
                )

        self.review_status = review_status
        self._deserialize_manipulation_ids(curation.get("manipulation_id"))
        self._deserialize_manipulations(curation.get("manipulations") or {})

        self.status["loaded"] = True

    def prepare_save(self, fields_to_save=None):

        fields_to_save = fields_to_save or LoadConfig.fields_from_resource(
            NATIVE_ASSIGNMENTS_CONFIG,
            enabled_only=False,
        )

        save_data = {
            "assignments": {
                "ids": self.ids,
            },
            "stats": self.stats,
            "curation": {
                "review_status": self.review_status,
                "manipulation_id": self._serialize_manipulation_ids(),
                "manipulations": self._serialize_manipulations(),
            },
        }

        return deepcopy(
            {
                "data": save_data,
                "fields": fields_to_save,
                "attributes": {
                    "object_type": "AssignmentsData",
                    "format_version": 1,
                },
            }
        )

    def save(self, path, fields_to_save=None, *, mat_version="7.3"):
        prepared = self.prepare_save(fields_to_save)
        save_file(
            path,
            prepared["data"],
            prepared["fields"],
            mat_version=mat_version,
            root_attributes=prepared["attributes"],
            root="/",
        )

    @staticmethod
    def _serialize_component_refs(
        refs: list[dict],
    ) -> np.ndarray:
        """
        Columns:
            session_id, footprint_id, neuron_id
        """
        if not refs:
            return np.empty((0, 3), dtype=np.int64)

        return np.asarray(
            [
                (
                    int(ref["session_id"]),
                    int(ref["footprint_id"]),
                    int(ref["neuron_id"]),
                )
                for ref in refs
            ],
            dtype=np.int64,
        )

    @staticmethod
    def _deserialize_component_refs(
        values,
    ) -> list[dict]:

        values = np.asarray(values, dtype=np.int64)

        if values.size == 0:
            return []

        values = np.atleast_2d(values)

        if values.shape[1] != 3:
            raise ValueError(
                "Invalid manipulation component " f"reference shape: {values.shape}"
            )

        return [
            {
                "session_id": int(session_id),
                "footprint_id": int(footprint_id),
                "neuron_id": int(neuron_id),
            }
            for (session_id, footprint_id, neuron_id) in values
        ]

    def _serialize_manipulations(self) -> dict[str, dict]:

        serialized = {}

        for manipulation_id, record in self.manipulations.items():
            serialized[str(int(manipulation_id))] = {
                "type": str(record["type"]),
                "origin": (self._serialize_component_refs(record.get("origin", []))),
                "sources": (self._serialize_component_refs(record.get("sources", []))),
                "results": (self._serialize_component_refs(record.get("results", []))),
            }

        return serialized

    def _deserialize_manipulations(self, data) -> None:

        self.manipulations = {}

        if not data:
            self.next_manipulation_id = 0
            return

        for key, record in data.items():

            manipulation_id = int(key)

            self.manipulations[manipulation_id] = {
                "type": str(record["type"]),
                "origin": (
                    self._deserialize_component_refs(
                        record.get("origin", np.empty((0, 3)))
                    )
                ),
                "sources": (
                    self._deserialize_component_refs(
                        record.get("sources", np.empty((0, 3)))
                    )
                ),
                "results": (
                    self._deserialize_component_refs(
                        record.get("results", np.empty((0, 3)))
                    )
                ),
            }

        self.next_manipulation_id = (
            max(self.manipulations) + 1 if self.manipulations else 0
        )

    def _serialize_manipulation_ids(self) -> np.ndarray:

        if not self.manipulation_id:
            return np.empty((0, 2), dtype=np.int64)

        return np.asarray(
            sorted(
                (int(neuron_id), int(manipulation_id))
                for neuron_id, manipulation_id in self.manipulation_id.items()
            ),
            dtype=np.int64,
        )

    def _deserialize_manipulation_ids(self, values) -> None:

        if values is None:
            self.manipulation_id = {}
            return

        values = np.asarray(values, dtype=np.int64)

        if values.size == 0:
            self.manipulation_id = {}
            return

        values = np.atleast_2d(values)

        if values.shape[1] != 2:
            raise ValueError(
                "Invalid manipulation_id data: "
                f"expected shape (N, 2), got "
                f"{values.shape}."
            )

        self.manipulation_id = {
            int(neuron_id): int(manipulation_id)
            for neuron_id, manipulation_id in values
        }


def move_single_row(a, old_index, new_index):
    """
    could be changed to using "np.take" instead of slicing
    """
    if old_index < new_index:
        a[:, old_index:new_index], a[:, new_index] = (
            a[:, old_index + 1 : new_index + 1],
            a[:, old_index].copy(),
        )
    elif new_index < old_index:
        a[:, new_index + 1 : old_index + 1], a[:, new_index] = (
            a[:, new_index:old_index],
            a[:, old_index].copy(),
        )

    return a
