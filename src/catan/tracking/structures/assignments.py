from pathlib import Path
from typing import Any, Dict, Optional, Literal

from catan.core.io import load_file, save_file, NATIVE_ASSIGNMENTS_CONFIG
from catan.core.structures.load_config import LoadConfig, FieldSpec

from catan.core.utils import pad_axis
from catan.core.structures.session import SessionData

import numpy as np
from scipy import sparse

class Assignments:
    """
    Class to store and manage neuron assignments across sessions.
    """

    HDF5_VERSION = "1.0"

    union: Optional[SessionData] = None

    def __init__(self):

        self.reset()


    def reset(self):
        """
        Reset the assignments to an empty state.
        """
        self.ids: np.ndarray = np.zeros((0, 0), int)  # nNeurons x nSessions
        self.stats: Dict[str, np.ndarray] = {
            "p_matched": np.zeros((0, 0, 2), float),
            "shifts": np.zeros((0, 0, 2), float),  # nNeurons x nSessions x 2 (x,y)
        }

        self.union = SessionData(name="union")

    def pad_empty(self, n_neurons: int, n_sessions: int):
        """
        Pad the assignments and stats arrays to accommodate new neurons and sessions.
        """
        self.ids = pad_axis(self.ids, (n_neurons, n_sessions), -1)

        self.stats["p_matched"] = pad_axis(
            self.stats["p_matched"], (n_neurons, n_sessions, 0), np.nan
        )
        self.stats["shifts"] = pad_axis(
            self.stats["shifts"], (n_neurons, n_sessions, 0), np.nan
        )

        if self.union is None:
            return
        
        self.union.idx_eval = pad_axis(self.union.idx_eval, (n_neurons,), True)

    def move_session(self, session_id: int, new_session_id: int):
        """
        need to move code over
        """

        if new_session_id >= 0:
            # Move the session's data in assignments and tracking
            self.ids = move_single_row(
                self.ids, session_id, new_session_id
            )
            self.stats["p_matched"] = move_single_row(
                self.stats["p_matched"], session_id, new_session_id
            )
            self.stats["shifts"] = move_single_row(
                self.stats["shifts"], session_id, new_session_id
            )
        else:
            self.ids = np.delete(self.ids, session_id, axis=1)
            self.stats["p_matched"] = np.delete(
                self.stats["p_matched"], session_id, axis=1
            )
            self.stats["shifts"] = np.delete(
                self.stats["shifts"], session_id, axis=1
            )

        self.updating_neuron_presence()  # Update neuron presence and clean union data

        pass

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
        self.stats["p_matched"][:, session_id, :] = np.nan
        self.stats["shifts"][:, session_id, :] = np.nan

        self.updating_neuron_presence()  # Update neuron presence and clean union data

    def updating_neuron_presence(self):

        ## remove neurons that are no longer present in any session
        neuron_presence = (self.ids >= 0).sum(axis=1) > 0
        self.ids = self.ids[neuron_presence, :]
        self.stats["p_matched"] = self.stats["p_matched"][neuron_presence, :, :]
        self.stats["shifts"] = self.stats["shifts"][neuron_presence, :, :]

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
        self.union.register_spatial(footprints=footprints_cleaned)

        # print(f"Updated union data now contains {self.union.n_neurons} neurons.")

    @staticmethod
    def _from_file(
        path: str | Path,
        fields_to_load: dict[str, dict[str, FieldSpec]] | None = None,
    ) -> "Assignments":
        data = load_file(path, fields_to_load, config_name=NATIVE_ASSIGNMENTS_CONFIG, root="/")
        return Assignments._from_dict(data)

    @staticmethod
    def _from_dict(data: dict) -> "Assignments":
        assignments = Assignments()
        assignments.register_data(**data)
        return assignments
    
    def register_data(self, **data):

        ids = data["assignments"].get("ids")
        assert ids is not None, "IDs must be provided in the data dictionary"
        assert isinstance(ids, np.ndarray), "IDs must be a numpy array"
        self.ids = ids

        stats = data["stats"]
        assert isinstance(stats, dict) and all(
            isinstance(v, np.ndarray) for v in stats.values()
        ), "Stats must be a dictionary of numpy arrays"
        # assert stats is not None, "Stats must be provided in the data dictionary"
        self.stats = stats

        if "union" in data:
            self.union = SessionData()
            self.union.register_data(**data["union"])

    def save(
        self,
        path: str | Path,
        fields_to_save: dict[str, dict[str, FieldSpec]] | None = None,
        *,
        mat_version: Literal["pre73", "7.3"] = "7.3",
    ) -> None:
        
        fields_to_save = fields_to_save or LoadConfig.fields_from_resource(
            NATIVE_ASSIGNMENTS_CONFIG,
            enabled_only=False,
        )

        save_file(
            path, 
            self, 
            fields_to_save, 
            mat_version=mat_version,
            root_attributes={"object_type": "AssignmentsData", "format_version": 1},
            root="/"
        )


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
