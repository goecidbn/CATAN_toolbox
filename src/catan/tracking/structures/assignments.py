from pathlib import Path
from typing import Any, Dict, Optional
from catan.core.io import load_hdf5, write_optional_array
from catan.core.utils import pad_axis
from catan.core.structures.load_config import LoadConfig
from catan.core.structures.session import SessionData
import h5py
import numpy as np
from scipy import sparse

class Assignments:
    """
    Class to store and manage neuron assignments across sessions.
    """

    HDF5_VERSION = "1.0"

    def __init__(self, params: Optional[dict] = None):

        self.params = params if params is not None else {}
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

        self.union = SessionData(name="union", params=self.params)

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


    def save(self, fname: str):

        ext = Path(fname).suffix
        if ext in [".h5", ".hdf5"]:
            with h5py.File(
                fname, "w"
            ) as f:
                self._save_to_hdf5(f)
        else:
            raise ValueError(f"Unsupported file extension: {ext}. Use '.h5' or '.hdf5'.")
        print(f"Saved neuron registration to {fname}")


    def _save_to_hdf5(self, h5ref: h5py.File | h5py.Group) -> None:
        h5ref.attrs["object_type"] = "AssignmentResults"
        h5ref.attrs["schema_version"] = self.HDF5_VERSION

        write_optional_array(h5ref, "IDs", self.ids, compression="gzip")

        stats_group = h5ref.create_group("stats")
        for key, value in self.stats.items():
            write_optional_array(stats_group, key, value, compression="gzip")

        union_group = h5ref.create_group(f"union")
        self.union.to_hdf5(union_group)


    @staticmethod
    def load(fname: str, params: dict) -> "Assignments":

        assignments = Assignments(params)

        ext = Path(fname).suffix
        if ext in [".h5", ".hdf5"]:
            with h5py.File(fname, "r") as f:
                data = assignments._from_hdf5(f)
        else:
            raise ValueError(f"Unsupported file extension: {ext}. Use '.h5' or '.hdf5'.")

        assignments.register_data(**data)

        return assignments

    def _from_hdf5(self, h5ref: h5py.File | h5py.Group, fields_to_load: Optional[dict] = None) -> dict[str, Any]:

        if h5ref.attrs.get("schema_version") != self.HDF5_VERSION:
            raise ValueError(
                f"Schema version mismatch: expected {self.HDF5_VERSION}, found {h5ref.attrs.get('schema_version')}"
            )

        if h5ref.attrs.get("object_type") != "AssignmentResults":
            raise ValueError(
                "The provided HDF5 group does not contain an AssignmentResults object."
            )

        fields_to_load = LoadConfig.fields_from_resource("catan_assignments.json")
        data = load_hdf5(h5ref, fields_to_load=fields_to_load)

        if "union" in h5ref:
            union_group = h5ref["union"]
            assert isinstance(
                union_group, h5py.Group
            ), "Union group is not a valid HDF5 group"

            fields_to_load = LoadConfig.fields_from_resource("catan_session.json")
            data["union"] = SessionData.from_hdf5(union_group, fields_to_load=fields_to_load)

        return data


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
