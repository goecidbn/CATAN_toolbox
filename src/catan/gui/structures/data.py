from typing import Dict, Optional, Tuple, List

import numpy as np
from scipy import sparse

# import importlib

from catan.core.structures import SessionData
# from catan.gui.background_tasks import TaskContext
from catan import Tracking

# from catan.gui.data.utils import move_index_along_axis

print("reloading data.py")
# importlib.reload(neuron_tracking)  # reload to pick up changes during development


class Neurons:

    n: int = 0
    centroids: np.ndarray
    union_footprints: Dict[int, sparse.csc_matrix]


class Data(Tracking):

    def __init__(self, state):

        super().__init__()
        self.state = state
        self.current_session: Optional[SessionData] = None

        self.state.current_session_changed.connect(self._on_current_session_changed)

    def _on_current_session_changed(self, session_id: int):
        if session_id < 0 or session_id >= len(self.sessions):
            raise ValueError(f"Invalid session_id {session_id}")

        self.current_session = self.sessions[session_id]

        self.change_trace_presence(session_id, True)

    def unregister_neurons(self, session_id: int):

        super().unregister_neurons(session_id)
        self.state.update_selected_components(None)

    def remove_session(self, session_id: int):
        super().remove_session(session_id)
        self.state.update_selected_components(None)

    def change_trace_presence(self, session_id: int, to_present: Optional[bool] = None):
        """
        should be realized by session structure directly
        """
        session = self.sessions[session_id]
        ## default to "toggle" if nothing provided
        if to_present is None:
            to_present = not session.traces_loaded

        if session.traces_loaded == to_present:
            return

        if to_present:
            session.load_data(["temporal"])
        else:
            session.clean_traces()
        self.state.data_changed.emit(("traces", -1))

    def move_session(self, old_index: int, new_index: int):
        """
        should be moved to neuron_tracking
        """
        print(f"Moving session from index {old_index} to {new_index}")
        session = self.sessions.pop(old_index)
        self.sessions.insert(new_index, session)

        self.state.update_selected_components(None)

        # Reorder all session-indexed arrays here.
        # if "centroids" in self.neurons:
        # self.neurons.centroids = move_index_along_axis(
        #     self.neurons.centroids, axis=1, old=old_index, new=new_index
        # )

    # def remove_session(self, session_id: int):
    #     if session_id < 0 or session_id >= len(self.sessions):
    #         raise ValueError(f"Invalid session_id {session_id}")

    #     self.sessions.pop(session_id)

    #     # Reorder all session-indexed arrays here.
    #     self.assignments = np.delete(self.assignments, session_id, axis=1)

    #     # rebuild_results = self.rebuild_neurons()
    #     # if rebuild_results is not None:
    #     # self.neurons = rebuild_results
    #     # if "centroids" in self.neurons:
    #     #     self.neurons["centroids"] = np.delete(self.neurons["centroids"], session_id, axis=1)
    #     # self.session_metadata = remove_from_list(...)
    #     # invalidate statistic caches

    # def rebuild_neuron(self, neuron_id: int, ctx: Optional[TaskContext] = None):

    #     if not self.sessions:
    #         return

    #     session_ids = np.where(self.state.assignments[neuron_id, :] >= 0)[0]
    #     footprint_ids = self.state.assignments[neuron_id, session_ids]

    #     # build new centroids for neuron
    #     for session_id, fp_id in zip(session_ids, footprint_ids):

    #         # centroid = self.sessions[session_id].centroids[fp_id, :]
    #         session = self.sessions[session_id]
    #         if not session or session.centroids is None:
    #             raise ValueError(
    #                 f"Session {session_id} not found in data.sessions or centroids are missing"
    #             )

    #         self.neurons.centroids[fp_id, session_id, :] = session.centroids[fp_id, :]

    #     ## build new union footprint for neuron
    #     footprints = sparse.hstack(
    #         [
    #             (
    #                 self.sessions[s].A[:, fp_id]
    #                 if fp_id >= 0
    #                 else sparse.csc_matrix((np.prod(self.sessions[s].dims), 1))
    #             )
    #             for s, fp_id in zip(session_ids, footprint_ids)
    #         ],
    #         format="csc",
    #     )
    #     union_footprint = footprints.mean(axis=1)
    #     union_footprint[union_footprint < 0.001] = 0  # threshold small values to zero
    #     union_footprint = sparse.csc_matrix(union_footprint)
    #     self.neurons.union_footprints[neuron_id] = union_footprint

    # def rebuild_neurons(self, ctx: Optional[TaskContext] = None) -> Neurons | None:
    #     assignments = self.state.assignments
    #     if ctx is not None:
    #         ctx.message("Rebuilding neuron data...")
    #         ctx.progress(0)
    #     fields = ["centroids"]
    #     quality_fields = ["SNR_comp", "r_values", "cnn_preds"]

    #     neurons = Neurons()
    #     neurons.n = assignments.shape[0]
    #     if not self.sessions or assignments is None:
    #         print("No data or assignments available to build neurons.")
    #         return None

    #     centroids = np.full(assignments.shape + (2,), np.nan, dtype=np.float32)
    #     for session_id, footprints in enumerate(assignments.T):

    #         if (
    #             self.sessions[session_id] is None
    #             or self.sessions[session_id].centroids is None
    #         ):
    #             continue
    #         fp_idx = np.where(footprints >= 0)[0]
    #         fp_ids = assignments[fp_idx, session_id]
    #         centroids[fp_idx, session_id, :] = self.sessions[session_id].centroids[
    #             fp_ids, :
    #         ]
    #     neurons.centroids = centroids

    #     neurons.union_footprints = {}
    #     for neuron, footprint_ids in enumerate(assignments):
    #         if ctx is not None and ctx.cancel_check():
    #             ctx.message("Cancelled by user.")
    #             return None

    #         footprints = sparse.hstack(
    #             [
    #                 (
    #                     self.sessions[s].A[:, f]
    #                     if f >= 0
    #                     else sparse.csc_matrix((np.prod(self.sessions[s].dims), 1))
    #                 )
    #                 for s, f in enumerate(footprint_ids)
    #             ],
    #             format="csc",
    #         )
    #         union_footprint = footprints.mean(axis=1)
    #         union_footprint[union_footprint < 0.001] = (
    #             0  # threshold small values to zero
    #         )
    #         union_footprint = sparse.csc_matrix(union_footprint)
    #         neurons.union_footprints[neuron] = union_footprint

    #         if ctx is not None:
    #             ctx.progress(int((neuron + 1) / len(assignments) * 100))

    #     return neurons
