"""
function written by Alexander Schmidt, allowing for complete registration of neuron footprints across several sessions

last updated on September 7th, 2026
"""

from dataclasses import dataclass, field

import os
import numpy as np
from typing import Any, Dict, Optional, Tuple, List, Union, Literal
import sys, copy, logging, time, numbers, warnings
from scipy import sparse
from scipy.optimize import linear_sum_assignment

from pathlib import Path

from catan.core.io import NATIVE_SESSION_CONFIG, get_backend
from catan.core.data import center_of_mass
from catan.core.structures.load_config.config import LoadConfig, FieldSpec

from catan.core.structures import SessionData, NeuronComponent
from catan.core.structures.session_snapshot import (
    read_session_snapshot,
    write_session_snapshot,
)
from catan.core.analysis import calculate_statistics, calculate_p
from catan.core.alignment import _shift_sparse_bilinear

from .structures import Model, Assignments

logging.basicConfig(level=logging.INFO)


@dataclass(slots=True)
class ModelCountCalculation:

    counts: np.ndarray

    # These originate from the same expensive
    # calculate_statistics() call.
    idx_remove: np.ndarray
    remove_info: Any


@dataclass(slots=True)
class SessionProcessingState:

    geometry_revision: int = 0

    # The session currently has geometry/remapping,
    # but it depends on upstream geometry that has
    # subsequently changed.
    alignment_stale: bool = False


class Tracking:

    _model: Dict[str, Model] = {}
    _current_model: Optional[str] = None

    _assignments: Dict[str, "Assignments"] = {}
    _current_assignments: Optional[str] = None

    HDF5_VERSION = 1

    def __init__(
        self,
        neighbor_distance=25.0,
        bins=64,
        n_threads=1,
        use_kde=False,
        pxtomu=1.0,
        L=512,
        logLevel=logging.ERROR,
    ):
        """
        Central class for neuron tracking via session registration, model building and neuron registration

        Parameters
        ----------

        neighbor_distance : float = 25.
            the distance (in mu m) up to where neuron similarities and distances are calculated and registered for model building (shouldn't be much smaller, as the distance model requires at least some further ranging data)

        bins : int = 64
            number of bins used for model building and count registration. Should be a number 2^n to allow scaling down.

        n_threads: int = 1
            the number of threads to use, when calculating footprint similarity - the costly part of the analysis

        use_kde: bool = False
            specifies, whether neurons from areas of lowest and highest neuron density are excluded when building the probabilistic model. Costly, but can avoid some weird behavior

        pxtomu: float = 1.
            the factor to transform pixels to micrometer (mu m) distance

        L: int = 512
            number of pixels along one dimension (actually - this should rather be the window length in mu m)
        """
        self.log = logging.getLogger("matchinglogger")
        self.log.setLevel(logLevel)

        self.params = {}
        self.params["neighbor_distance"] = neighbor_distance
        self.params["bins"] = bins
        self.params["n_threads"] = n_threads
        self.params["use_kde"] = use_kde
        self.params["pxtomu"] = pxtomu
        self.params["L"] = L

        self._update_bins(bins)

        self._session_processing = {}
        self._stale_assignments = {}

        self.reset_data()

        self.add_model("local")
        self.add_assignments("local")

    ### ============================================ ###
    ### ============== MODEL FUNCTIONS ============= ###
    ### ============================================ ###

    @property
    def model(self) -> Optional[Model]:
        if self._current_model is None:
            return None
        return self._model[self._current_model]

    @property
    def current_model_name(self):
        return self._current_model

    def change_model(self, name: str):
        if name not in self._model:
            raise ValueError(f"Model '{name}' does not exist.")
        self._current_model = name

    def add_model(self, name: str, model: Optional[str | Model] = None):

        if model is None:
            model = Model(params=self.params)
        elif isinstance(model, str):
            model = Model._from_file(path=model, params=self.params)

        if not isinstance(model, Model):
            raise ValueError(
                "model must be an instance of Model class or a path to a saved model."
            )

        self._model[name] = model
        self._current_model = name

    @property
    def available_models(self) -> List[str]:
        return list(self._model.keys())

    ### ============================================ ###
    ### ============ ASSIGNMENT FUNCTIONS ========== ###
    ### ============================================ ###

    @property
    def assignments(self) -> Optional[Assignments]:
        if self._current_assignments is None:
            return None
        return self._assignments[self._current_assignments]

    @property
    def current_assignments(self):
        return self._current_assignments

    def change_assignments(self, name: str):
        if name not in self._assignments:
            raise ValueError(f"Assignments '{name}' does not exist.")
        self._current_assignments = name

        self.update_sessions_with_assignments()

    def add_assignments(
        self, name: str, assignments: Optional[str | Assignments] = None
    ):
        copied = False
        if assignments is None:
            assignments = Assignments()
        elif isinstance(assignments, str):
            if assignments in self._assignments:
                assignments = self._assignments[assignments].copy()
                copied = True
            else:
                assignments = Assignments._from_file(path=assignments)

        if not isinstance(assignments, Assignments):
            raise ValueError(
                "assignments must be an instance of Assignments class or a path to a saved assignments."
            )

        ok = self.check_assignments_compatibility(assignments)
        if not ok:
            raise ValueError(
                "The provided assignments are not compatible with the current model and sessions."
            )
        self._assignments[name] = assignments
        self._current_assignments = name

        self._stale_assignments.setdefault(name, set())
        self._ensure_assignment_session_count(assignments)

        if copied:
            self.rebuild_union()
        self.update_sessions_with_assignments()

    def remove_assignments(self, name: str):
        if name == "local":
            raise ValueError("Cannot remove the 'local' assignments.")
        if name not in self._assignments:
            raise ValueError(f"Assignments '{name}' does not exist.")

        del self._assignments[name]
        self._stale_assignments.pop(name, None)

        if self._current_assignments == name:
            self.change_assignments("local")

    @property
    def available_assignments(self) -> List[str]:
        return list(self._assignments.keys())

    def update_sessions_with_assignments(self):

        if self.assignments is None:
            return

        # for session in self.sessions:
        # if session.id >= (n_status := len(self.assignments.matched_status)):
        #     self.assignments.matched_status.extend(
        #         [False] * (session.id - n_status + 1)
        #     )
        # print(
        #     "Warning: trying to update matched_status with invalid session_id:", session.id
        # )
        # continue
        # self.assignments.matched_status[session.id] = False

        for session, assignment_ids in zip(self.sessions, self.assignments.ids.T):

            # if np.any(assignment_ids >= 0):
            #     # mark session as matched, if it has assignments
            #     self.assignments.matched_status[session.id] = True

            update_included = False
            if session.included is not None:
                idx_assigned_from_session = np.where(session.included)[0]
                idx_assigned_from_assignments = assignment_ids[assignment_ids >= 0]

                assignment_in_session = np.isin(
                    idx_assigned_from_assignments, idx_assigned_from_session
                )
                if not assignment_in_session.all():
                    # warnings.warn(f"Session {session_id} has neurons in 'assignments' that are not marked as valid in the session data (included).")
                    # warnings.warn(f"Neurons in assignments but not in session included: {idx_assigned_from_assignments[~assignment_in_session]}")
                    update_included = True

                session_in_assignment = np.isin(
                    idx_assigned_from_session, idx_assigned_from_assignments
                )
                if not session_in_assignment.all():
                    # warnings.warn(f"Session {session_id} has neurons marked as valid in the session data (included) that are not present in 'assignments'.")
                    # warnings.warn(f"Neurons in session included but not in assignments: {idx_assigned_from_session[~session_in_assignment]}")
                    update_included = True
            else:
                # warnings.warn(f"Session {session_id} does not have 'included' defined. It will be updated based on 'assignments'.")
                update_included = True

            if update_included:
                session.included = np.zeros(session.n_neurons, dtype=bool)
                session.included[assignment_ids[assignment_ids >= 0]] = True

    ### ============================================= ###
    ### ============= DEFINE DATA LOADING =========== ###
    ### ============================================= ###

    def get_session(
        self,
        from_file: Optional[str | Path] = None,
        from_data: Optional[SessionData] = None,
        from_session_index: Optional[int] = None,
        fields_to_load: Optional[dict] = None,
        align_to_reference=True,
    ) -> SessionData:
        """
        Returns a SessionData object based on the provided input. The session can be loaded
        * `from_file`: a path to the session file stored on disk. Requires additional `fields_to_load` argument to specify which data to load. If `align_to_reference` is True and a reference session exists, the session will be aligned to the reference session's alignment template.
        * `from_data`: an existing SessionData object
        * `from_session_index`: an index to retrieve the session from the list of registered sessions


        """

        if from_file is not None:
            assert isinstance(
                from_file, (str, Path)
            ), "from_file must be a string or Path"

            this_data = SessionData._from_file(
                str(from_file),
                fields_to_load,
                self.alignment_references if align_to_reference else None,
            )
        elif from_data is not None:
            assert isinstance(
                from_data, SessionData
            ), "from_data must be a SessionData instance"
            this_data = from_data
        elif from_session_index is not None:
            assert isinstance(
                from_session_index, int
            ), "from_session_index must be an integer"
            assert (
                0 <= from_session_index < len(self.sessions)
            ), "from_session_index is out of range"
            this_data = self.sessions[from_session_index]
        else:
            raise ValueError(
                "Either from_file, from_data, or from_session_index must be provided."
            )
        return this_data

    def reset_data(self):
        self.sessions: List[SessionData] = []

    def _processing_state(self, session_id: int) -> SessionProcessingState:

        session = self.sessions[session_id]

        state = self._session_processing.get(session)
        if state is None:
            state = SessionProcessingState()

            self._session_processing[session] = state

        return state

    def geometry_revision(self, session_id: int) -> int:
        return self._processing_state(session_id).geometry_revision

    def alignment_is_stale(self, session_id: int) -> bool:
        return self._processing_state(session_id).alignment_stale

    def require_current_alignment(self, session_id: int):
        session = self.sessions[session_id]

        if not session.status["spatial_loaded"] or session.footprints is None:
            raise ValueError(f"Session {session_id}: spatial data must be loaded.")

        if not session.status["aligned"]:
            raise ValueError(f"Session {session_id}: alignment has not succeeded.")

        if self.alignment_is_stale(session_id):
            raise ValueError(
                f"Session {session_id}: alignment is outdated; "
                "realign this session before further processing."
            )

    def mark_geometry_changed(self, session_id: int):
        """
        Mark dependencies invalid after the aligned geometry of one session has changed.

        The changed session itself is assumed to have a valid newly committed alignment.

        All later sessions have stale alignment because their remapping may depend on this session.
        """

        if not (0 <= session_id < len(self.sessions)):
            raise IndexError(f"Invalid session_id {session_id}.")

        changed = self.sessions[session_id]

        changed_state = self._processing_state(session_id)

        changed_state.geometry_revision += 1
        changed_state.alignment_stale = False

        # ============================================
        # Downstream remappings
        # ============================================
        for downstream_id in range(session_id + 1, len(self.sessions)):
            self._processing_state(downstream_id).alignment_stale = True

        # ============================================
        # Model evidence
        # ============================================
        affected_sessions = self.sessions[session_id:]
        affected_paths = {
            str(session.path)
            for session in affected_sessions
            if session.path is not None
        }

        for model in self._model.values():
            model.invalidate_counts_for_paths(affected_paths)

        # ============================================
        # Assignments
        # ============================================
        for name in self._assignments:

            self._stale_assignments.setdefault(name, set()).update(affected_paths)

        # Compatibility flags for existing GUI code.
        for session in affected_sessions:
            session.status["registered_to_model"] = False

        return affected_paths

    def mark_alignment_current(self, session_id: int):

        state = self._processing_state(session_id)

        state.alignment_stale = False

    def assignment_is_stale(
        self, session_id: int, *, assignment_name: str | None = None
    ) -> bool:

        if assignment_name is None:
            assignment_name = self._current_assignments

        if assignment_name is None:
            return False

        session = self.sessions[session_id]

        if session.path is None:
            return False

        return str(session.path) in self._stale_assignments.get(assignment_name, set())

    def mark_assignment_current(
        self, session_id: int, *, assignment_name: str | None = None
    ):

        if assignment_name is None:
            assignment_name = self._current_assignments

        if assignment_name is None:
            return

        session = self.sessions[session_id]

        if session.path is None:
            return

        self._stale_assignments.setdefault(assignment_name, set()).discard(
            str(session.path)
        )

    def register_session(
        self,
        from_file: Optional[str | Path] = None,
        from_data: Optional[SessionData] = None,
        fields_to_load: Optional[dict] = None,
        name: Optional[str] = None,
        align=True,
        **kwargs,
    ) -> int:
        """
        Register a new session from a file or loaded data, load the data and add it to the list of sessions. The session is aligned to the previous sessions if align=True.

        Input

        - from_file: str

            Path to the session file

        - name: Optional[str]

            Optional name for this session

        - fields_to_load: list[str] = ["spatial", "traces", "quality"]

            Specifies which data should be loaded and can be either combination of the three above - but setting all is strongly encouraged.

            spatial: loads footprint and background data - necessary to do any kind of further processing
            traces: loads temporal traces - not entirely necessary, but is used to test for overlapping neurons with highly correlated activity
            quality: loads quality parameters (SNR_comp, r_values, cnn_preds for CaImAn) which are used for thresholding

        - align: bool = True

            Flag for aligning the spatial components to prior registered sessions (using rigid and non-rigid correction). Highly encouraged to leave this enabled, unless sessions are already aligned. For further details see demo notebook `alignment.ipynb`

        """

        if from_file is not None:
            this_data = self.get_session(
                from_file=from_file,
                fields_to_load=fields_to_load,
                align_to_reference=align,
            )

            this_data.path = str(from_file)
            this_data.name = name if name is not None else Path(from_file).parent.name

        elif from_data is not None:
            assert isinstance(
                from_data, SessionData
            ), "from_data must be a SessionData instance"
            this_data = from_data
            if name is None:
                if this_data.name is None:
                    if this_data.path is None:
                        this_data.name = f"Session{len(self.sessions):03d}"
                    else:
                        this_data.name = Path(this_data.path).parent.name
            else:
                this_data.name = name
        else:
            raise ValueError("Either from_file or from_data must be provided.")
        paths = [session.path for session in self.sessions]
        if this_data.path in paths:
            raise ValueError(f"Session {this_data.path} is already registered.")

        this_data.id = len(self.sessions)

        self.sessions.append(this_data)
        self._ensure_assignment_session_count()

        return this_data.id

    def alignment_references_for_session(self, session_id: int | None = None):

        if session_id is None:
            candidate_sessions = self.sessions

        else:
            if not (0 <= session_id <= len(self.sessions)):
                raise IndexError(f"Invalid session_id {session_id}.")

            # Alignment of a session is defined
            # relative to earlier sessions only.
            candidate_sessions = self.sessions[:session_id]

        aligned_sessions = [
            session
            for session in candidate_sessions
            if (
                session.status["aligned"]
                and session.path is not None
                and session.background_template is not None
            )
        ]

        if not aligned_sessions:
            return None

        alignment_window = min(len(aligned_sessions), 10)

        references = {}

        for session in aligned_sessions[-alignment_window:]:

            assert session.path is not None

            matrix = (
                np.eye(3, dtype=float)
                if session.remap is None
                else session.remap.matrix
            )

            references[str(session.path)] = {
                "template": (session.background_template.copy()),
                "matrix": (np.asarray(matrix, dtype=float).copy()),
            }

        return references

    @property
    def alignment_references(self):

        return self.alignment_references_for_session()

    ### ============================================ ###
    ### ============= COUNT REGISTRATION =========== ###
    ### ============================================ ###

    def update_counts_with_data(
        self,
        from_file: Optional[str | Path] = None,
        from_data: Optional[SessionData] = None,
        from_session_index: Optional[int] = None,
        align_to_reference=True,
        *,
        session_distances=(1,),
        apply_removals=True,
    ):

        this_data = self.get_session(
            from_file=from_file,
            from_data=from_data,
            from_session_index=(from_session_index),
            align_to_reference=(align_to_reference),
        )

        session_id = this_data.id

        if not (0 <= session_id < len(self.sessions)):
            raise ValueError(
                "Model-count registration " "requires a registered session."
            )

        return self.update_model_counts(
            session_id,
            session_distances=(session_distances),
            apply_removals=(apply_removals),
        )

    def calculate_model_counts(
        self,
        this_data: SessionData,
        *,
        ref_data: SessionData | None = None,
        mode: Literal["same", "cross"] = "cross",
    ) -> ModelCountCalculation:
        """
        Calculate one model-count contribution.

        This method does not modify SessionData and does not modify Model.

        TODO:
            * might just change everything to require "total counts" only, removing NN-calculation
            * change how correlation is calculated: just apply centroid distance shift! (test performance/timing before that)

        """

        if mode == "same":
            ref_data = this_data

        elif mode == "cross":
            if ref_data is None:
                raise ValueError("ref_data must be provided for cross-session counts.")

        else:
            raise ValueError("mode must be 'same' or 'cross'.")

        assert isinstance(ref_data, SessionData)

        (
            footprint_shifts,
            footprint_distances,
            footprint_correlations,
            idx_remove,
            remove_info,
        ) = calculate_statistics(
            this_data,
            ref_data,
            distance_threshold=self.params.get("neighbor_distance", 25.0),
            nP=12,
            params=self.params,
        )

        idx_remove = np.asarray(idx_remove, dtype=int).reshape(-1)

        # ============================================
        # Local inclusion masks
        # ============================================
        #
        # For same-session calculations, removal candidates do not
        # participate in the NN mask used below.

        idx_this = np.asarray(this_data.included, dtype=bool).copy()
        idx_ref = np.asarray(ref_data.included, dtype=bool).copy()

        if mode == "same" and idx_remove.size:
            idx_this[idx_remove] = False

            # ref_data is this_data in same mode.
            idx_ref = idx_this

        # ============================================
        # Neighbours / nearest neighbours
        # ============================================

        neighbors = footprint_distances < self.params.get("neighbor_distance", 15.0)

        is_NN = np.zeros((ref_data.n_neurons, this_data.n_neurons), dtype=bool)

        if mode == "cross":
            min_distance = np.nanmin(footprint_distances, axis=1)

            idx_finite = ~np.isnan(min_distance)

            min_distance_idx = np.nanargmin(
                footprint_distances[idx_ref & idx_finite, :], axis=1
            )

            is_NN[idx_ref & idx_finite, min_distance_idx] = True

        else:
            is_NN[idx_this, idx_this] = True

        # ============================================
        # Histograms
        # ============================================

        histo_options = {
            "bins": self.params["nbins"],
            "range": [
                self.params["arrays"]["distance_bounds"][[0, -1]],
                self.params["arrays"]["correlation_bounds"][[0, -1]],
            ],
        }

        if mode == "same":

            idxes = neighbors & ~is_NN
            counts = np.histogram2d(
                footprint_distances[idxes],
                footprint_correlations[idxes],
                **histo_options,
            )[0].astype(int)

        else:

            counts = np.zeros(
                (self.params["nbins"], self.params["nbins"], 3), dtype=int
            )

            idxes = neighbors
            counts[..., 0] = np.histogram2d(
                footprint_distances[idxes],
                footprint_correlations[idxes],
                **histo_options,
            )[0].astype(int)

            idxes = neighbors & is_NN
            counts[..., 1] = np.histogram2d(
                footprint_distances[idxes],
                footprint_correlations[idxes],
                **histo_options,
            )[0].astype(int)

            idxes = neighbors & ~is_NN
            counts[..., 2] = np.histogram2d(
                footprint_distances[idxes],
                footprint_correlations[idxes],
                **histo_options,
            )[0].astype(int)

        return ModelCountCalculation(
            counts=counts, idx_remove=idx_remove, remove_info=remove_info
        )

    def update_model_pair_counts(self, reference_session_id: int, session_id: int):

        if self.model is None:
            raise ValueError("No current model available.")

        if reference_session_id == session_id:
            raise ValueError(
                "Cross-session count calculation requires two different sessions."
            )

        reference = self.sessions[reference_session_id]

        session = self.sessions[session_id]

        self.require_current_alignment(reference_session_id)
        self.require_current_alignment(session_id)

        if reference.path is None or session.path is None:
            raise ValueError(
                "Both sessions require paths for model count registration."
            )

        result = self.calculate_model_counts(session, ref_data=reference, mode="cross")

        self.model.set_cross_counts(
            str(reference.path),
            str(session.path),
            result.counts,
            source_revisions=(
                self.geometry_revision(reference_session_id),
                self.geometry_revision(session_id),
            ),
        )

        return result

    def update_model_same_counts(self, session_id: int, *, apply_removals=True):

        if self.model is None:
            raise ValueError("No current model available.")

        session = self.sessions[session_id]

        self.require_current_alignment(session_id)

        if session.path is None:
            raise ValueError(f"Session {session_id} has no path.")

        result = self.calculate_model_counts(session, mode="same")

        if apply_removals and result.idx_remove.size:
            session.included[result.idx_remove] = False

        self.model.set_same_counts(
            str(session.path),
            result.counts,
            source_revision=(self.geometry_revision(session_id)),
        )

        return result

    def update_model_counts(
        self, session_id: int, *, session_distances=(1,), apply_removals: bool = True
    ):

        if self.model is None:
            raise ValueError("No current model available.")

        if not (0 <= session_id < len(self.sessions)):
            raise IndexError(f"Invalid session_id {session_id}.")

        session = self.sessions[session_id]

        self.require_current_alignment(session_id)

        distances = self._normalize_session_distances(session_id, session_distances)

        results = {"same": None, "cross": {}}

        # ============================================
        # Same-session evidence
        # ============================================
        results["same"] = self.update_model_same_counts(
            session_id, apply_removals=apply_removals
        )

        # ============================================
        # Cross-session evidence
        # ============================================
        for distance in distances:

            reference_session_id = session_id - distance

            if reference_session_id < 0:
                continue

            reference = self.sessions[reference_session_id]

            # The pair is structurally requested, but
            # cannot currently be calculated.
            if not reference.status["aligned"]:
                continue

            result = self.update_model_pair_counts(reference_session_id, session_id)

            results["cross"][(reference_session_id, session_id)] = result

        # Temporary compatibility flag.
        # Later this will be derived from count-record
        # completeness / staleness instead.
        session.status["registered_to_model"] = True

        return results

    @staticmethod
    def _normalize_session_distances(
        session_id: int, session_distances
    ) -> tuple[int, ...]:

        if session_distances is None:
            return tuple(range(1, session_id + 1))

        distances = tuple(sorted({int(distance) for distance in session_distances}))

        if any(distance <= 0 for distance in distances):
            raise ValueError("session_distances must contain positive integers.")

        return distances

    def model_count_state(self, session_id: int, *, session_distances=(1,)):
        """Inspect required count records without calculating or modifying them."""
        if self.model is None:
            raise ValueError("No current model available.")

        if not (0 <= session_id < len(self.sessions)):
            raise IndexError(f"Invalid session_id {session_id}.")

        distances = self._normalize_session_distances(session_id, session_distances)

        def session_path(index):
            path = self.sessions[index].path
            if path is None:
                raise ValueError(f"Session {index} has no path.")
            return str(path)

        def record_state(kind, key, source_ids):
            if key not in self.model.counts[kind]:
                return "missing"

            if key in self.model.stale_counts[kind]:
                return "stale"

            if any(self.alignment_is_stale(index) for index in source_ids):
                return "stale"

            revisions = tuple(self.geometry_revision(index) for index in source_ids)
            expected = revisions[0] if kind == "same" else revisions
            recorded = self.model.count_revisions[kind].get(key)

            if recorded != expected:
                return "stale"

            return "current"

        path = session_path(session_id)

        cross = {}
        for distance in distances:
            reference_id = session_id - distance
            if reference_id < 0:
                continue

            key = (session_path(reference_id), path)
            cross[reference_id] = record_state("cross", key, (reference_id, session_id))

        return {
            "same": record_state("same", path, (session_id,)),
            "cross": cross,
        }

    def plan_model_update(
        self,
        *,
        session_ids=None,
        from_session_id=None,
        mode="pending",
        session_distances=(1,),
    ):
        if self.model is None:
            raise ValueError("No current model available.")

        if self.model.loaded:
            raise ValueError(
                "The current model was loaded from file and cannot be updated."
            )

        accepted = {
            "pending": {"missing", "stale"},
            "missing": {"missing"},
            "stale": {"stale"},
            "force": {"missing", "stale", "current"},
        }
        if mode not in accepted:
            raise ValueError(f"Unknown processing mode {mode!r}.")

        if session_ids is not None and from_session_id is not None:
            raise ValueError("Specify session_ids or from_session_id, not both.")

        n_sessions = len(self.sessions)

        if from_session_id is not None:
            if not 0 <= from_session_id < n_sessions:
                raise IndexError(f"Invalid session_id {from_session_id}.")
            selected = set(range(from_session_id, n_sessions))
        else:
            selected = set(range(n_sessions) if session_ids is None else session_ids)

        if any(index < 0 or index >= n_sessions for index in selected):
            raise IndexError("Selection contains an invalid session_id.")

        distances = (
            None
            if session_distances is None
            else self._normalize_session_distances(0, session_distances)
        )

        same_sessions = []
        cross_pairs = []

        for target_id in range(n_sessions):
            reference_ids = [
                target_id - distance
                for distance in self._normalize_session_distances(target_id, distances)
                if target_id - distance >= 0
            ]

            relevant_references = [
                index
                for index in reference_ids
                if index in selected or target_id in selected
            ]

            if target_id not in selected and not relevant_references:
                continue

            state = self.model_count_state(target_id, session_distances=distances)

            if target_id in selected and state["same"] in accepted[mode]:
                same_sessions.append(target_id)

            for reference_id in relevant_references:
                if state["cross"][reference_id] in accepted[mode]:
                    cross_pairs.append((reference_id, target_id))

        required = set(same_sessions)
        for reference_id, target_id in cross_pairs:
            required.update((reference_id, target_id))

        return {
            "session_ids": sorted(selected),
            "session_distances": distances,
            "same_sessions": same_sessions,
            "cross_pairs": cross_pairs,
            "load_sessions": [
                index
                for index in sorted(required)
                if not self.sessions[index].status["spatial_loaded"]
                or self.sessions[index].footprints is None
            ],
            "alignment_sessions": [
                index
                for index in sorted(required)
                if not self.sessions[index].status["aligned"]
                or self.alignment_is_stale(index)
            ],
            "check_fit": bool(selected)
            and (bool(cross_pairs) or not self.model.fitted or self.model.fit_stale),
        }

    def process_model_updates(
        self,
        *,
        session_ids=None,
        from_session_id=None,
        mode="pending",
        session_distances=(1,),
    ):
        plan = self.plan_model_update(
            session_ids=session_ids,
            from_session_id=from_session_id,
            mode=mode,
            session_distances=session_distances,
        )

        if plan["load_sessions"] or plan["alignment_sessions"]:
            raise ValueError(
                "Model update requires current spatial data and alignment. "
                f"Load sessions: {plan['load_sessions']}; "
                f"align sessions: {plan['alignment_sessions']}."
            )

        required = set(plan["same_sessions"])
        for reference_id, target_id in plan["cross_pairs"]:
            required.update((reference_id, target_id))

        # Validate every dependency before modifying any count records.
        for session_id in sorted(required):
            self.require_current_alignment(session_id)

        for session_id in plan["same_sessions"]:
            self.update_model_same_counts(
                session_id,
                apply_removals=False,
            )

        for reference_id, target_id in plan["cross_pairs"]:
            self.update_model_pair_counts(reference_id, target_id)

        # Refresh the existing GUI flag from actual record completeness.
        for session_id in sorted(required | set(plan["session_ids"])):
            state = self.model_count_state(
                session_id,
                session_distances=plan["session_distances"],
            )
            self.sessions[session_id].status["registered_to_model"] = state[
                "same"
            ] == "current" and all(
                value == "current" for value in state["cross"].values()
            )

        counts = self.model.aggregate_counts(
            session_order=[
                str(session.path)
                for session in self.sessions
                if session.path is not None
            ],
            session_distances=plan["session_distances"],
        )

        cross_count = int(counts["cross"][..., 0].sum())

        fitted = False
        if plan["check_fit"] and cross_count >= 20:
            self.fit_model(
                session_distances=plan["session_distances"],
            )
            fitted = True

        return {
            "plan": plan,
            "fitted": fitted,
            "cross_count": cross_count,
        }

    def fit_model(self, *, session_ids=None, session_distances=(1,), use_cdf=True):
        """
        Fit the current model from registered count records.

        Parameters
        ----------
        session_ids:
            Optional subset of currently registered sessions
            to use for model fitting.

        session_distances:
            Allowed differences in current session order for
            cross-session count records. ``None`` uses all
            currently order-compatible pairs.

        use_cdf:
            Passed to Model.fit_model_to_counts().
        """

        if self.model is None:
            raise ValueError("No model available to fit.")

        # Current session order is deliberately supplied here,
        # rather than stored inside Model.
        session_order = []
        for session in self.sessions:

            if session.path is None:
                continue

            session_order.append(str(session.path))

        selected_paths = None
        if session_ids is not None:
            selected_paths = []

            for session_id in session_ids:
                session = self.sessions[int(session_id)]

                if session.path is None:
                    raise ValueError(f"Session {session_id} " "has no path.")

                selected_paths.append(str(session.path))

        if session_distances is not None:
            session_distances = tuple(
                sorted({int(distance) for distance in session_distances})
            )

        counts = self.model.aggregate_counts(
            session_order=session_order,
            session_paths=selected_paths,
            session_distances=(session_distances),
        )

        self.model.fit_model_to_counts(counts["cross"], use_cdf=use_cdf)

        return counts

    ### ============================================ ###
    ### =========== ASSIGNMENT FUNCTIONS =========== ###
    ### ============================================ ###

    def session_assigned(self, session_id):
        if self.assignments is None:
            return False
        if session_id >= self.assignments.matched_status.shape[0]:
            return False
        return self.assignments.matched_status[session_id]

    def plan_assignment_update(
        self, *, session_ids=None, from_session_id=None, mode="pending"
    ):
        if self.assignments is None:
            raise ValueError("No current assignments available.")

        accepted = {
            "pending": {"missing", "stale"},
            "missing": {"missing"},
            "stale": {"stale"},
            "force": {"missing", "stale", "current"},
        }
        if mode not in accepted:
            raise ValueError(f"Unknown processing mode {mode!r}.")

        if session_ids is not None and from_session_id is not None:
            raise ValueError("Specify session_ids or from_session_id, not both.")

        n_sessions = len(self.sessions)

        if from_session_id is not None:
            if not 0 <= from_session_id < n_sessions:
                raise IndexError(f"Invalid session_id {from_session_id}.")
            selected = set(range(from_session_id, n_sessions))
        else:
            selected = set(range(n_sessions) if session_ids is None else session_ids)

        if any(index < 0 or index >= n_sessions for index in selected):
            raise IndexError("Selection contains an invalid session_id.")

        assigned = {
            index for index in range(n_sessions) if self.session_assigned(index)
        }

        def assignment_state(index):
            if index not in assigned:
                return "missing"
            if self.assignment_is_stale(index):
                return "stale"
            return "current"

        requested = {
            index for index in selected if assignment_state(index) in accepted[mode]
        }

        first = None
        preserved = []
        clear_sessions = []
        register_sessions = []

        if requested:
            first = min(requested)

            # A stale earlier assignment cannot provide a current prefix.
            stale_prefix = [
                index
                for index in assigned
                if index < first and self.assignment_is_stale(index)
            ]
            if stale_prefix:
                first = min(stale_prefix)

            preserved = sorted(index for index in assigned if index < first)
            clear_sessions = sorted(index for index in assigned if index >= first)

            # Include requested missing sessions and every existing
            # registration whose union dependency will change.
            register_sessions = sorted(requested | set(clear_sessions))

        required = sorted(set(preserved) | set(register_sessions))

        model_state = "not_required"
        if len(required) > 1:
            if self.model is None or not self.model.fitted:
                model_state = "missing"
            elif self.model.fit_stale:
                model_state = "stale"
            else:
                model_state = "current"

        return {
            "session_ids": sorted(selected),
            "from_session_id": first,
            "preserved_sessions": preserved,
            "clear_sessions": clear_sessions,
            "register_sessions": register_sessions,
            "load_sessions": [
                index
                for index in required
                if not self.sessions[index].status["spatial_loaded"]
                or self.sessions[index].footprints is None
            ],
            "alignment_sessions": [
                index
                for index in required
                if not self.sessions[index].status["aligned"]
                or self.alignment_is_stale(index)
            ],
            "model_state": model_state,
        }

    def build_assignment_update(
        self,
        *,
        session_ids=None,
        from_session_id=None,
        mode="pending",
    ):
        plan = self.plan_assignment_update(
            session_ids=session_ids,
            from_session_id=from_session_id,
            mode=mode,
        )

        if not plan["register_sessions"]:
            return None

        if plan["load_sessions"] or plan["alignment_sessions"]:
            raise ValueError(
                "Assignment update requires current spatial data and alignment. "
                f"Load sessions: {plan['load_sessions']}; "
                f"align sessions: {plan['alignment_sessions']}."
            )

        if plan["model_state"] in ("missing", "stale"):
            raise ValueError(
                "Update model counts and fit the model before "
                "reprocessing assignments."
            )

        required = plan["preserved_sessions"] + plan["register_sessions"]
        for session_id in required:
            self.require_current_alignment(session_id)

        source = self.assignments
        name = self.current_assignments
        candidate, neuron_id_map = source.copy_prefix(plan["from_session_id"])

        # Borrow session data without copying footprints or traces.
        # All assignment/union mutations belong to the candidate.
        worker = copy.copy(self)
        worker.sessions = list(self.sessions)
        worker._model = dict(self._model)
        worker._assignments = {name: candidate}
        worker._current_assignments = name
        worker._stale_assignments = {
            name: set(self._stale_assignments.get(name, set()))
        }

        Tracking.rebuild_union(worker)

        for session_id in plan["register_sessions"]:
            # Use the core implementation, bypassing GUI notifications.
            Tracking.assign_neurons(
                worker,
                from_session_index=session_id,
                align_to_reference=False,
                clean_traces=False,
            )

            if not worker.session_assigned(session_id):
                raise RuntimeError(f"Session {session_id} was not registered.")

        # Recover manipulation associations through stable footprint refs.
        # Later history entries supersede earlier entries for a neuron.
        for manipulation_id in sorted(candidate.manipulations):
            record = candidate.manipulations[manipulation_id]

            for ref in record.get("results", []):
                session_id = int(ref["session_id"])
                footprint_id = int(ref["footprint_id"])

                if not 0 <= session_id < candidate.ids.shape[1]:
                    raise ValueError(
                        f"Manipulation {manipulation_id} references "
                        f"invalid session {session_id}."
                    )
                if footprint_id < 0:
                    raise ValueError(
                        f"Manipulation {manipulation_id} has "
                        "an invalid footprint reference."
                    )

                rows = np.flatnonzero(candidate.ids[:, session_id] == footprint_id)
                if rows.size > 1:
                    raise ValueError(
                        f"Footprint {footprint_id} in session "
                        f"{session_id} belongs to multiple neurons."
                    )
                if rows.size:
                    candidate.manipulation_id[int(rows[0])] = manipulation_id

        return {
            "plan": plan,
            "source": source,
            "assignment_name": name,
            "candidate": candidate,
            "neuron_id_map": neuron_id_map,
        }

    def assign_neurons(
        self,
        from_file: Optional[str | Path] = None,
        from_data: Optional[SessionData] = None,
        from_session_index: Optional[int] = None,
        align_to_reference=True,
        clean_traces=True,
        force_registration=False,
        p_thr=[0.5, 0.3],
    ):
        if self.assignments is None:
            ## or just directly initialize "local"?
            raise ValueError(
                "No assignments structure defined. Please add an assignments structure before registering neurons."
            )

        this_data = self.get_session(
            from_file=from_file,
            from_data=from_data,
            from_session_index=from_session_index,
            align_to_reference=align_to_reference,
        )
        try:
            assert (
                this_data.included is not None
            ), "Session data must have included defined before registering neurons - run session.get_included_from_footprints() or session.get_included_from_quality() first."

            if self.session_assigned(this_data.id) and not force_registration:
                return

            # Validate dependencies before removing existing assignments.
            self.require_current_alignment(this_data.id)

            other_sessions_assigned = any(
                self.session_assigned(session_id)
                for session_id in range(len(self.sessions))
                if session_id != this_data.id
            )

            if other_sessions_assigned:
                if self.model is None or not self.model.fitted:
                    raise ValueError(
                        "A fitted model is required before registering "
                        "neurons against existing assignments."
                    )

                if self.model.fit_stale:
                    raise ValueError(
                        "The current model fit is outdated; "
                        "update model counts and refit before registering neurons."
                    )

            if force_registration:
                self.unassign_neurons(this_data.id)

            if self.assignments.union is None or self.assignments.union.n_neurons == 0:
                ## first session to be registered, just add all neurons to union and assignments
                self.rebuild_union()
                footprints = this_data.footprints[:, this_data.included]
                self.assignments.union.update_footprints(
                    footprints=footprints,
                    mode="replace",
                    included_values=True,
                    synthetic_values=False,
                )

                actually_good = np.where(this_data.included)[0]
                N_add = len(actually_good)

                self.assignments.pad_empty(n_neurons=N_add, n_sessions=0)

                self.assignments.ids[:, this_data.id] = actually_good

                # # first occurence of neuron defined as p_match = 1, shift = 0
                # self.assignments.stats["p_matched"][:, this_data.id, 0] = 1.0
                # self.assignments.stats["shifts"][:, this_data.id, :] = 0.0
                # self.assignments.stats["fp_corr"][:, this_data.id] = 1.0

                self.assignments.matched_status[this_data.id] = True
                self.mark_assignment_current(this_data.id)

                return

            if self.model is None or not self.model.fitted:
                raise ValueError(
                    "No model is defined. Please add a model before registering neurons."
                )

            ### obtain matching probability from cross session statistics and model
            footprint_shifts, footprint_distances, footprint_correlations, _, _ = (
                calculate_statistics(
                    this_data,
                    self.assignments.union,
                    distance_threshold=self.model.distance_cutoff,
                    nP=12,
                    # params=self.params,
                )
            )
            p_same = calculate_p(
                footprint_distances,
                footprint_correlations,
                self.model.f_same,
                self.params["neighbor_distance"],
            )

            ### ======================================== ###
            ### ==== Hungarian Algorithm (matching) ==== ###
            ### ======================================== ###
            ### run hungarian algorithm (HA)
            ### with (1-p_same) as score
            ### ======================================== ###

            matches = linear_sum_assignment(1 - p_same.toarray())
            p_matched = p_same.toarray()[matches]
            # print("\n \t ## Matching results ##")

            ## thresholds for accepting matches and removing non-matches
            ## (HA matches all pairs, but we only want matches above p_thr)
            idx_TP = np.where(p_matched > p_thr[0])[0]
            # print(matches)
            if len(idx_TP) > 0:
                matched_ref = matches[0][idx_TP]  # matched neurons in s_ref
                matched = matches[1][idx_TP]  # matched neurons in s
            else:
                matched_ref = np.array([], "int")
                matched = np.array([], "int")

            ## find neurons which were not matched in current and reference session
            non_matched_ref = np.setdiff1d(
                list(range(self.assignments.union.n_neurons)), matched_ref
            )
            non_matched = np.setdiff1d(
                list(np.where(this_data.included)[0]), matches[1][idx_TP]
            )
            non_matched = non_matched[this_data.included[non_matched]]

            ## calculate number of matches found
            # TP = np.sum(p_matched > p_thr[0]).astype("float32")

            ## removing footprints from the data which were competing with another one
            ## to be matched and lost, but have significant probability to be the same
            ## this step ensures, that downstream session don't confuse this one and the
            ## 'winner', leading to arbitrary assignments between two clusters
            for nm in non_matched:
                p_all = p_same[:, nm].todense()
                if np.any(p_all > p_thr[1]):
                    #    print(f'!! neuron {nm} is removed, as it is nonmatched and has high match probability:',p_all)[p_all>0])
                    non_matched = non_matched[non_matched != nm]

            ### =================================================== ###
            ### ============== store matching results ============= ###
            ### =================================================== ###

            N_add = len(non_matched)  ## assuming there are never empty rows

            # print(f"Previous shape of assignments: {self.assignments.shape}")
            # print(
            #     f"Session {this_data.path} matched {len(matched)} neurons and added {N_add} new neurons to the union."
            # )

            ## prepare to hold new results by padding existing arrays
            assert (
                self.assignments.ids.shape[1] > this_data.id
            ), "Assignments session axis is not synchronized with registered sessions."

            assert np.all(
                self.assignments.ids[:, this_data.id] == -1
            ), "Session already has assignments; cannot overwrite."
            self.assignments.pad_empty(n_neurons=N_add, n_sessions=0)

            # ... matched neurons are added
            self.assignments.ids[matched_ref, this_data.id] = matched

            self.assignments.stats["p_matched"][matched_ref, this_data.id, 0] = (
                p_matched[idx_TP]
            )
            self.assignments.stats["shifts"][matched_ref, this_data.id, :] = (
                footprint_shifts[matched_ref, matched]
            )
            self.assignments.stats["fp_corr"][matched_ref, this_data.id] = (
                footprint_correlations[matched_ref, matched]
            )

            if N_add > 0:
                ## ... and non-matched (new) neurons are appended
                self.assignments.ids[-N_add:, this_data.id] = non_matched
                # self.assignments.stats["p_matched"][-N_add:, this_data.id, 0] = 1.0

            ## write best non-matching probability
            p_all = p_same.toarray()

            for reference_id, footprint_id in zip(matched_ref, matched):
                alternatives = p_all[reference_id].copy()
                alternatives[footprint_id] = 0.0

                self.assignments.stats["p_matched"][reference_id, this_data.id, 1] = (
                    alternatives.max(initial=0.0)
                )

            if N_add > 0:
                new_rows = np.arange(
                    self.assignments.ids.shape[0] - N_add,
                    self.assignments.ids.shape[0],
                )
                self.assignments.stats["p_matched"][new_rows, this_data.id, 1] = p_all[
                    :, non_matched
                ].max(axis=0, initial=0.0)
            # p_all = p_same.toarray()
            # self.assignments.stats["p_matched"][matched_ref, this_data.id, 1] = [
            #     max(
            #         p_all[
            #             c,
            #             np.where(
            #                 p_all[c, :]
            #                 != self.assignments.stats["p_matched"][c, this_data.id, 0]
            #             )[0],
            #         ]
            #     )
            #     for c in matched_ref
            # ]

            # self.assignments.stats["p_matched"][non_matched, this_data.id, 1] = np.max(
            #     p_all[non_matched, :], axis=1
            # )

            self.update_union_footprints(
                this_data.footprints,
                self.assignments.ids[:, this_data.id],
                weights=self.assignments.stats["fp_corr"][:, this_data.id],
                shifts=self.assignments.stats["shifts"][:, this_data.id, :],
            )

            ## ... and finalize!
            self.assignments.matched_status[this_data.id] = True

            # if np.any(np.all(self.tracking["p_matched"] > 0.9, axis=2)):
            #     print("double match!")
            #     return

            self.mark_assignment_current(this_data.id)
        finally:
            if clean_traces:
                this_data.clean_data("traces")

    def exclude_component(self, component: NeuronComponent) -> NeuronComponent:

        if self.assignments is None:
            raise ValueError("No assignments available.")

        if component.session_id is None:
            raise ValueError("A concrete session component " "is required.")

        session_id = int(component.session_id)
        neuron_id = int(component.neuron_id)

        footprint_id = int(self.assignments.ids[neuron_id, session_id])

        if footprint_id < 0:
            raise ValueError(f"{component} has no footprint.")

        session = self.sessions[session_id]

        # ----------------------------------------------
        # Retire actual detected component.
        # ----------------------------------------------
        session.included[footprint_id] = False

        # ----------------------------------------------
        # Add singleton bookkeeping neuron.
        # ----------------------------------------------

        self.assignments.pad_empty(n_neurons=1, n_sessions=0)
        retired_neuron_id = self.assignments.ids.shape[0] - 1

        # Remove from original tracked neuron.
        self.assignments.ids[neuron_id, session_id] = -1

        # And retain it under its own identity.
        self.assignments.ids[retired_neuron_id, session_id] = footprint_id

        # ----------------------------------------------
        # Only these two union neurons changed.
        # ----------------------------------------------

        self.rebuild_union_neurons([neuron_id, retired_neuron_id])

        # This is deliberately stored rather than inferred.
        self.assignments.union.included[retired_neuron_id] = False
        self.assignments.union.synthetic[retired_neuron_id] = False

        # GUI mirror.
        self.state.assignments = self.assignments.ids

        self.notify_change(("assignments", -1))

        return NeuronComponent(neuron_id=retired_neuron_id, session_id=session_id)

    def update_union_footprints(
        self,
        footprints_new,
        assignments_new,
        weights: Optional[np.ndarray] = None,
        shifts: Optional[np.ndarray] = None,
    ):
        if self.assignments is None:
            return

        if self.assignments.union is None:
            self.assignments.union = SessionData()

        ### =================================================== ###
        ### ========= update reference data structure ========= ###
        ### =================================================== ###
        ### update footprint shapes of matched neurons with
        ### A_ref = (1-p/2)*A_ref + p/2*A
        ### to maintain part or all of original shape,
        ### depending on p_matched
        ### =================================================== ###

        ## get proper references:
        nA_prev = self.assignments.union.footprints.shape[1]
        nA_post = len(assignments_new)
        # print(f"Updating union footprints: nA_prev={nA_prev}, nA_post={nA_post}")

        ## shift union footprints to "new" location of neuron to ensure proper union construction
        if shifts is None:
            shifts = np.zeros((nA_post, 2))
        distances = np.sqrt(np.square(shifts).sum(axis=-1))

        if weights is None:
            weights = np.full(nA_post, 1.0)

        fp_updated: List[sparse.csc_matrix] = []
        for n_idx, fp_idx in enumerate(assignments_new):

            footprint_existed = (
                n_idx < nA_prev
                and self.assignments.union.footprints[:, n_idx].sum() > 0
            )

            if fp_idx < 0 and footprint_existed:
                # print(f"Neuron {n_idx} not present in session, retaining existing footprint.")
                ## neuron not present in session:
                ## retain existing footprint
                fp_updated.append(self.assignments.union.footprints[:, n_idx])

            elif fp_idx < 0 and not footprint_existed:
                # print(f"Neuron {n_idx} not present in session and not detected before, creating dummy footprint.")
                ## neuron not present in session and not detected before:
                ## create dummy entry
                fp_updated.append(sparse.csc_matrix((footprints_new.shape[0], 1)))

            elif fp_idx >= 0 and footprint_existed:
                # print(f"Neuron {n_idx} matched to previous, updating footprint.")
                ## neuron matched to previous:
                ## update footprint accordingly
                fp_updated.append(
                    (
                        _shift_sparse_bilinear(
                            self.assignments.union.footprints[:, n_idx],
                            self.assignments.union.dims,
                            -shifts[n_idx, 0],
                            -shifts[n_idx, 1],
                            order="C",
                        )
                        if distances[n_idx] > 0.5
                        else self.assignments.union.footprints[:, n_idx]
                    ).multiply(1 - weights[n_idx] / 2)
                    + footprints_new[:, fp_idx].multiply(weights[n_idx] / 2)
                )

            elif fp_idx >= 0 and not footprint_existed:
                # print(f"Neuron {n_idx} new in session, assigning current footprint.")
                ## neuron new in session:
                ## assign current footprint
                fp_updated.append(footprints_new[:, fp_idx])

        # ## update union data
        self.assignments.union.update_footprints(
            footprints=sparse.hstack(fp_updated, format="csc"),
            mode="replace",
            included_values=True,
            synthetic_values=False,
        )

    def _build_union_footprint_for_neuron(self, neuron_id: int) -> sparse.csc_matrix:
        """
        Rebuild one union footprint from the currently assigned
        session footprints.

        Uses the same sequential shift/weighted-update logic as
        update_union_footprints().
        """

        if self.assignments is None:
            raise ValueError("No assignments available.")

        union = self.assignments.union

        if union is None:
            raise ValueError("No union data available.")

        if not (0 <= neuron_id < self.assignments.ids.shape[0]):
            raise IndexError(f"Invalid neuron ID {neuron_id}.")

        footprint = None

        shifts = self.assignments.stats.get("shifts")
        weights = self.assignments.stats.get("fp_corr")

        for session_id in range(self.assignments.ids.shape[1]):

            fp_id = int(self.assignments.ids[neuron_id, session_id])

            if fp_id < 0:
                continue

            session = self.sessions[session_id]

            current = session.footprints[:, fp_id]

            # First occurrence defines the initial
            # union footprint.
            if footprint is None:
                footprint = current.copy()
                continue

            shift = shifts[neuron_id, session_id] if shifts is not None else np.zeros(2)
            weight = weights[neuron_id, session_id] if weights is not None else 1.0

            if any(np.isnan(shift)):
                shift = np.zeros_like(shift)
            if np.isnan(weight):
                weight = 1.0

            distance = np.sqrt(np.square(shift).sum())
            if distance > 0.5:
                footprint = _shift_sparse_bilinear(
                    footprint, union.dims, -shift[0], -shift[1], order="C"
                )

            footprint = footprint.multiply(1 - weight / 2) + current.multiply(
                weight / 2
            )

        if footprint is None:

            n_pixels = (
                union.footprints.shape[0]
                if union.footprints.shape[0] > 0
                else self.sessions[0].footprints.shape[0]
            )
            footprint = sparse.csc_matrix((n_pixels, 1))

        return footprint.tocsc()

    @staticmethod
    def _replace_csc_columns(
        matrix: sparse.csc_matrix,
        replacements: dict[
            int,
            sparse.csc_matrix,
        ],
    ) -> sparse.csc_matrix:

        matrix = matrix.tocsc()

        n_rows, n_cols = matrix.shape

        data_parts = []
        index_parts = []

        indptr = np.zeros(n_cols + 1, dtype=matrix.indptr.dtype)

        nnz = 0

        for column in range(n_cols):

            replacement = replacements.get(column)

            if replacement is None:

                start = matrix.indptr[column]
                stop = matrix.indptr[column + 1]

                data = matrix.data[start:stop]

                indices = matrix.indices[start:stop]

            else:

                replacement = replacement.tocsc()

                if replacement.shape != (
                    n_rows,
                    1,
                ):
                    raise ValueError(
                        "Replacement column has "
                        f"shape {replacement.shape}, "
                        f"expected {(n_rows, 1)}."
                    )

                data = replacement.data
                indices = replacement.indices

            data_parts.append(data)
            index_parts.append(indices)

            nnz += len(data)
            indptr[column + 1] = nnz

        data = (
            np.concatenate(data_parts)
            if data_parts
            else np.asarray([], dtype=matrix.dtype)
        )

        indices = (
            np.concatenate(index_parts)
            if index_parts
            else np.asarray([], dtype=matrix.indices.dtype)
        )

        return sparse.csc_matrix(
            (data, indices, indptr),
            shape=matrix.shape,
        )

    def rebuild_union(self):

        if self.assignments is None:
            return

        self.assignments.union = SessionData(name="union")

        for session_id, assignment_ids in enumerate(self.assignments.ids.T):

            # A registered session may deliberately have an
            # empty assignments column:
            #
            # - not loaded yet
            # - alignment failed
            # - tracking postponed
            #
            # Such a session contributes nothing to the union.
            if not np.any(assignment_ids >= 0):
                continue

            weights = self.assignments.stats.get("fp_corr")

            if weights is not None:
                weights = weights[:, session_id]

            shifts = self.assignments.stats.get("shifts")

            if shifts is not None:
                shifts = shifts[:, session_id]

            self.update_union_footprints(
                self.sessions[session_id].footprints, assignment_ids, weights, shifts
            )

        self.rebuild_union_included()

    def rebuild_union_included(self):
        """
        checks if at least one footprint per neuron is included
        (should only be either all or none)
        """

        if self.assignments is None or self.assignments.union is None:
            return

        included = np.zeros(self.assignments.ids.shape, dtype=bool)
        for session_id, ids in enumerate(self.assignments.ids.T):
            neuron_ids = np.where(ids >= 0)[0]
            if neuron_ids.size == 0:
                continue
            included[neuron_ids, session_id] = self.sessions[session_id].included[
                ids[neuron_ids]
            ]

        self.assignments.union.included = np.any(included, axis=1)

    def rebuild_union_neurons(self, neuron_ids):
        """
        Rebuild only selected neuron columns of the union.

        All other union footprints and centroids remain unchanged.
        """

        if self.assignments is None:
            raise ValueError("No assignments available.")

        union = self.assignments.union

        if union is None:
            raise ValueError("No union data available.")

        neuron_ids = np.unique(np.atleast_1d(neuron_ids).astype(int))

        n_neurons = self.assignments.ids.shape[0]

        if np.any((neuron_ids < 0) | (neuron_ids >= n_neurons)):
            raise IndexError("Invalid neuron ID in " f"{neuron_ids}.")

        # ==================================================
        # Make sure union storage has a column for every
        # assignment row.
        #
        # This matters when exclude_component() has just
        # appended a singleton neuron.
        # ==================================================

        n_union = union.footprints.shape[1]

        if n_union > n_neurons:
            raise ValueError("Union has more neurons than assignments.")

        if n_union < n_neurons:

            n_add = n_neurons - n_union

            empty = sparse.csc_matrix((union.footprints.shape[0], n_add))

            union.footprints = sparse.hstack([union.footprints, empty], format="csc")

            if union.centroids is None:
                union.centroids = np.full((n_neurons, 2), np.nan)
            else:
                union.centroids = np.pad(
                    union.centroids,
                    ((0, n_add), (0, 0)),
                    constant_values=np.nan,
                )

            if len(union.included) < n_neurons:
                union.included = np.pad(
                    union.included,
                    (0, n_neurons - len(union.included)),
                    constant_values=True,
                )

            if len(union.synthetic) < n_neurons:
                union.synthetic = np.pad(
                    union.synthetic,
                    (0, n_neurons - len(union.synthetic)),
                    constant_values=False,
                )

        # ==================================================
        # Rebuild requested footprints only.
        # ==================================================

        replacements = {
            neuron_id: self._build_union_footprint_for_neuron(int(neuron_id))
            for neuron_id in neuron_ids
        }

        new_block = sparse.hstack(
            [replacements[int(neuron_id)] for neuron_id in neuron_ids],
            format="csc",
        )

        old_block = union.footprints[:, neuron_ids]

        union.footprints = self._replace_csc_columns(union.footprints, replacements)

        # ==================================================
        # Update derived spatial information only for those
        # neuron rows.
        # ==================================================

        union.centroids[neuron_ids, :] = center_of_mass(
            new_block,
            *union.dims,
            convert=union.params.get("pxtomu", 1.0),
        )

        union.n_neurons = n_neurons

        # Keep the automatically generated union background
        # consistent without recomputing the whole projection.
        if union.background is not None:
            delta = new_block.sum(axis=1) - old_block.sum(axis=1)
            union.background += np.asarray(delta).reshape(union.dims)

    def check_assignments_compatibility(self, assignments):

        if not isinstance(assignments, Assignments):
            raise ValueError("assignments must be an instance of Assignments class.")

        n_sessions = len(self.sessions)
        if assignments.ids.shape[1] > n_sessions:
            raise ValueError(
                "The assignments contain more sessions than the current tracking object."
            )

        ## check neuron numbers
        for session_id in range(assignments.ids.shape[1]):
            session = self.sessions[session_id]
            n_neurons_session = session.n_neurons
            n_neurons_assignments = np.max(assignments.ids[:, session_id]) + 1

            # print(f"Session {session_id}: {n_neurons_assignments} neurons in assignments, {n_neurons_session} neurons in session data.")
            if n_neurons_assignments > n_neurons_session:
                warnings.warn(
                    f"Session {session_id} has more neurons in assignments ({n_neurons_assignments}) than in the session data ({n_neurons_session})."
                )
                return False

        ## check session vs union centroids
        # warnings.warn("to be implemented: check if union centroids match session centroids (after alignment)")

        return True

    def move_session(self, session_id: int, new_session_id: int):
        """
        (Re)Moves a session's data from one session ID to another, updating assignments and tracking accordingly.
        """
        if session_id < 0 or session_id >= len(self.sessions):
            raise ValueError(
                "Invalid session_id. It must be within the range of existing sessions."
            )
        if new_session_id >= len(self.sessions):
            raise ValueError(
                "Invalid new_session_id. It must be within the range of existing sessions."
            )

        session = self.sessions.pop(session_id)

        n_assigned = 0 if self.assignments is None else self.assignments.ids.shape[1]

        if new_session_id >= 0:
            self.sessions.insert(new_session_id, session)
            self.reindex_sessions_after_order_change()

            if session_id >= n_assigned or new_session_id >= n_assigned:
                return

        else:
            # remove the session
            self.reindex_sessions_after_order_change()

            if len(self.sessions) == 0:
                ## when last session is removed
                if self.assignments:
                    self.assignments.reset()
                self.reset_data()
                return

            if session_id >= n_assigned or new_session_id >= n_assigned:
                return
        if self.assignments:
            self.assignments.move_session(session_id, new_session_id)

    def unassign_neurons(self, session_id: int):

        if self.assignments is None:
            return
        self.assignments.matched_status[session_id] = False

        if len(self.sessions) == 1:
            self.assignments.reset()
            return
        self.assignments.unassign_neurons(session_id)

    def reindex_sessions_after_order_change(self):
        # print("Reindexing sessions after order change...")
        for session_id, session in enumerate(self.sessions):
            if session is None:
                continue
            session.id = session_id

    def _ensure_assignment_session_count(
        self, assignments: Assignments | None = None
    ) -> None:

        assignment_sets = (
            [assignments]
            if assignments is not None
            else list(self._assignments.values())
        )

        n_sessions = len(self.sessions)

        for assignment in assignment_sets:
            missing = n_sessions - assignment.ids.shape[1]
            if missing < 0:
                raise ValueError(
                    "Assignments contain more session "
                    "columns than registered sessions."
                )

            for _ in range(missing):
                assignment.pad_empty(n_neurons=0, n_sessions=1)

    def classify_sessions(self, interval=None, **kwargs):
        # max_shift=50.0, min_zscore=4.0):
        """
        checks all sessions to pass certain criteria to
        be included in the further analysis
        """
        # max_shift = kwargs.get("max_shift", self.params["max_session_shift"])
        # min_zscore = kwargs.get(
        #     "min_zscore", self.params["min_session_correlation_zscore"]
        # )

        n_session = len(self.sessions)  # alignment["shift"].shape[0]
        status = np.zeros(n_session, bool)

        ## if 'sessions' is provided (tuple), it specifies range
        ## of sessions to be included
        if interval is None:
            sStart = 0  # np.where(~np.all(np.isnan(alignment["c_max"]), axis=1))[0][0]
            sEnd = n_session
        else:
            sStart = max(0, interval[0] - 1)
            sEnd = interval[-1]

        status[sStart:sEnd] = True
        for session in self.sessions:
            session.evaluate_alignment_status()
            status[session.id] = session.status["aligned"]
        ## check for coherence with other sessions (low shift, high correlation)
        # abs_shift = np.array([np.sqrt(x**2 + y**2) for (x, y) in alignment["shift"]])
        # status[abs_shift > max_shift] = False  ## huge shift
        # status[np.nanmedian(alignment["c_zscored"], axis=1) < min_zscore] = False
        # status[np.all(np.isnan(alignment["c_zscored"]), axis=1)] = False

        ## finally, check if data can be loaded properly
        # for s in np.where(status)[0]:

        #     if not alignment["file_paths"][s].exists():
        #         status[s] = False
        # alignment["alignment_status"] = status
        self.alignment_status = status
        return status

    def build_borders(self, margin=0.0):

        dims = self.sessions[0].dims  # dims of first should be the same in all sessions
        shifts = np.array(
            [
                this_data.remap.shift if this_data.remap else (0, 0)
                for this_data in self.sessions
            ]
        )
        thr_high = np.nanmin(dims + shifts, axis=0)
        thr_low = np.nanmax(shifts, axis=0)

        borders = np.vstack([thr_low + margin, thr_high - margin])
        return borders

    def classify_components(self, **kwargs):
        """
        checks all clusters to pass certain criteria to be considered in the analysis

        Each cluster is required to:
            * pass all "lowest" thresholds (SNR, r-value, CNN-prediction)
            * at least one of the "min" thresholds (SNR, r-value, CNN-prediction)
            * be present in at least 'min_cluster_count' sessions
            * have a center of mass within the borders of the imaging window, leaving some margin of 'border_margin'
        """
        # clusters = getattr(self, self.cluster_field)
        assert (
            self.assignments is not None
        ), "No assignments structure defined. Please add an assignments structure before classifying components."

        n_cluster, n_session = self.assignments.ids.shape
        status = np.ones(n_cluster, bool)

        # status_session = alignment.get("alignment_status", None)
        if self.alignment_status is None:
            self.classify_sessions()
            # status_session = np.ones(n_session, bool)
        # else:
        # status_session = self.alignment_status

        ## check for neuron detection thresholds
        # if not hasattr(self, "thr"):
        # set_thresholds()
        thr = {
            # component quality
            "SNR_lowest": 1.0,
            "SNR_min": 2.5,
            "rval_lowest": -1.0,
            "rval_min": 0.6,
            "cnn_lowest": 0.1,
            "cnn_min": 0.9,
            # tracking quality
            "p_matched": 0.3,
            "min_cluster_count": 2.0,
        }

        status_detected = (
            # (
            #     ## minimum requirements for each neuron
            #     (clusters["SNR_comp"] > thr["SNR_lowest"])
            #     & (clusters["r_values"] > thr["rval_lowest"])
            #     & (clusters["cnn_preds"] > thr["cnn_lowest"])
            # )
            # & (
            #     ## each neuron needs to exceed at least one of the following thresholds
            #     (clusters["SNR_comp"] > thr["SNR_min"])
            #     | (clusters["r_values"] > thr["rval_min"])
            #     | (clusters["cnn_preds"] > thr["cnn_min"])
            # )
            # &
            self.assignments.stats["p_matched"][..., 0]
            > thr["p_matched"]
        )

        ## remove components from sessions that are not included in the data
        status_detected[:, ~self.alignment_status] = False

        ## check for presence in at least 'min_cluster_count' sessions
        status[
            status_detected[:, self.alignment_status].sum(1) < thr["min_cluster_count"]
        ] = False

        # if borders is None:
        borders = self.build_borders(kwargs.get("border_margin", 2.0))

        ## check for distance from imaging window borders
        for i in range(2):
            idx_remove_low = self.assignments.union.centroids[:, i] < (borders[0, i])
            # status[np.any(idx_remove_low, 1)] = False
            status[idx_remove_low] = False

            idx_remove_high = self.assignments.union.centroids[:, i] > (borders[1, i])
            # status[np.any(idx_remove_high, 1)] = False
            status[idx_remove_high] = False
        # clusters["status"] = status
        return status

    ### ---- Model manipulation functions ---- ###

    def _update_bins(self, nbins):

        self.params["nbins"] = nbins
        # self.params["arrays"] = self.build_arrays(nbins)

        ## create value arrays for distance and footprint correlation
        arrays = {}
        arrays["distance_bounds"] = np.linspace(
            0, self.params["neighbor_distance"], nbins + 1
        )
        arrays["correlation_bounds"] = np.linspace(0, 1, nbins + 1)

        distance_step = self.params["neighbor_distance"] / nbins
        correlation_step = 1.0 / nbins

        arrays["distance"] = arrays["distance_bounds"][:-1] + distance_step / 2
        arrays["correlation"] = arrays["correlation_bounds"][:-1] + correlation_step / 2

        self.params["arrays"] = arrays

    ### ===================================================== ###
    ### ============= refitting from tracking =============== ###
    ### ===================================================== ###

    def get_footprints_at_session(
        self, s: int, ds=np.inf, s_cuts: list[int] = [], complete_new: bool = False
    ):

        s -= 1  # adjust for 0-indexing

        ## load data from session s to store already detected ones
        session_path = self.sessions[s].path

        print(f"loading data from session {s+1}: {session_path}")
        ld = load_hdf5(session_path, subpath="/estimates")
        dims = ld["Cn"].shape
        print("...done")

        ## classify sessions and components from tracking, to use only good ones
        self.classify_sessions()
        borders = self.build_borders(margin=2.0)
        self.classify_components(borders=borders)

        clusters = getattr(self, self.cluster_field)
        n_cluster = clusters["status"].sum()

        ## prepare some dictionaries for storing in- and output data
        dataIn = {}
        dataOut = {}
        idxes = {
            "in": {
                "active": np.zeros(n_cluster, "bool"),
                "silent": np.zeros(n_cluster, "bool"),
                "match_to_c": None,  # indexing of matching cluster number
                "match_to_n": None,  # indexing to session neuron number
            },
            "out": {
                "active": None,
                "silent": None,
            },
        }

        ## initialize input data with random values
        T = ld["C"].shape[1]
        dataIn["Cn"] = ld["Cn"]
        dataIn["C"] = np.random.rand(n_cluster, T)

        if complete_new:

            ## if (for whatever reason) you just want to throw in n_cluster random footprints
            ## (unsure if this even works without specifying 'footprints')
            idxes["in"]["silent"] = np.ones(n_cluster, "bool")
            dataIn["b"] = np.random.rand(int(np.prod(dims)), 1)
            dataIn["f"] = np.random.rand(1, T)
        else:
            ## hand over data from session s
            ## find active and silent neurons in session s
            detected = self.assignments.ids[:, s] >= 0
            isSilent = clusters["status"] & ~detected
            idxes["in"]["nSilent"] = isSilent.sum()
            isActive = clusters["status"] & detected
            idxes["in"]["nActive"] = isActive.sum()

            idxes["in"]["active"][: idxes["in"]["nActive"]] = True
            idxes["in"]["silent"][idxes["in"]["nActive"] :] = True

            c_idx = np.concatenate([np.where(isActive)[0], np.where(isSilent)[0]])
            n_idx = self.assignments.ids[isActive, s]

            idxes["in"]["match_to_c"] = c_idx
            idxes["in"]["match_to_n"] = n_idx

            # dataIn["A"][:, : idxes["in"]["nActive"]]
            dataIn["A"] = ld["A"][:, n_idx]
            # dataIn["A"] = alignment["A"][str(s)][:, n_idx]

            ## load trace components of active cells from session s
            T1 = ld["C"].shape[1]  # adjusted for a session, where T != T1
            dataIn["C"][: idxes["in"]["nActive"], :T1] = ld["C"][n_idx, :]

            ## load background components from session s
            if not (ld["b"].shape[0] == dataIn["A"].shape[0]):
                ld["b"] = ld["b"].transpose()
            dataIn["b"] = ld["b"]
            if not (ld["f"].shape[1] == dataIn["C"].shape[1]):
                ld["f"] = ld["f"].transpose()
            dataIn["f"] = ld["f"]

        ## if given, find closest measurement cuts (session batches)
        s_cuts = np.array(s_cuts)
        s_min = s_cuts[s_cuts < s].max() if np.any(s_cuts < s) else 0
        s_max = s_cuts[s_cuts > s].min() if np.any(s_cuts > s) else np.inf

        ## find footprint data from adjacent sessions for neurons which were not detected
        print(
            f"There are {idxes['in']['nActive']}/{n_cluster} active neurons in session {s+1} and {idxes['in']['nSilent']} silent neuron footprints are attempted to be reconstructed from adjacent sessions."
        )
        print(
            f"Building silent footprints for session {s+1} from matched footprints from sessions between {s_min} and {s_max}"
        )
        dataIn_silent = np.empty((np.prod(dims), idxes["in"]["nSilent"]))
        for i, c in enumerate(idxes["in"]["match_to_c"][idxes["in"]["silent"]]):

            A_tmp = np.zeros((np.prod(dims), 1))

            ## search closest previous session with valid footprint
            s_pre = np.where(self.assignments.ids[c, :s] >= 0)[0]
            if len(s_pre) > 0 and (s - s_pre[-1]) <= ds and s_pre[-1] >= s_min:
                s_ref = s_pre[-1]
                # print(
                #     f"Found previous session {s_ref+1} for cluster {c} (silent in session {s+1})"
                # )
                n_ref = self.assignments.ids[c, s_ref]
                A_tmp += (
                    1.0 / abs(s_ref - s) * alignment["A"][str(s_ref)][:, n_ref]
                ).toarray()

            ## search closest following session with valid footprint
            s_post = s + 1 + np.where(self.assignments.ids[c, s + 1 :] >= 0)[0]
            if len(s_post) > 0 and (s_post[0] - s) <= ds and s_post[0] < s_max:
                s_ref = s_post[0]
                # print(
                #     f"Found following session {s_ref+1} for cluster {c} (silent in session {s+1})"
                # )
                n_ref = self.assignments.ids[c, s_ref]
                A_tmp += (
                    1.0 / abs(s_ref - s) * alignment["A"][str(s_ref)][:, n_ref]
                ).toarray()
            # print(A_tmp.max())
            dataIn_silent[:, i] = A_tmp.ravel()

        ## remap silent neuron footprints to current session
        dataIn_silent = sparse.csc_matrix(dataIn_silent)
        if np.all(np.isfinite(alignment["shift"][s, :].max())):
            dataIn_silent = apply_remap(
                dataIn_silent,
                dims=(512, 512),
                shift=-alignment["shift"][s, :],
            )

        ## and store in data along with detected neurons
        dataIn["A"] = normalize_sparse_array(
            sparse.hstack([dataIn["A"], dataIn_silent], format="csc")
        )

        return dataIn, dataOut, idxes

    def refit_footprints_from_tracking(self, s: int, **kwargs):
        """
        refits the model using the footprints of neurons detected in session s as "same" distribution
        and all other pairs of neurons as "different" distribution

        kwargs:
            ds: maximum session distance to consider for finding matching footprints for silent neurons (default: np.inf)
            s_cuts: list of session numbers where there are cuts in the data, to limit search for matching footprints (default: [])
        """

        dataIn, dataOut, idxes = self.get_footprints_at_session(s, **kwargs)

        # print("dataIn keys:", dataIn.keys())
        # print("dataOut keys:", dataOut.keys())
        # print("idxes:", idxes)

        p_init, bounds = self.estimate_initial_parameters(dataIn, idxes)

        self.model = fit_model(
            dataIn,
            idxes,
            p_init,
            bounds,
            params=self.params,
        )

    ### ======================================================== ###
    ### ============== saving and loading methods ============== ###
    ### ======================================================== ###

    def load_session_data(self, path: str | Path) -> list[SessionData]:
        backend = get_backend(path)

        with backend.open_read(path) as ref:
            object_type = backend.get_attribute(ref, "/", "object_type")
            if object_type not in {"SessionData", "SessionList"}:
                raise ValueError("Not a CATAN session file.")

            version = int(backend.get_attribute(ref, "/", "format_version", default=1))
            if version not in (1, 2):
                raise ValueError(f"Unsupported session format version: {version}")

            roots = sorted(
                (
                    name
                    for name in backend.list_groups(ref)
                    if name.startswith("session_") and name[8:].isdigit()
                ),
                key=lambda name: int(name[8:]),
            )
            roots = [f"/{name}" for name in roots] or ["/"]

            if version == 2:
                return [
                    read_session_snapshot(backend, ref, root=root) for root in roots
                ]

            # Legacy files lack reliable alignment/source-config state.
            # Restore source references; subsequent loading uses raw data.
            result = []

            for root in roots:
                original_path = backend.get_attribute(ref, "/", "path", root=root)
                if isinstance(original_path, bytes):
                    original_path = original_path.decode("utf-8")

                if not original_path:
                    raise ValueError(
                        f"Legacy session at {root} has no original source path."
                    )

                result.append(SessionData(path=str(original_path)))

            return result

    def save_sessions(
        self,
        path: str | Path,
        *,
        mat_version: Literal["pre73", "7.3"] = "7.3",
        object_type: str = "SessionList",
    ) -> None:
        if not self.sessions:
            raise ValueError("No sessions to save.")

        backend = get_backend(path, for_write=True, mat_version=mat_version)

        with backend.open_write(path) as ref:
            backend.set_attribute(ref, "/", "object_type", object_type)
            backend.set_attribute(ref, "/", "format_version", 2)

            for session_id, session in enumerate(self.sessions):
                state = self._processing_state(session_id)

                write_session_snapshot(
                    backend,
                    ref,
                    session,
                    root=f"/session_{session_id:03d}",
                    processing={
                        "geometry_revision": state.geometry_revision,
                        "alignment_stale": state.alignment_stale,
                    },
                )

    ### ================================================= ###
    ### === HANDOVER FUNCTIONS FOR SAVING AND LOADING === ###
    ### ================================================= ###

    def save_model(
        self,
        output_fname: Optional[str | Path] = None,
        suffix: str = "",
        ext: str = ".hdf5",
    ):

        if self.model is None:
            raise ValueError("No model to save. Please fit a model before saving.")

        if output_fname is None:
            output_fname = self.get_result_directory() / f"catan_model{suffix}{ext}"
        else:
            self.get_result_directory(Path(output_fname).parent)

        self.model.save(str(output_fname))

    def save_assignments(
        self,
        output_fname: Optional[str | Path] = None,
        suffix: str = "",
        ext: str = ".hdf5",
    ):
        if self.assignments is None:
            raise ValueError(
                "No assignments to save. Please run the registration before saving."
            )

        if output_fname is None:
            output_fname = (
                self.get_result_directory() / f"catan_registration{suffix}{ext}"
            )
        else:
            self.get_result_directory(Path(output_fname).parent)
        self.assignments.save(str(output_fname))

    def get_result_directory(
        self,
        output_directory: Path | str | None = None,
    ) -> Path:
        if output_directory is None:
            output_directory = self._default_matching_directory()

        output_directory = Path(output_directory).expanduser().resolve()
        assert isinstance(
            output_directory, Path
        ), f"output_directory {output_directory} should be a Path object"

        output_directory.mkdir(
            parents=True,
            exist_ok=True,
        )

        return output_directory

    def _default_matching_directory(
        self,
    ) -> Path:

        candidate_paths = [
            Path(session.path).parent
            for session in self.sessions
            if session.path is not None
        ]
        common_path = os.path.commonpath([str(path) for path in candidate_paths])
        return Path(common_path) / "tracking"
