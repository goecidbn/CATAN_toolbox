"""
function written by Alexander Schmidt, based on the paper "Sheintuch et al., ...", allowing for complete registration of neuron footprints across several sessions

TODO:
  * write plotting procedure for cluster footprints (3D), to allow manual corrections
  * save data-attribute / structure after model-building, not only after registration
  * change save structure, such that all that is needed for further analysis is readily accessible:
      - filePath of results file
      - no redundancy in SNR, r_values, cnn saving
      - 'remap' into 'alignment' structure in results
      - cm only needed once

last updated on January 28th, 2024
"""

import os
from typing import Any, Dict, Optional, Tuple, List, Union, Literal
import sys, copy, logging, time, numbers, warnings

# from catan.core.structures.load_config import LoadConfig
# from catan.core.structures.load_config_manager import LoadConfigManager
# from platformdirs import user_config_dir

from pathlib import Path

from catan.core.io import NATIVE_SESSION_CONFIG, get_backend
from catan.core.structures.load_config.config import LoadConfig, FieldSpec
import numpy as np
from scipy import sparse
from scipy.optimize import linear_sum_assignment

from catan.core.structures import SessionData
from catan.core.analysis import calculate_statistics, calculate_p
from catan.core.alignment import _shift_sparse_bilinear

from .structures import Model, Assignments

logging.basicConfig(level=logging.INFO)


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

        # self.load_configs = LoadConfigManager(
        #     user_dir=(
        #         Path(user_config_dir("CATAN"))
        #         / "load_configs"
        #     ),
        #     default_config="CaImAn",
        # )

        # self.kernel = {"idxes": {}, "kde": {}}
        self.reference_data = None

        self.reset_data()

        self.counts = {
            "same": np.zeros((self.params["nbins"], self.params["nbins"]), int),
            "cross": np.zeros((self.params["nbins"], self.params["nbins"], 3), int),
        }


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

    def add_model(self, name: str, model: Optional[str|Model] = None):

        if model is None:
            model = Model(params=self.params)
        elif isinstance(model, str):
            model = Model._from_file(path=model, params=self.params)

        if not isinstance(model, Model):
            raise ValueError("model must be an instance of Model class or a path to a saved model.")
        
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

    def add_assignments(self, name: str, assignments: Optional[str|Assignments]=None):
        if assignments is None:
            assignments = Assignments()
        elif isinstance(assignments, str):
            assignments = Assignments._from_file(path=assignments)

        if not isinstance(assignments, Assignments):
            raise ValueError("assignments must be an instance of Assignments class or a path to a saved assignments.")

        ok = self.check_assignments_compatibility(assignments)
        if not ok:
            raise ValueError("The provided assignments are not compatible with the current model and sessions.")
        self._assignments[name] = assignments
        self._current_assignments = name

        self.update_sessions_with_assignments()
        
    @property
    def available_assignments(self) -> List[str]:
        return list(self._assignments.keys())

    def update_sessions_with_assignments(self):

        for session in self.sessions:
            session.status["matched"] = False

        if self.assignments is None:
            return

        for session_id, (session, assignment_ids) in enumerate(zip(self.sessions, self.assignments.ids.T)):

            if np.any(assignment_ids >= 0):
                # mark session as matched, if it has assignments
                session.status["matched"] = True

            update_idx_eval = False
            if session.idx_eval is not None:
                idx_assigned_from_session = np.where(session.idx_eval)[0]
                idx_assigned_from_assignments = assignment_ids[assignment_ids >= 0]

                assignment_in_session = np.isin(idx_assigned_from_assignments, idx_assigned_from_session)
                if not assignment_in_session.all():
                    # warnings.warn(f"Session {session_id} has neurons in 'assignments' that are not marked as valid in the session data (idx_eval).")
                    # warnings.warn(f"Neurons in assignments but not in session idx_eval: {idx_assigned_from_assignments[~assignment_in_session]}")
                    update_idx_eval = True
                
                session_in_assignment = np.isin(idx_assigned_from_session, idx_assigned_from_assignments)
                if not session_in_assignment.all():
                    # warnings.warn(f"Session {session_id} has neurons marked as valid in the session data (idx_eval) that are not present in 'assignments'.")
                    # warnings.warn(f"Neurons in session idx_eval but not in assignments: {idx_assigned_from_session[~session_in_assignment]}")
                    update_idx_eval = True
            else:
                # warnings.warn(f"Session {session_id} does not have 'idx_eval' defined. It will be updated based on 'assignments'.")
                update_idx_eval = True

            if update_idx_eval:
                session.idx_eval = np.zeros(session.n_neurons, dtype=bool)
                session.idx_eval[assignment_ids[assignment_ids >= 0]] = True

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

            this_data = SessionData.from_file(
                str(from_file),
                fields_to_load, 
                self.alignment_template if align_to_reference else None
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
            this_data = self.get_session(from_file=from_file, fields_to_load=fields_to_load, align_to_reference=align)

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
            raise ValueError(
                "Either from_file or from_data must be provided."
            )
        paths = [session.path for session in self.sessions]
        if this_data.path in paths:
            raise ValueError(
                f"Session {this_data.path} is already registered."
            )
        
        this_data.id = len(self.sessions)

        self.sessions.append(this_data)

        return this_data.id

    @property
    def alignment_template(self):

        if len(self.sessions) == 0:
            return None

        aligned_sessions = [
            session for session in self.sessions if session.status["aligned"]
        ]
        if len(aligned_sessions) == 0:
            return None

        alignment_window = min(len(aligned_sessions), 10)

        return np.stack(
            [session.background for session in aligned_sessions[-alignment_window:]],
            axis=0,
        )

    ### ============================================ ###
    ### ============= COUNT REGISTRATION =========== ###
    ### ============================================ ###

    def update_counts_with_data(
        self,
        from_file: Optional[str | Path] = None,
        from_data: Optional[SessionData] = None,
        from_session_index: Optional[int] = None,
        align_to_reference=True,
    ):
        """
        takes existing model and adds new data from footprints to it
        """
        this_data = self.get_session(
            from_file=from_file,
            from_data=from_data,
            from_session_index=from_session_index,
            align_to_reference=align_to_reference,
        )

        if not this_data.status["aligned"]:
            print(
                f"[model update] Session {this_data.id} ({this_data.path}) did not pass quality criteria, skipping."
            )
            return
        
        if this_data.status["registered_to_model"]:
            # print(
            #     f"[model update] Session {this_data.id} ({this_data.path}) already registered to model, skipping."
            # )
            return

        # build both models: self and cross (nNN from self and NN from cross)
        self.update_model_counts(this_data, mode="same")
        # self.this_data = this_data

        if self.reference_data is not None:
            self.update_model_counts(this_data, mode="to_reference")
        this_data.status["registered_to_model"] = True
        # self.alignment_template = copy.deepcopy(this_data.background)
        self.reference_data = copy.deepcopy(this_data)


    def update_model_counts(self, this_data: SessionData, mode="to_reference"):
        """
        Function to update counts in the joint model

        inputs:
        - s,s_ref: int / string
            key of current (s) and reference (s_ref) session
        - use_kde: bool
            defines, whether kde (kernel density estimation) is used to ...

            TODO:
            * might just change everything to require "total counts" only, removing NN-calculation
            * change how correlation is calculated: just apply centroid distance shift! (test performance/timing before that)
        """

        # print(this_data)
        if mode == "to_reference":
            ## compare to reference session
            ref_data = self.reference_data
        elif mode == "same":
            ## find and mark potential duplicates of neuron footprints in session
            ref_data = this_data
        else:
            raise ValueError("mode must be 'to_reference' or 'same'")
        assert isinstance(ref_data, SessionData), "Reference data not defined!"

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

        if mode == "same" and len(idx_remove) > 0:
            this_data.idx_eval[idx_remove] = False

        idx_this = this_data.idx_eval
        idx_ref = ref_data.idx_eval

        ### ======================================== ###
        ### =========== define neighbours ========== ###
        ### ======================================== ###
        ## find all neuron pairs below a distance threshold
        neighbors = footprint_distances < self.params.get("neighbor_distance", 15.0)
        is_NN = np.zeros((ref_data.n_neurons, this_data.n_neurons), bool)
        if mode == "to_reference":
            min_distance = np.nanmin(footprint_distances, axis=1)
            idx_finite = ~np.isnan(min_distance)

            min_distance_idx = np.nanargmin(
                footprint_distances[idx_ref & idx_finite, :], axis=1
            )
            # min_distance_idx = np.nanargmin(footprint_distances, axis=1)
            is_NN[
                idx_ref & idx_finite,
                min_distance_idx,
            ] = True
        else:
            is_NN[idx_this, idx_this] = True

        # print(f"number of neighbor pairs: {np.sum(neighbors)} ({np.sum(is_NN)} NN)")
        t_start = time.time()
        histo_options = {
            "bins": self.params["nbins"],
            "range": [
                self.params["arrays"]["distance_bounds"][[0, -1]],
                self.params["arrays"]["correlation_bounds"][[0, -1]],
            ],
        }
        if mode == "same":
            idxes = neighbors & ~is_NN & ref_data.idx_kde[:, None]
            # print(idxes.sum(), "counts to add")

            self.counts["same"] += np.histogram2d(
                footprint_distances[idxes],
                # footprint_correlations["shifted"][idxes],
                footprint_correlations[idxes],
                **histo_options,
            )[0].astype(int)

        else:
            idxes = neighbors & ref_data.idx_kde[:, None]
            self.counts["cross"][..., 0] += np.histogram2d(
                footprint_distances[idxes],
                # footprint_correlations["shifted"][idxes],
                footprint_correlations[idxes],
                **histo_options,
            )[0].astype(int)

            idxes = neighbors & is_NN & ref_data.idx_kde[:, None]
            self.counts["cross"][..., 1] += np.histogram2d(
                footprint_distances[idxes],
                # footprint_correlations["shifted"][idxes],
                footprint_correlations[idxes],
                **histo_options,
            )[0].astype(int)

            idxes = neighbors & ~is_NN & ref_data.idx_kde[:, None]
            self.counts["cross"][..., 2] += np.histogram2d(
                footprint_distances[idxes],
                # footprint_correlations["shifted"][idxes],
                footprint_correlations[idxes],
                **histo_options,
            )[0].astype(int)

        t_end = time.time()
        # print(f"Updating joint model took {t_end - t_start:.2f} seconds.")

    ### ============================================ ###
    ### =========== ASSIGNMENT FUNCTIONS =========== ###
    ### ============================================ ###
    
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
            raise ValueError("No assignments structure defined. Please add an assignments structure before registering neurons.")
        
        this_data = self.get_session(
            from_file=from_file,
            from_data=from_data,
            from_session_index=from_session_index,
            align_to_reference=align_to_reference,
        )
        assert this_data.idx_eval is not None, "Session data must have idx_eval defined before registering neurons - run session.get_idx_eval_from_footprints() or session.get_idx_eval_from_quality() first."

        if force_registration:
            self.unassign_neurons(this_data.id)

        if this_data.status["matched"]:
            # print(
            #     f"[register] Session {this_data.name} already registered, skipping."
            # )
            return

        if not this_data.status["aligned"] or this_data.footprints is None:

            print(
                f"[register] Session {this_data.path} did not pass quality criteria, skipping."
            )
            self.assignments.pad_empty(n_neurons=0, n_sessions=1)

            if clean_traces:
                this_data.clean_data("traces")
            return

        if not self.assignments.union.status["spatial_loaded"]:
            ## first session to be registered, just add all neurons to union and assignments

            footprints = this_data.footprints[:, this_data.idx_eval]
            self.assignments.union.register_spatial(footprints=footprints, dims=this_data.dims)

            actually_good = np.where(this_data.idx_eval)[0]
            N_add = len(actually_good)

            self.assignments.pad_empty(n_neurons=N_add, n_sessions=1)

            self.assignments.ids[:, this_data.id] = actually_good

            # first occurence of neuron defined as p_match = 1, shift = 0
            self.assignments.stats["p_matched"][:, this_data.id, 0] = 1.0
            self.assignments.stats["shifts"][:, this_data.id, :] = 0.0

            this_data.status["matched"] = True
            if clean_traces:
                this_data.clean_data("traces")
            return

        if self.model is None or not self.model.fitted:
            raise ValueError("No model is defined. Please add a model before registering neurons.")

        
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
        non_matched_ref = np.setdiff1d(list(range(self.assignments.union.n_neurons)), matched_ref)
        non_matched = np.setdiff1d(
            list(np.where(this_data.idx_eval)[0]), matches[1][idx_TP]
        )
        non_matched = non_matched[this_data.idx_eval[non_matched]]

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
        ### ========= update reference data structure ========= ###
        ### =================================================== ###
        ### update footprint shapes of matched neurons with 
        ### A_ref = (1-p/2)*A_ref + p/2*A
        ### to maintain part or all of original shape, 
        ### depending on p_matched
        ### =================================================== ###

        ## shift union footprints to "new" location of neuron to ensure proper union construction
        shifted = sparse.hstack(
            [
                (
                    _shift_sparse_bilinear(
                        self.assignments.union.footprints[:, m_ref],  # .reshape(512, 512),
                        self.assignments.union.dims,
                        -footprint_shifts[m_ref, m, 0],
                        -footprint_shifts[m_ref, m, 1],
                        order="C",
                        # output_format="csc",
                    )  # .reshape(-1, 1)
                    if footprint_distances[m_ref, m] > 0.5
                    else self.assignments.union.footprints[:, m_ref]
                ).multiply(1 - footprint_correlations[m_ref, m] / 2)
                + this_data.footprints[:, m].multiply(footprint_correlations[m_ref, m] / 2)
                for m_ref, m in zip(matched_ref, matched)
            ],
            format="csc",
        )

        # self.assignments.union.footprints[:, matched_ref] = self.assignments.union.footprints[:, matched_ref].multiply(
        #     1 - p_matched[idx_TP] / 2
        # ) + this_data.footprints[:, matched].multiply(p_matched[idx_TP] / 2)

        self.assignments.union.footprints.toarray()[:, matched_ref] = shifted.toarray()
        ## append new neuron footprints to union
        footprints_updated = sparse.hstack(
            [sparse.coo_matrix(self.assignments.union.footprints), this_data.footprints[:, non_matched]],
            format="csc",
        )
        # ## update union data
        self.assignments.union.register_spatial(footprints=footprints_updated)

        # print(f"union now holds {self.assignments.union.n_neurons} neurons after session {this_data.id} ({this_data.path}) was registered.")
        # print(f"Shape of footprints: {self.assignments.union.footprints.shape}, shape of idx_eval: {this_data.idx_eval.shape}")

        ### =================================================== ###
        ### ============== store matching results ============= ###
        ### =================================================== ###

        N_add = len(non_matched)  ## assuming there are never empty rows

        # print(f"Previous shape of assignments: {self.assignments.shape}")
        # print(
        #     f"Session {this_data.path} matched {len(matched)} neurons and added {N_add} new neurons to the union."
        # )

        ## prepare to hold new results by padding existing arrays
        if self.assignments.ids.shape[1] <= this_data.id:
            ## either append to end
            self.assignments.pad_empty(n_neurons=N_add, n_sessions=1)
        else:
            ## or just write into already existing rows, if possible
            assert np.all(
                self.assignments.ids[:, this_data.id] == -1
            ), "Session already has assignments, cannot overwrite!"
            # print(f"adding {N_add} new neurons to union for session {this_data.id}")
            self.assignments.pad_empty(n_neurons=N_add, n_sessions=0)


        # ... matched neurons are added
        self.assignments.ids[matched_ref, this_data.id] = matched

        self.assignments.stats["p_matched"][matched_ref, this_data.id, 0] = p_matched[idx_TP]
        self.assignments.stats["shifts"][matched_ref, this_data.id, :] = footprint_shifts[
            matched_ref, matched
        ]

        if N_add>0:
            ## ... and non-matched (new) neurons are appended
            self.assignments.ids[-N_add:, this_data.id] = non_matched
            self.assignments.stats["p_matched"][-N_add:, this_data.id, 0] = 1.0

        ## write best non-matching probability
        p_all = p_same.toarray()
        self.assignments.stats["p_matched"][matched_ref, this_data.id, 1] = [
            max(
                p_all[
                    c,
                    np.where(
                        p_all[c, :] != self.assignments.stats["p_matched"][c, this_data.id, 0]
                    )[0],
                ]
            )
            for c in matched_ref
        ]

        ## ... and finalize!
        this_data.status["matched"] = True
        if clean_traces:
            this_data.clean_data("traces")

        # if np.any(np.all(self.tracking["p_matched"] > 0.9, axis=2)):
        #     print("double match!")
        #     return

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

        self.sessions[session_id].status["matched"] = False

        if self.assignments is None:
            return

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
        assert self.assignments is not None, "No assignments structure defined. Please add an assignments structure before classifying components."

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
    
    def load_session_data(
        self,
        path: str | Path,
        fields_to_load: dict[str, dict[str, FieldSpec]] | None = None,
        *,
        # mat_version: Literal["pre73", "7.3"] = "7.3",
        test_object_type: str = "SessionData",
    ) -> list[dict[str, Any]]:
        """Load a CATAN-native session container."""
        data = []

        backend = get_backend(path, for_write=False)
        with backend.open_read(path) as ref:
            object_type = backend.get_attribute(ref, "/", "object_type")
            if object_type == test_object_type:
            #     raise ValueError(
            #         f"Invalid object type: expected 'SessionData', got '{object_type}'"
            #     )

                fields_to_load = LoadConfig.fields_from_resource(
                    NATIVE_SESSION_CONFIG,
                    enabled_only=False,
                )
                n_sessions = backend.get_attribute(ref, "/", "n_sessions")
                for s in range(n_sessions):
                    data.append(backend.load(ref, fields_to_load, root=f"/session_{s:03d}"))

            else:
                if fields_to_load is None:
                    fields_to_load = LoadConfig.fields_from_resource(
                        NATIVE_SESSION_CONFIG,
                        enabled_only=False,
                    )
                data_out = backend.load(ref, fields_to_load)
                if data_out.get("metadata") is None:
                    data_out["metadata"] = {}
                if data_out["metadata"].get("path") is None:
                    data_out["metadata"]["path"] = str(path)
                data.append(data_out)
        
        return data

    def save_sessions(
        self,
        path: str | Path,
        *,
        mat_version: Literal["pre73", "7.3"] = "7.3",
        object_type: str = "SessionData",
    ) -> None:
        """Save one or several sessions as a CATAN-native session container.

        If ``fields_to_save`` is omitted, the packaged ``catan_session.json``
        structure is used. A single SessionData object is still stored below
        ``session_000`` so the native container layout remains uniform.
        """
        
        fields_to_save = LoadConfig.fields_from_resource(
            NATIVE_SESSION_CONFIG,
            enabled_only=False,
        )
        backend = get_backend(path, for_write=True, mat_version=mat_version)

        with backend.open_write(path) as ref:
            backend.set_attribute(ref, "/", "object_type", object_type)
            backend.set_attribute(ref, "/", "format_version", 1)

            for session in self.sessions:
                backend.write(ref, session, fields_to_save, root=f"/session_{session.id:03d}")
                
    # def save_sessions(
    #     self,
    #     output_fname: Optional[str | Path] = None,
    #     suffix: str = "",
    #     ext: str = ".hdf5",
    # ):
    #     if output_fname is None:
    #         output_fname = self.get_result_directory() / f"catan_model{fix_suffix(suffix)}{ext}"
    #     else:
    #         self.get_result_directory(Path(output_fname).parent)

    #     print(f"Saving session data to {output_fname}...")

    #     if ext in [".h5", ".hdf5"]:
    #         with h5py.File(
    #             output_fname, "w"
    #         ) as f:
    #             self.save_sessions_to_hdf5(f)
    #     else:
    #         raise ValueError(f"Unsupported file extension: {ext}. Use '.h5' or '.hdf5'.")
    #     print(f"Saved session data to {output_fname}")


    # def save_sessions_to_hdf5(self, h5ref: h5py.Group | h5py.File) -> None:
    #     print("Saving session data to HDF5...")
    #     h5ref.attrs["object_type"] = "SessionData"
    #     h5ref.attrs["schema_version"] = self.HDF5_VERSION

    #     # sessions_group = group.create_group("sessions")
    #     h5ref.attrs["n_sessions"] = len(self.sessions)

    #     for session in self.sessions:
    #         session_group = h5ref.create_group(f"session_{session.id:03d}")
    #         session.to_hdf5(session_group)
        
    # def load_session_data(self, fname: str | Path, fields_to_load: Optional[dict] = None) -> List[SessionData]:
    #     """
    #     Triggers loading data from a file. Depending on the provided file, it either loads a single session or multiple sessions (informed by hdf5 attributes).
    #     """
    #     ext = Path(fname).suffix
    #     if ext in [".h5", ".hdf5"]:
    #         with h5py.File(fname, "r") as h5ref:
                
    #             if h5ref.attrs.get("object_type") == "SessionData":
    #                 return self.load_sessions_from_hdf5(h5ref)
    #             else:
    #                 this_data = SessionData(path=fname)
    #                 this_data.from_hdf5(h5ref, fields_to_load=fields_to_load)
    #                 if this_data.path is None:
    #                     this_data.path = str(fname)
    #                 return [this_data]
    #     elif ext == ".mat":
    #         this_data = SessionData(path=fname)
    #         this_data.from_mat(str(fname), fields_to_load=fields_to_load)
    #         # load_mat(fname, fields_to_load=fields_to_load)
    #         return [this_data]
    #     else:
    #         raise ValueError(f"Unsupported file extension: {ext}. Use '.h5' or '.hdf5'.")
        
    # def load_sessions_from_hdf5(
    #     self, h5ref: h5py.Group | h5py.File
    # ) -> List[SessionData]:
    #     """
    #     Loads and returns session data from an HDF5 group
    #     """
    #     if h5ref.attrs.get("object_type") != "SessionData":
    #         raise ValueError(
    #             "The provided HDF5 group does not contain a SessionData object."
    #         )

    #     if h5ref.attrs.get("schema_version") != self.HDF5_VERSION:
    #         raise ValueError(
    #             f"Schema version mismatch: expected {self.HDF5_VERSION}, found {h5ref.attrs.get('schema_version')}"
    #         )

    #     ## define fields as found in saved hdf5 structure
    #     # self.load_configs.select("CATAN session")
    #     # assert self.load_configs.current is not None, "No load configuration found for 'CATAN session'."
    #     fields_to_load = LoadConfig.fields_from_resource("catan_session.json")

    #     n_sessions = h5ref.attrs["n_sessions"]
    #     assert isinstance(n_sessions, numbers.Integral), "Number of sessions should be an integer"

    #     sessions = []
    #     for s in range(n_sessions):
    #         session_group = h5ref[f"session_{s:03d}"]
    #         assert isinstance(
    #             session_group, h5py.Group
    #         ), f"Session group for session {s} is not a valid HDF5 group"
    #         session = SessionData()
    #         data = session.from_hdf5(session_group, fields_to_load)
    #         # print("fields_to_load:", fields_to_load)
    #         # print("\n\t data keys: ", data.keys())
    #         session.register_data(self.alignment_template,**data)
    #         sessions.append(session)
        
    #     return sessions

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
            output_fname = self.get_result_directory() / f"catan_model{fix_suffix(suffix)}{ext}"
        else:
            self.get_result_directory(Path(output_fname).parent)
        # output_directory = self.get_result_directory(Path(output_fname).parent)
        # fname = output_directory / f"catan_model{fix_suffix(suffix)}{ext}"

        self.model.save(str(output_fname))

    def save_assignments(
        self,
        output_fname: Optional[str | Path] = None,
        suffix: str = "",
        ext: str = ".hdf5",
    ):
        if self.assignments is None:
            raise ValueError("No assignments to save. Please run the registration before saving.")

        if output_fname is None:
            output_fname = self.get_result_directory() / f"catan_registration{fix_suffix(suffix)}{ext}"
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

