from tqdm.auto import tqdm
import numpy as np
from scipy import sparse, spatial

from catan.core.structures.session import SessionData
from catan.core.alignment import calculate_img_correlation

from typing import Optional, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed

# from threading import Lock


def calculate_statistics(
    this_data: SessionData,
    reference_data: SessionData,
    distance_threshold: float = 25.0,
    **kwargs,
    # mode: str = "cross",
):
    """
    function to calculate footprint correlations used for the matching procedure for
      - nearest neighbour (NN) - closest center of mass to reference footprint
      - non-nearest neighbour (nNN) - other neurons with center of mass distance below threshold

    footprint correlations are calculated as shifted and non-shifted (depending on model used)

    this method is used both for comparing statistics
      - referencing previous session (used for matching)
      - referencing itself (used for removal of duplicated footprints and kernel calculation)

    """

    if this_data.A is None or reference_data.A is None:
        raise ValueError("Both this_data and reference_data must have A attribute set.")

    if (
        this_data.A.shape == reference_data.A.shape
        and len(this_data.A.indices) == len(reference_data.A.indices)
        and np.all(this_data.A.indices == reference_data.A.indices)
        and np.all(this_data.A.indptr == reference_data.A.indptr)
        and np.allclose(this_data.A.data, reference_data.A.data)
    ):
        mode = "same"
    else:
        mode = "cross"
    # print(
    #     f"Calculating statistics for {mode}-matching of {this_data.n_neurons} vs {reference_data.n_neurons} neurons"
    # )
    # calculate distance between footprints and identify NN
    centroid_distance = spatial.distance.cdist(
        reference_data.centroids, this_data.centroids
    )

    def process_neuron(i):
        """Process a single reference neuron against all nearby neurons in this_data"""
        local_removals = []
        local_remove_info = []
        local_distances = centroid_distance[i, :].copy()
        local_correlations = np.full(this_data.n_neurons, np.nan)
        local_shifts = np.full((this_data.n_neurons, 2), np.nan)

        if not reference_data.idx_eval[i] or i in idx_remove:
            return (
                i,
                local_shifts,
                local_distances,
                local_correlations,
                local_removals,
                local_remove_info,
            )

        for j in np.where(local_distances < distance_threshold)[0]:
            if not this_data.idx_eval[j] or j in idx_remove:
                continue

            A_ref = reference_data.A[:, i]  # .toarray()

            ## calculate pairwise correlation between reference and current set of neuron footprints
            local_correlations[j], _, local_shifts[j, :] = calculate_img_correlation(
                this_data.A[:, j],  # .toarray(),
                A_ref,
                crop=True,
                shift=True,  # (key == "shifted"),
                # binary=params.get("binary", False),
                mode="cosine_union",
                gamma=0.1 if mode == "cross" else 0.9,
                shift_optimized=mode == "cross",
            )
            if mode == "cross":
                local_distances[j] = np.sqrt(np.sum([s**2 for s in local_shifts[j, :]]))

            if (
                (mode != "same")
                or (i == j)
                or (this_data.trace is None)
                or (not isinstance(this_data.quality, dict))
                or (this_data.quality.get("SNR_comp", None) is None)
            ):
                continue

            remove_id, remove_info = check_removal(
                # footprint_correlation["shifted"][(i, j)],
                local_correlations[j],
                this_data.trace,
                this_data.quality["SNR_comp"],
                i,
                j,
            )
            if remove_id is not None:
                local_removals.append(remove_id)
                local_remove_info.append(remove_info)

        return (
            i,
            local_shifts,
            local_distances,
            local_correlations,
            local_removals,
            local_remove_info,
        )

    ## prepare arrays to hold statistics
    centroid_shift = np.full((reference_data.n_neurons, this_data.n_neurons, 2), np.nan)
    footprint_correlation = np.full(
        (reference_data.n_neurons, this_data.n_neurons), np.nan
    )
    idx_remove = []  # only gets populated, when 'same' is True
    remove_info = {}  # only gets populated, when 'same' is True

    # Execute in parallel with progress bar
    max_workers = kwargs.get("nP", 12)
    if max_workers > 1:

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {
                executor.submit(process_neuron, i): i
                for i in range(reference_data.n_neurons)
            }

            for future in tqdm(
                as_completed(futures),
                total=reference_data.n_neurons,
                desc="calculating footprint correlation of %d neurons"
                % reference_data.idx_eval.sum(),
                leave=False,
            ):
                (
                    i,
                    local_shifts,
                    local_distances,
                    local_correlations,
                    local_removals,
                    local_remove_info,
                ) = future.result()

                # Merge results into global arrays
                centroid_shift[i, :] = local_shifts
                centroid_distance[i, :] = local_distances
                footprint_correlation[i, :] = local_correlations
                idx_remove.extend(local_removals)

                for j, id in enumerate(local_removals):
                    remove_info[id] = local_remove_info[j]
    else:
        for i in tqdm(
            range(reference_data.n_neurons),
            desc="calculating footprint correlation of %d neurons"
            % reference_data.idx_eval.sum(),
            leave=False,
        ):
            (
                _,
                local_shifts,
                local_distances,
                local_correlations,
                local_removals,
                local_remove_info,
            ) = process_neuron(i)

            centroid_shift[i, ...] = local_shifts
            centroid_distance[i, :] = local_distances
            footprint_correlation[i, :] = local_correlations

            idx_remove.extend(local_removals)

            for j, id in enumerate(local_removals):
                remove_info[id] = local_remove_info[j]
    # Remove duplicates from idx_remove
    idx_remove = list(set(idx_remove))

    return (
        centroid_shift,
        centroid_distance,
        footprint_correlation,
        idx_remove,
        remove_info,
    )


def check_removal(
    fp_corr: float, trace: np.ndarray, SNR_comp: np.ndarray, i: int, j: int
) -> Tuple[Optional[int], Tuple]:

    thr_fp_corr = [0.3, 0.7]
    thr_trace_corr = [0.3, 0.7]
    # tag footprints for removal when calculating statistics for self-matching and they pass some criteria:
    # 1) significant overlap with closeby neuron ("contestant")
    if np.isnan(fp_corr) or fp_corr < thr_fp_corr[0]:
        return None, ()

    # 2) high correlation of neuronal activity ("trace") (or very high footprint correlation!)
    trace_corr = np.corrcoef(trace[i, :], trace[j, :])[0, 1]
    if trace_corr > thr_trace_corr[1] or (
        fp_corr > thr_fp_corr[1] and trace_corr > thr_trace_corr[0]
    ):

        # 3) lower SNR than contestant
        return int(j) if SNR_comp[i] > SNR_comp[j] else int(i), (
            (int(i), int(j)),
            fp_corr,
            trace_corr,
            (SNR_comp[i], SNR_comp[j]),
        )
        # print('removing neuron %d (%d vs %d) from data (Acorr: %.3f, Ccorr: %.3f; SNR: %.2f vs %.2f)'%(idx_remove[-1],i,j,footprint_correlation[1,i,j],C_corr,SNR_comp[i],SNR_comp[j]))

    return None, ()


def calculate_p(d_ROIs, fp_corr, p_model, neighbor_distance=12):
    """
    evaluates the probability of neuron footprints belonging to the same neuron. It uses an interpolated version of p_same

    This function requires the successful building of a matching model first
    """

    ## evaluate probability-function for each of a neurons neighbors
    # print(neighbor_distance)
    neighbors = d_ROIs < neighbor_distance
    p_same = np.zeros_like(d_ROIs)
    p_same[neighbors] = p_model(d_ROIs[neighbors], fp_corr[neighbors])

    p_same[np.isnan(p_same)] = 0  # fill "bad" entries with zeros
    p_same = np.clip(
        p_same, 0, 1
    )  # function-shapes may allow for values exceeding [0,1]

    return sparse.csc_matrix(p_same)
