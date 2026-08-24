import numpy as np

def scale_down_counts(counts, times=1):
    """
    scales down the whole matrix "counts" by a factor of 2^times
    """

    if times == 0:
        return counts

    assert counts.shape[0] > 8, "No further scaling down allowed"

    if len(counts.shape) > 2:
        cts = np.zeros(tuple((d // 2 for d in counts.shape[:2])) + (counts.shape[2],))
        for d in range(counts.shape[2]):
            for i in range(2):
                for j in range(2):
                    cts[..., d] += counts[i::2, j::2, d]
    else:
        cts = np.zeros(tuple((d // 2 for d in counts.shape[:2])))  # + (3,))
        for i in range(2):
            for j in range(2):
                cts += counts[i::2, j::2]

    return scale_down_counts(cts, times - 1)