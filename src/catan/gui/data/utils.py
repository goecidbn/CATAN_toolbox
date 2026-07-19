import numpy as np


def move_index_along_axis(array: np.ndarray, axis: int, old: int, new: int):
    order = list(range(array.shape[axis]))
    value = order.pop(old)
    order.insert(new, value)
    return np.take(array, order, axis=axis)
