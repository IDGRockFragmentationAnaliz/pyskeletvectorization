import numpy as np
from numba import njit


@njit(cache=True)
def raveled_offsets_c8(image_shape):
    h, w = image_shape
    return np.array([
        1,
        -w + 1,
        -w,
        -w - 1,
        -1,
        w - 1,
        w,
        w + 1,
    ], dtype=np.int64)