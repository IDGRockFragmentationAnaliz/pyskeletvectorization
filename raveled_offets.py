import numpy as np
from numba import njit


@njit
def raveled_offsets_c8(image_shape):
    """
    Возвращает raveled-смещения соседей вследующем плрядке:

        [3 2 1]
        [4 x 0]
        [5 6 7]

    Для C-order 2D массива.

    Returns
    -------
    raveled_offsets : np.ndarray
        Смещения, которые надо прибавлять к flat-index центра.
    """

    if len(image_shape) != 2:
        raise ValueError("Expected 2D image_shape")
    h, w = image_shape
    offsets_yx = np.array([
        [ 0,  1],   # bit 0: right
        [-1,  1],   # bit 1: up-right
        [-1,  0],   # bit 2: up
        [-1, -1],   # bit 3: up-left
        [ 0, -1],   # bit 4: left
        [ 1, -1],   # bit 5: down-left
        [ 1,  0],   # bit 6: down
        [ 1,  1],   # bit 7: down-right
    ], dtype=np.int64)

    raveled_offsets = offsets_yx[:, 0] * w + offsets_yx[:, 1]

    return raveled_offsets