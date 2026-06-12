import numpy as np
from numba import njit
from .pixel_graph import pixel_graph
from .trace_csrgraph_paths import _trace_paths


def vectorize(img, return_flat=False, simplify_tolerance=1.0):
    """
    Возвращает список ломаных линий:
        [np.array([[x, y], ...]), ...]

    Если return_flat=True, возвращает:
        points_xy, offsets

    где линия i:
        points_xy[offsets[i]:offsets[i + 1]]

    simplify_tolerance:
        0.0 — без упрощения.
        > 0.0 — Douglas–Peucker на лету после построения каждой линии.
    """

    image_thin = img.astype(bool, copy=True)

    if not image_thin.any():
        if return_flat:
            return (
                np.empty((0, 2), dtype=np.float64),
                np.array([0], dtype=np.int64),
            )

        return []

    graph, coords = pixel_graph(image_thin.astype(bool))
    graph = graph.tocsr()
    node_count = graph.shape[0]

    if node_count == 0:
        if return_flat:
            return (
                np.empty((0, 2), dtype=np.float64),
                np.array([0], dtype=np.int64),
            )

        return []

    indptr = graph.indptr
    indices = graph.indices

    points_xy, offsets = _trace_paths(
        indices,
        indptr,
        coords,
        float(simplify_tolerance),
    )

    if return_flat:
        return points_xy, offsets

    lines = [
        points_xy[offsets[i]:offsets[i + 1]]
        for i in range(offsets.size - 1)
    ]

    return lines