import numpy as np
from numba import njit
from .raveled_offets import raveled_offsets_c8
from scipy import sparse
from .simplify_links import SIMPLE_MASK


def neighborhood_pixel_graph(image, simplify=True):
    import time

    t = time.perf_counter()
    nodes, bit_neighborhood = _build_bit_neighborhood(image)
    if simplify:
        bit_neighborhood = SIMPLE_MASK[bit_neighborhood]

    neighbor_offsets = raveled_offsets_c8(image.shape)
    point_start, points_end = get_links(nodes, bit_neighborhood, neighbor_offsets)

    data = np.ones(point_start.size, dtype=np.uint8)

    graph = sparse.coo_matrix(
        (data, (point_start, points_end)),
        shape=(nodes.size, nodes.size)
    ).tocsr()
    assert_symmetric_graph(graph)
    coordinates = np.column_stack(np.unravel_index(nodes, image.shape))


    return graph, coordinates


def assert_symmetric_graph(graph):
    """
    Проверяет, что sparse-граф симметричен:
        edge i -> j существует тогда и только тогда, когда edge j -> i существует.
    """
    g = graph.astype(bool).tocsr()
    diff = g != g.T

    if diff.nnz != 0:
        rows, cols = diff.nonzero()
        a = rows[0]
        b = cols[0]

        raise ValueError(
            f"Graph is not symmetric: mismatch at ({a}, {b}). "
            f"One of edges {a}->{b} or {b}->{a} is missing. "
            f"Total asymmetric entries: {diff.nnz}"
        )


@njit(inline="always")
def popcount_u8(x):
    c = 0
    for i in range(8):
        c += (x >> i) & 1
    return c


@njit(cache=True)
def get_links(nodes, bit_neighborhood, neighbor_offsets):
    total_links = 0

    # Первый проход: считаем количество рёбер
    for b in bit_neighborhood:
        total_links += popcount_u8(b)

    point_start = np.empty(total_links, dtype=np.uint32)
    points_end = np.empty(total_links, dtype=np.uint32)

    link_num = 0

    # Второй проход: заполняем рёбра
    for p_num in range(nodes.shape[0]):
        p = nodes[p_num]
        b = bit_neighborhood[p_num]

        for i in range(8):
            if (b >> i) & 1:
                q = p + neighbor_offsets[i]
                q_num = np.searchsorted(nodes, q)
                point_start[link_num] = p_num
                points_end[link_num] = q_num
                link_num += 1

    return point_start, points_end


def _build_bit_neighborhood(mask: np.ndarray):
    mask = mask.astype(bool, copy=False)
    h, w = mask.shape
    padded = np.pad(mask, 1, mode="constant", constant_values=False)
    padded = padded.ravel()
    nodes = np.flatnonzero(mask)
    nodes = nodes.astype(np.int64, copy=False)
    w = np.int64(w)
    nodes_padded = nodes + np.int64(2) * (nodes // w) + (w + np.int64(3))
    bitmask = get_neighborhood(nodes_padded, padded, w + np.int64(2))
    return nodes, bitmask


@njit(cache=True)
def get_neighborhood(nodes, image, w):
    n = nodes.size
    bit_neighborhood = np.empty(n, dtype=np.uint8)

    for i in range(n):
        node = nodes[i]

        bitmask = np.uint8(0)

        bitmask |= np.uint8(image[node + 1]) << 0
        bitmask |= np.uint8(image[node - w + 1]) << 1
        bitmask |= np.uint8(image[node - w]) << 2
        bitmask |= np.uint8(image[node - w - 1]) << 3
        bitmask |= np.uint8(image[node - 1]) << 4
        bitmask |= np.uint8(image[node + w - 1]) << 5
        bitmask |= np.uint8(image[node + w]) << 6
        bitmask |= np.uint8(image[node + w + 1]) << 7

        bit_neighborhood[i] = bitmask

    return bit_neighborhood