import numpy as np
from numba import njit
from .raveled_offets import raveled_offsets_c8
from scipy import sparse
from .simplify_links import SIMPLE_MASK


def neighborhood_pixel_graph(image, simplify=True):
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


@njit
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
    """
        Нумерация соседей:

        [3 2 1]
        [4 x 0]
        [5 6 7]
    """
    mask = mask.astype(bool)
    padded = np.pad(mask, 1, mode='constant', constant_values=False)

    views = _build_views3x3(mask)
    nodes = np.flatnonzero(mask)
    bit_neighborhood = np.zeros_like(nodes, dtype=np.uint8)
    bit_offsets = [
        (0, 1),  # bit 0: right
        (-1, 1),  # bit 1: up-right
        (-1, 0),  # bit 2: up
        (-1, -1),  # bit 3: up-left
        (0, -1),  # bit 4: left
        (1, -1),  # bit 5: down-left
        (1, 0),  # bit 6: down
        (1, 1),  # bit 7: down-right
    ]
    for bit_id, offset in enumerate(bit_offsets):
        neighbor_view = views[offset]
        neighbor_values = neighbor_view.ravel()[nodes]
        bit_neighborhood |= neighbor_values.astype(np.uint8) << bit_id

    return nodes, bit_neighborhood


def _build_views3x3(image):
    offsets_3x3 = [(dy, dx) for dy in range(-1, 2) for dx in range(-1, 2)]
    view = {}
    padded = np.pad(image, 1, mode='constant', constant_values=False)
    h, w = image.shape
    for dy, dx in offsets_3x3:
        view[(dy, dx)] = padded[
            1 + dy : 1 + dy + h,
            1 + dx : 1 + dx + w
        ]
    return view