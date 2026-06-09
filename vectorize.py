import numpy as np
from numba import njit
from .pixel_graph import neighborhood_pixel_graph


def vectorize(img, return_flat=False):
    """
    Возвращает список ломаных линий:
        [np.array([[x, y], ...]), ...]

    Если return_flat=True, возвращает:
        points_xy, offsets

    где линия i:
        points_xy[offsets[i]:offsets[i + 1]]
    """

    image_thin = img > 0

    if not image_thin.any():
        if return_flat:
            return (
                np.empty((0, 2), dtype=np.float64),
                np.array([0], dtype=np.int64),
            )

        return []

    import time
    start_time = time.perf_counter()
    graph, coords = neighborhood_pixel_graph(image_thin.astype(bool))
    elapsed_time = time.perf_counter() - start_time
    graph = graph.tocsr()
    print("тест основного алгоритма", elapsed_time)

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

    degrees = np.diff(indptr)

    important_mask = degrees != 2

    flat_nodes, offsets = _trace_paths_numba(
        indptr,
        indices,
        degrees,
        important_mask,
    )

    # coords хранит [row, col]
    # Нужно [x, y] = [col, row]
    rc = coords[flat_nodes]

    # Если float не нужен, лучше оставить int:
    # points_xy = rc[:, ::-1].copy()
    points_xy = rc[:, ::-1].astype(np.float64, copy=True)

    if return_flat:
        return points_xy, offsets

    lines = [
        points_xy[offsets[i]:offsets[i + 1]]
        for i in range(offsets.size - 1)
    ]

    return lines


@njit(cache=True)
def _mark_reverse_edge(indptr, indices, visited, a, b):
    """
    Отмечает обратное ребро b -> a.

    Для пиксельного графа степень малая, обычно <= 8,
    поэтому линейный поиск по соседям дешёвый.
    """
    start = indptr[b]
    end = indptr[b + 1]

    for q in range(start, end):
        if indices[q] == a:
            visited[q] = 1
            return


@njit(cache=True)
def _trace_paths_numba(indptr, indices, degrees, important_mask):
    """
    Возвращает:
        flat_nodes: плоский массив индексов узлов
        offsets: границы линий в flat_nodes

    Линия i:
        flat_nodes[offsets[i]:offsets[i + 1]]
    """

    node_count = degrees.shape[0]
    edge_ref_count = indices.shape[0]

    visited = np.zeros(edge_ref_count, dtype=np.uint8)

    # В CSR неориентированное ребро обычно хранится дважды.
    # edge_ref_count ~= 2 * edge_count.
    #
    # Суммарное число узлов во всех ломаных <= edge_count + line_count.
    # Безопасный запас: edge_ref_count + node_count + 1.
    max_flat_nodes = edge_ref_count + node_count + 1
    max_lines = edge_ref_count + node_count + 1

    flat_nodes = np.empty(max_flat_nodes, dtype=np.int64)
    offsets = np.empty(max_lines + 1, dtype=np.int64)

    flat_count = 0
    line_count = 0
    offsets[0] = 0

    # 1. Изолированные точки
    for node in range(node_count):
        if important_mask[node] and degrees[node] == 0:
            flat_nodes[flat_count] = node
            flat_count += 1

            line_count += 1
            offsets[line_count] = flat_count

    # 2. Обычные линии: от важного узла до важного узла
    for start_node in range(node_count):
        if not important_mask[start_node]:
            continue

        nb_start = indptr[start_node]
        nb_end = indptr[start_node + 1]

        for p in range(nb_start, nb_end):
            if visited[p] != 0:
                continue

            next_node = indices[p]

            flat_nodes[flat_count] = start_node
            flat_count += 1

            prev_node = start_node
            current_node = next_node

            visited[p] = 1
            _mark_reverse_edge(indptr, indices, visited, start_node, current_node)

            while True:
                flat_nodes[flat_count] = current_node
                flat_count += 1

                if important_mask[current_node]:
                    break

                # current_node не важный, значит degree == 2
                p0 = indptr[current_node]
                p1 = p0 + 1

                if indices[p0] != prev_node:
                    p_next = p0
                else:
                    p_next = p1

                if visited[p_next] != 0:
                    break

                new_node = indices[p_next]

                visited[p_next] = 1
                _mark_reverse_edge(indptr, indices, visited, current_node, new_node)

                prev_node = current_node
                current_node = new_node

            line_count += 1
            offsets[line_count] = flat_count

    # 3. Остаточные циклы.
    # Важно: это ловит кольца даже если в изображении есть и обычные линии тоже.
    for start_node in range(node_count):
        nb_start = indptr[start_node]
        nb_end = indptr[start_node + 1]

        for p in range(nb_start, nb_end):
            if visited[p] != 0:
                continue

            next_node = indices[p]

            flat_nodes[flat_count] = start_node
            flat_count += 1

            prev_node = start_node
            current_node = next_node

            visited[p] = 1
            _mark_reverse_edge(indptr, indices, visited, start_node, current_node)

            while True:
                flat_nodes[flat_count] = current_node
                flat_count += 1

                if current_node == start_node:
                    break

                if degrees[current_node] != 2:
                    break

                p0 = indptr[current_node]
                p1 = p0 + 1

                if indices[p0] != prev_node:
                    p_next = p0
                else:
                    p_next = p1

                if visited[p_next] != 0:
                    break

                new_node = indices[p_next]

                visited[p_next] = 1
                _mark_reverse_edge(indptr, indices, visited, current_node, new_node)

                prev_node = current_node
                current_node = new_node

            line_count += 1
            offsets[line_count] = flat_count

    return flat_nodes[:flat_count], offsets[:line_count + 1]

