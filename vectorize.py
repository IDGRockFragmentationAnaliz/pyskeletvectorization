import numpy as np
from numba import njit
from .pixel_graph import pixel_graph
from .linesimplification import douglas_peucker_inplace


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

    degrees = np.diff(indptr)
    important_mask = degrees != 2

    points_xy, offsets = _trace_paths_numba(
        indptr,
        indices,
        degrees,
        important_mask,
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
def _append_node_point(points_xy, flat_count, coords, node):
    """
    coords хранит [row, col].
    points_xy хранит [x, y] = [col, row].
    """
    points_xy[flat_count, 0] = coords[node, 1]
    points_xy[flat_count, 1] = coords[node, 0]

    return flat_count + 1


@njit(cache=True)
def _finish_line(
    points_xy,
    flat_count,
    offsets,
    line_count,
    line_start,
    simplify_tolerance,
):
    """
    Завершает текущую линию.

    Если simplify_tolerance > 0:
        упрощает points_xy[line_start:flat_count] на месте
        и откатывает flat_count к новой длине.

    Если simplify_tolerance <= 0:
        ничего не упрощает.
    """

    if simplify_tolerance > 0.0:
        line = points_xy[line_start:flat_count]

        new_len = douglas_peucker_inplace(
            line,
            simplify_tolerance,
        )

        flat_count = line_start + new_len

    line_count += 1
    offsets[line_count] = flat_count

    return flat_count, line_count


@njit(cache=True)
def _trace_paths_numba(
    indptr,
    indices,
    degrees,
    important_mask,
    coords,
    simplify_tolerance,
):
    """
    Возвращает:
        points_xy: плоский массив координат всех линий
        offsets: границы линий в points_xy

    Линия i:
        points_xy[offsets[i]:offsets[i + 1]]
    """

    node_count = degrees.shape[0]
    edge_ref_count = indices.shape[0]

    visited = np.zeros(edge_ref_count, dtype=np.uint8)

    # В CSR неориентированное ребро обычно хранится дважды.
    # edge_ref_count ~= 2 * edge_count.
    #
    # Суммарное число точек во всех ломаных <= edge_count + line_count.
    # Безопасный запас: edge_ref_count + node_count + 1.
    max_flat_points = edge_ref_count + node_count + 1
    max_lines = edge_ref_count + node_count + 1

    points_xy = np.empty((max_flat_points, 2), dtype=coords.dtype)
    offsets = np.empty(max_lines + 1, dtype=np.int64)

    flat_count = 0
    line_count = 0
    offsets[0] = 0

    # 1. Изолированные точки
    for node in range(node_count):
        if important_mask[node] and degrees[node] == 0:
            line_start = flat_count

            flat_count = _append_node_point(
                points_xy,
                flat_count,
                coords,
                node,
            )

            flat_count, line_count = _finish_line(
                points_xy,
                flat_count,
                offsets,
                line_count,
                line_start,
                simplify_tolerance,
            )

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

            line_start = flat_count

            flat_count = _append_node_point(
                points_xy,
                flat_count,
                coords,
                start_node,
            )

            prev_node = start_node
            current_node = next_node

            visited[p] = 1
            _mark_reverse_edge(
                indptr,
                indices,
                visited,
                start_node,
                current_node,
            )

            while True:
                flat_count = _append_node_point(
                    points_xy,
                    flat_count,
                    coords,
                    current_node,
                )

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
                _mark_reverse_edge(
                    indptr,
                    indices,
                    visited,
                    current_node,
                    new_node,
                )

                prev_node = current_node
                current_node = new_node

            flat_count, line_count = _finish_line(
                points_xy,
                flat_count,
                offsets,
                line_count,
                line_start,
                simplify_tolerance,
            )

    # 3. Остаточные циклы.
    # Важно: это ловит кольца даже если в изображении есть и обычные линии тоже.
    for start_node in range(node_count):
        nb_start = indptr[start_node]
        nb_end = indptr[start_node + 1]

        for p in range(nb_start, nb_end):
            if visited[p] != 0:
                continue

            next_node = indices[p]

            line_start = flat_count

            flat_count = _append_node_point(
                points_xy,
                flat_count,
                coords,
                start_node,
            )

            prev_node = start_node
            current_node = next_node

            visited[p] = 1
            _mark_reverse_edge(
                indptr,
                indices,
                visited,
                start_node,
                current_node,
            )

            while True:
                flat_count = _append_node_point(
                    points_xy,
                    flat_count,
                    coords,
                    current_node,
                )

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
                _mark_reverse_edge(
                    indptr,
                    indices,
                    visited,
                    current_node,
                    new_node,
                )

                prev_node = current_node
                current_node = new_node

            flat_count, line_count = _finish_line(
                points_xy,
                flat_count,
                offsets,
                line_count,
                line_start,
                simplify_tolerance,
            )

    return points_xy[:flat_count].copy(), offsets[:line_count + 1].copy()