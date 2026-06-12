import numpy as np
from numba import njit
from .linesimplification import douglas_peucker_inplace

@njit(cache=True)
def _trace_paths(indices, indptr, coords, simplify_tolerance):
    """
    indices : 1D array, shape (edge_ref_count,)
        CSR-массив соседей. Для неориентированного графа каждое ребро хранится дважды

    indptr : 1D array, shape (node_count + 1,)
        CSR-указатели графа. Соседи i-го узла находятся в:
            indices[indptr[i]:indptr[i + 1]]

    :returns:
        points_xy: плоский массив координат всех линий
        offsets: границы линий в points_xy

    Линия i:
        points_xy[offsets[i]:offsets[i + 1]]
    """
    degrees = np.diff(indptr)
    important_mask = degrees != 2

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
    for start_node in range(node_count):
        if degrees[start_node] == 0:
            flat_count, line_count = _trace_isolated_point(
                start_node,
                degrees,
                points_xy,
                flat_count,
                offsets,
                line_count,
                coords,
                simplify_tolerance
            )

    # 2. Обычные линии: от важного узла до важного узла
    for start_node in range(node_count):
        if not important_mask[start_node]:
            continue

        flat_count, line_count = _trace_paths_from_important_node(
            start_node,
            indptr,
            indices,
            visited,
            degrees,
            points_xy,
            flat_count,
            offsets,
            line_count,
            coords,
            simplify_tolerance,
        )

    # 3. Остаточные циклы.
    # Важно: это ловит кольца даже если в изображении есть и обычные линии тоже.
    for start_node in range(node_count):
        if degrees[start_node] != 2:
            continue

        flat_count, line_count = _trace_residual_cycles_from_node(
            start_node,
            indptr,
            indices,
            degrees,
            visited,
            points_xy,
            flat_count,
            offsets,
            line_count,
            coords,
            simplify_tolerance,
        )
    return points_xy[:flat_count].copy(), offsets[:line_count + 1].copy()


@njit(cache=True)
def _trace_isolated_point(
    start_node,
    degrees,
    points_xy,
    flat_count,
    offsets,
    line_count,
    coords,
    simplify_tolerance,
):
    """
    Обрабатывает одну возможную изолированную точку.

    Знание функции:
        start_node считается изолированной точкой только если:
            degrees[start_node] == 0
    """

    line_start = flat_count

    flat_count = _append_node_point(
        points_xy,
        flat_count,
        coords,
        start_node,
    )

    line_count += 1
    offsets[line_count] = flat_count

    return flat_count, line_count


@njit(cache=True)
def _trace_paths_from_important_node(
    start_node,
    indptr,
    indices,
    visited,
    degrees,
    points_xy,
    flat_count,
    offsets,
    line_count,
    coords,
    simplify_tolerance,
):
    """
    Трассирует все ещё не посещённые линии, выходящие из важного узла.

    Предусловие:
        start_node — важный узел графа.
        То есть degree(start_node) != 2.

    Смысл:
        из start_node запускаются трассы по каждому непосещённому ребру.
        Каждая трасса идёт через degree == 2 узлы, пока не встретит
        другой важный узел.
    """

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

        flat_count = _trace_degree2_chain_body(
            prev_node,
            current_node,
            indptr,
            indices,
            visited,
            degrees,
            points_xy,
            flat_count,
            coords,
        )

        line_count += 1
        offsets[line_count] = flat_count

        flat_count = _simplify_finished_line_inplace(
            points_xy,
            offsets,
            line_count,
            simplify_tolerance,
        )

    return flat_count, line_count


@njit(cache=True)
def _trace_residual_cycles_from_node(
    start_node,
    indptr,
    indices,
    degrees,
    visited,
    points_xy,
    flat_count,
    offsets,
    line_count,
    coords,
    simplify_tolerance,
):
    """
    Трассирует остаточные циклы, которые не были обработаны через важные узлы.

    Предусловие:
        start_node — узел остаточного цикла.
        Обычно degree(start_node) == 2.

    Смысл:
        после обработки всех важных узлов непосещёнными остаются только
        замкнутые компоненты, где нет концов и развилок.

    Функция проходит по всем непосещённым рёбрам start_node.
    Обычно в чистом цикле реально построится одна линия, а второе ребро
    уже окажется посещённым.
    """

    nb_start = indptr[start_node]
    nb_end = indptr[start_node + 1]
    visited[start_node] = 1

    for p in range(nb_start, nb_end):
        if visited[p] != 0:
            continue

        # начало обработки по координатам
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

        # начало цикла
        flat_count = _trace_degree2_chain_body(
            prev_node,
            current_node,
            indptr,
            indices,
            degrees,
            visited,
            points_xy,
            flat_count,
            coords,
        )
        # конец цикла

        line_count += 1
        offsets[line_count] = flat_count
        # Конец обработки по индексам

        flat_count = _simplify_finished_line_inplace(
            points_xy,
            offsets,
            line_count,
            simplify_tolerance,
        )

    return flat_count, line_count


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
def _simplify_finished_line_inplace(
    points_xy,
    offsets,
    line_count,
    simplify_tolerance,
):
    flat_count = offsets[line_count]

    if simplify_tolerance > 0.0:
        line_start = offsets[line_count - 1]
        line_end = offsets[line_count]

        line = points_xy[line_start:line_end]

        new_len = douglas_peucker_inplace(
            line,
            simplify_tolerance,
        )

        flat_count = line_start + new_len
        offsets[line_count] = flat_count

    return flat_count

@njit(cache=True)
def _trace_degree2_chain_body(
    prev_node,
    current_node,
    indptr,
    indices,
    degrees,
    visited,
    points_xy,
    flat_count,
    coords,
):
    """
    Трассирует цепочку через degree == 2 узлы.

    На каждой итерации:
        - добавляет current_node в points_xy;
        - если degree[current_node] != 2, завершает трассу;
        - выбирает следующее ребро, отличное от ребра назад;
        - если следующее ребро уже посещено, завершает трассу;
        - иначе помечает ребро и переходит дальше.

    Работает и для:
        - обычных линий между важными узлами;
        - остаточных циклов.
    """

    while True:
        flat_count = _append_node_point(
            points_xy,
            flat_count,
            coords,
            current_node,
        )

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

    return flat_count