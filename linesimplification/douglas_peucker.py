import numpy as np
from numba import njit


@njit(cache=True)
def simplify_line_douglas_peucker(points_xy, tolerance):
    """
    Упрощает одну ломаную Douglas–Peucker.

    points_xy:
        np.ndarray shape=(N, 2)

    tolerance:
        допуск в тех же единицах, что и координаты.
        Для пиксельных координат — в пикселях.

    Возвращает:
        np.ndarray shape=(M, 2)
    """

    n = points_xy.shape[0]

    if n <= 2:
        return points_xy.copy()

    if tolerance <= 0.0:
        return points_xy.copy()

    tolerance_sq = tolerance * tolerance

    keep = np.zeros(n, dtype=np.uint8)

    stack_left = np.empty(n, dtype=np.int64)
    stack_right = np.empty(n, dtype=np.int64)

    is_closed = (
        n > 3
        and points_xy[0, 0] == points_xy[n - 1, 0]
        and points_xy[0, 1] == points_xy[n - 1, 1]
    )

    if is_closed:
        # Если линия замкнута, первая и последняя точки совпадают.
        # Нельзя просто запускать DP от первой до последней:
        # получится вырожденный отрезок нулевой длины.
        #
        # Поэтому ищем точку, максимально удалённую от стартовой,
        # и разбиваем кольцо на две обычные открытые ломаные:
        # 0 -> split и split -> n - 1.

        sx = points_xy[0, 0]
        sy = points_xy[0, 1]

        max_dist_sq = -1.0
        split_idx = 1

        for i in range(1, n - 1):
            dx = float(points_xy[i, 0]) - float(sx)
            dy = float(points_xy[i, 1]) - float(sy)
            dist_sq = dx * dx + dy * dy

            if dist_sq > max_dist_sq:
                max_dist_sq = dist_sq
                split_idx = i

        if split_idx > 0:
            _dp_mark_range(
                points_xy,
                keep,
                0,
                split_idx,
                tolerance_sq,
                stack_left,
                stack_right,
            )

            _dp_mark_range(
                points_xy,
                keep,
                split_idx,
                n - 1,
                tolerance_sq,
                stack_left,
                stack_right,
            )

        keep[0] = 1
        keep[n - 1] = 1

    else:
        _dp_mark_range(
            points_xy,
            keep,
            0,
            n - 1,
            tolerance_sq,
            stack_left,
            stack_right,
        )

    out_count = 0

    for i in range(n):
        if keep[i] != 0:
            out_count += 1

    simplified = np.empty((out_count, 2), dtype=points_xy.dtype)

    pos = 0

    for i in range(n):
        if keep[i] != 0:
            simplified[pos, 0] = points_xy[i, 0]
            simplified[pos, 1] = points_xy[i, 1]
            pos += 1

    return simplified

@njit(cache=True)
def _point_segment_distance_sq(px, py, ax, ay, bx, by):
    ax = float(ax)
    ay = float(ay)
    bx = float(bx)
    by = float(by)
    px = float(px)
    py = float(py)

    dx = bx - ax
    dy = by - ay

    denom = dx * dx + dy * dy

    if denom == 0.0:
        ddx = px - ax
        ddy = py - ay
        return ddx * ddx + ddy * ddy

    t = ((px - ax) * dx + (py - ay) * dy) / denom

    if t <= 0.0:
        ddx = px - ax
        ddy = py - ay
        return ddx * ddx + ddy * ddy

    if t >= 1.0:
        ddx = px - bx
        ddy = py - by
        return ddx * ddx + ddy * ddy

    proj_x = ax + t * dx
    proj_y = ay + t * dy

    ddx = px - proj_x
    ddy = py - proj_y

    return ddx * ddx + ddy * ddy


@njit(cache=True)
def _dp_mark_range(points_xy, keep, first, last, tolerance_sq, stack_left, stack_right):
    keep[first] = 1
    keep[last] = 1

    stack_size = 0

    stack_left[stack_size] = first
    stack_right[stack_size] = last
    stack_size += 1

    while stack_size > 0:
        stack_size -= 1

        left = stack_left[stack_size]
        right = stack_right[stack_size]

        if right <= left + 1:
            continue

        ax = points_xy[left, 0]
        ay = points_xy[left, 1]
        bx = points_xy[right, 0]
        by = points_xy[right, 1]

        max_dist_sq = -1.0
        max_idx = -1

        for i in range(left + 1, right):
            px = points_xy[i, 0]
            py = points_xy[i, 1]

            dist_sq = _point_segment_distance_sq(
                px, py,
                ax, ay,
                bx, by,
            )

            if dist_sq > max_dist_sq:
                max_dist_sq = dist_sq
                max_idx = i

        if max_dist_sq > tolerance_sq:
            keep[max_idx] = 1

            stack_left[stack_size] = left
            stack_right[stack_size] = max_idx
            stack_size += 1

            stack_left[stack_size] = max_idx
            stack_right[stack_size] = right
            stack_size += 1