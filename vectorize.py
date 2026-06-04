import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
from rasterio.rio.helpers import coords
from skan import Skeleton
from .pixel_graph import neighborhood_pixel_graph

def vectorize(img):
    """
    Возвращает только список ломаных линий.

    Формат:
        lines = [
            np.array([[x1, y1], [x2, y2], ...]),
            np.array([[x1, y1], [x2, y2], ...]),
            ...
        ]

    Здесь:
        x = col
        y = row

    Ломаная строится от одного важного узла до другого.
    Важный узел — это:
        - конец линии, degree == 1
        - развилка, degree >= 3
        - изолированная точка, degree == 0

    Узлы с degree == 2 считаются промежуточными точками ломаной.
    """

    image_thin = img > 0

    if not np.any(image_thin):
        return []
    graph, coords = neighborhood_pixel_graph(image_thin.astype(bool))
    graph = graph.tocsr()

    node_count = graph.shape[0]

    if node_count == 0:
        return []

    degrees = np.asarray(graph.astype(bool).sum(axis=1)).ravel()

    # adjacency[i] = список соседей узла i
    adjacency = []
    for i in range(node_count):
        start = graph.indptr[i]
        end = graph.indptr[i + 1]
        adjacency.append(graph.indices[start:end].tolist())

    # Важные узлы: концы, развилки, изолированные точки
    important_nodes = set(np.where(degrees != 2)[0].tolist())

    visited_edges = set()
    lines = []

    def edge_key(a, b):
        return tuple(sorted((int(a), int(b))))

    def nodes_to_xy_line(path_nodes):
        """
        Преобразует индексы узлов Skeleton в np.array([[x, y], ...]).
        sk.coordinates хранит [row, col], а для matplotlib нужно [x, y] = [col, row].
        """
        line = []

        for node_idx in path_nodes:
            row, col = coords[node_idx]
            line.append([col, row])

        return np.asarray(line, dtype=float)

    # Случай изолированных точек
    for node in important_nodes:
        if degrees[node] == 0:
            lines.append(nodes_to_xy_line([node]))

    # Обычный случай: идём от важного узла до следующего важного узла
    for start_node in important_nodes:
        for next_node in adjacency[start_node]:
            key = edge_key(start_node, next_node)

            if key in visited_edges:
                continue

            path = [start_node]

            prev_node = start_node
            current_node = next_node

            visited_edges.add(key)

            while True:
                path.append(current_node)

                # Дошли до конца или развилки
                if current_node in important_nodes:
                    break

                neighbours = adjacency[current_node]

                # Ищем следующий узел, кроме того, откуда пришли
                candidates = [
                    node for node in neighbours
                    if node != prev_node
                ]

                if not candidates:
                    break

                # Для degree == 2 тут должен быть ровно один кандидат
                new_node = candidates[0]
                key = edge_key(current_node, new_node)

                if key in visited_edges:
                    break

                visited_edges.add(key)

                prev_node = current_node
                current_node = new_node

            lines.append(nodes_to_xy_line(path))

    # Отдельный случай: замкнутый цикл, где у всех узлов degree == 2
    # Например, кольцо без концов и развилок.
    if not important_nodes:
        for start_node in range(node_count):
            for next_node in adjacency[start_node]:
                key = edge_key(start_node, next_node)

                if key in visited_edges:
                    continue

                path = [start_node]

                prev_node = start_node
                current_node = next_node

                visited_edges.add(key)

                while True:
                    path.append(current_node)

                    if current_node == start_node:
                        break

                    neighbours = adjacency[current_node]

                    candidates = [
                        node for node in neighbours
                        if node != prev_node
                    ]

                    if not candidates:
                        break

                    new_node = candidates[0]
                    key = edge_key(current_node, new_node)

                    if key in visited_edges:
                        break

                    visited_edges.add(key)

                    prev_node = current_node
                    current_node = new_node

                lines.append(nodes_to_xy_line(path))

    return lines

