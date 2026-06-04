def bitmask_to_3x3(bitmask: int, *, dtype=np.uint8) -> np.ndarray:
    """
    Преобразует 8-битную маску окрестности в матрицу 3x3.

    Нумерация битов:

        [3 2 1]
        [4 x 0]
        [5 6 7]

    Центр всегда равен 1.
    """
    bitmask = int(bitmask)

    out = np.zeros((3, 3), dtype=dtype)

    # center
    out[1, 1] = 1

    bit_positions = {
        0: (1, 2),  # right
        1: (0, 2),  # up-right
        2: (0, 1),  # up
        3: (0, 0),  # up-left
        4: (1, 0),  # left
        5: (2, 0),  # down-left
        6: (2, 1),  # down
        7: (2, 2),  # down-right
    }

    for bit_id, pos in bit_positions.items():
        out[pos] = 1 if (bitmask & (1 << bit_id)) else 0
    return out