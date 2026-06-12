# pyskeletvectorization

```python
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from pyskeletvectorization import vectorize


def main():
    img = np.array([
        [0, 1, 0, 1, 0, 1, 0],
        [1, 1, 1, 0, 1, 1, 1],
        [0, 1, 0, 1, 0, 1, 0],
        [0, 1, 0, 0, 0, 1, 0],
        [0, 1, 0, 0, 0, 0, 0],
        [0, 0, 1, 1, 1, 1, 1],
        [0, 0, 0, 0, 0, 1, 0],
    ], dtype=np.uint8) * 255

    lines = vectorize(img, simplify_tolerance=1.0)

    fig, ax = plt.subplots(figsize=(4, 4))
    ax.imshow(img, cmap="gray_r", interpolation="nearest")

    prepared = []
    for line in lines:
        line = np.asarray(line, dtype=np.float32)

        if line.ndim == 2 and line.shape[1] == 2:
            prepared.append(line)
        elif line.size == 4:
            prepared.append(line.reshape(2, 2))

    # Узлы линий синим
    if prepared:
        nodes = np.vstack(prepared)
        nodes = np.unique(nodes, axis=0)

        ax.scatter(
            nodes[:, 0],
            nodes[:, 1],
            s=35,
            c="blue",
            marker="o",
            zorder=2
        )

        ax.add_collection(
            LineCollection(
                prepared,
                colors="red",
                linewidths=2,
                zorder=3
            )
        )

    ax.set_xlim(-0.5, img.shape[1] - 0.5)
    ax.set_ylim(img.shape[0] - 0.5, -0.5)
    ax.set_aspect("equal")
    ax.grid(color="gray", linewidth=0.5)
    plt.show()


if __name__ == "__main__":
    main()
```