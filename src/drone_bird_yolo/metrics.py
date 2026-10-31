from __future__ import annotations

from pathlib import Path
from typing import Any, Sequence

import numpy as np


def class_only_confusion_arrays(
    matrix: Any,
    class_count: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Recorta a matriz do Ultralytics sem exibir a linha/coluna background.

    A normalização mantém o total original de cada classe verdadeira no
    denominador. Assim, objetos não detectados continuam reduzindo os valores
    exibidos, mesmo sem a linha visual de background.
    """
    array = np.asarray(matrix, dtype=float)
    if array.ndim != 2 or array.shape[0] < class_count or array.shape[1] < class_count:
        raise ValueError("Matriz de confusão incompatível com a quantidade de classes.")

    class_matrix = array[:class_count, :class_count].copy()
    true_class_totals = array.sum(axis=0)[:class_count]
    normalized = np.divide(
        class_matrix,
        true_class_totals.reshape(1, -1),
        out=np.zeros_like(class_matrix),
        where=true_class_totals.reshape(1, -1) > 0,
    )
    return class_matrix, normalized


def _plot_matrix(
    matrix: np.ndarray,
    class_names: Sequence[str],
    destination: Path,
    *,
    normalized: bool,
) -> None:
    import matplotlib.pyplot as plt

    display = matrix.copy()
    annotation_threshold = 0.005 if normalized else 0.0
    color_threshold = 0.45 if normalized else 0.45 * max(float(display.max()), 1.0)

    figure, axis = plt.subplots(figsize=(8, 7))
    image = axis.imshow(
        display,
        cmap="Blues",
        vmin=0.0,
        vmax=1.0 if normalized else None,
        interpolation="none",
    )
    figure.colorbar(image, ax=axis, fraction=0.046, pad=0.05)

    for row_index, row in enumerate(display):
        for column_index, value in enumerate(row):
            if normalized and value < annotation_threshold:
                continue
            label = f"{value:.2f}" if normalized else f"{int(value)}"
            axis.text(
                column_index,
                row_index,
                label,
                ha="center",
                va="center",
                color="white" if value > color_threshold else "black",
                fontsize=12,
            )

    ticks = np.arange(len(class_names))
    axis.set_xticks(ticks, labels=class_names)
    axis.set_yticks(ticks, labels=class_names)
    axis.set_xlabel("Classe verdadeira")
    axis.set_ylabel("Classe prevista")
    axis.set_title(
        "Matriz de confusão normalizada - somente classes"
        if normalized
        else "Matriz de confusão - somente classes"
    )
    figure.tight_layout()
    figure.savefig(destination, dpi=250)
    plt.close(figure)


def save_class_only_confusion_matrices(
    metrics: Any,
    class_names: Sequence[str],
    save_dir: str | Path,
) -> tuple[Path, Path] | None:
    """Substitui os gráficos padrão por matrizes contendo apenas as classes."""
    confusion = getattr(metrics, "confusion_matrix", None)
    matrix = getattr(confusion, "matrix", None)
    if matrix is None:
        return None

    destination = Path(save_dir)
    destination.mkdir(parents=True, exist_ok=True)
    raw, normalized = class_only_confusion_arrays(matrix, len(class_names))
    raw_path = destination / "confusion_matrix.png"
    normalized_path = destination / "confusion_matrix_normalized.png"
    _plot_matrix(raw, class_names, raw_path, normalized=False)
    _plot_matrix(normalized, class_names, normalized_path, normalized=True)
    return raw_path, normalized_path
