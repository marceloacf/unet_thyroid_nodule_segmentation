from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from thyroid_segmentation.evaluation.metrics import calcular_metricas_batch


MODOS_VALIDOS = {"random", "first", "best", "worst"}


def gerar_predicoes_visualizacao(
    model,
    dataset,
    image_paths,
    threshold=0.5,
):
    """
    Executa a inferência no dataset e mantém, para cada imagem:
    - imagem de entrada já pré-processada;
    - ground truth;
    - probabilidade produzida pelo modelo;
    - máscara binária após threshold;
    - Dice e IoU por imagem.

    O dataset deve estar com training=False para preservar a mesma
    ordem de image_paths.
    """
    images_all = []
    masks_all = []
    probs_all = []

    for images, masks in dataset:
        probabilities = model(
            images,
            training=False,
        ).numpy()

        images_all.append(images.numpy())
        masks_all.append(masks.numpy())
        probs_all.append(probabilities)

    if not images_all:
        raise ValueError("O dataset de visualização está vazio.")

    images_np = np.concatenate(images_all, axis=0)
    masks_np = np.concatenate(masks_all, axis=0)
    probs_np = np.concatenate(probs_all, axis=0)

    if len(images_np) != len(image_paths):
        raise RuntimeError(
            "A quantidade de predições não corresponde à quantidade "
            f"de caminhos de imagem. Predições: {len(images_np)}; "
            f"imagens: {len(image_paths)}."
        )

    metrics = calcular_metricas_batch(
        y_true=masks_np,
        y_prob=probs_np,
        threshold=threshold,
    )

    preds_bin = (
        probs_np >= threshold
    ).astype(np.uint8)

    return {
        "images": images_np,
        "masks": (masks_np >= 0.5).astype(np.uint8),
        "probabilities": probs_np,
        "predictions": preds_bin,
        "dice": np.asarray(metrics["dice"]),
        "iou": np.asarray(metrics["iou"]),
        "precision": np.asarray(metrics["precision"]),
        "recall": np.asarray(metrics["recall"]),
        "image_paths": list(image_paths),
    }


def selecionar_indices_visualizacao(
    resultados,
    num_samples=15,
    mode="random",
    seed=42,
    indices_especificos=None,
):
    """
    Seleciona quais imagens serão exibidas.

    mode:
        random -> amostra aleatória reprodutível
        first  -> primeiras imagens
        best   -> maiores Dice
        worst  -> menores Dice

    Se indices_especificos for informado, ele tem prioridade sobre mode.
    """
    total = len(resultados["images"])

    if total == 0:
        return np.array([], dtype=int)

    if indices_especificos is not None:
        indices = np.asarray(
            indices_especificos,
            dtype=int,
        )

        invalidos = indices[
            (indices < 0) | (indices >= total)
        ]

        if len(invalidos) > 0:
            raise ValueError(
                "Há índices fora do intervalo do conjunto de teste: "
                f"{invalidos.tolist()}. Intervalo válido: 0 a {total - 1}."
            )

        return indices

    num_samples = min(
        int(num_samples),
        total,
    )

    mode = str(mode).lower()

    if mode not in MODOS_VALIDOS:
        raise ValueError(
            f"VISUALIZATION_MODE inválido: {mode}. "
            f"Use um de {sorted(MODOS_VALIDOS)}."
        )

    if mode == "first":
        return np.arange(
            num_samples,
            dtype=int,
        )

    if mode == "best":
        return np.argsort(
            resultados["dice"]
        )[::-1][:num_samples]

    if mode == "worst":
        return np.argsort(
            resultados["dice"]
        )[:num_samples]

    rng = np.random.default_rng(seed)

    return rng.choice(
        total,
        size=num_samples,
        replace=False,
    )


def _plot_overlay(
    ax,
    image,
    ground_truth,
    prediction,
):
    """
    Sobrepõe os contornos da máscara real e da predição na imagem.
    """
    ax.imshow(
        image,
        cmap="gray",
        vmin=0.0,
        vmax=1.0,
    )

    if np.any(ground_truth):
        ax.contour(
            ground_truth,
            levels=[0.5],
            colors=["lime"],
            linewidths=1.5,
        )

    if np.any(prediction):
        ax.contour(
            prediction,
            levels=[0.5],
            colors=["red"],
            linewidths=1.5,
        )

    ax.set_title(
        "Sobreposição\nGT=verde | Pred=vermelho"
    )
    ax.axis("off")


def visualizar_predicoes(
    resultados,
    indices,
    output_dir,
    samples_per_figure=5,
    save_figures=True,
    show_figures=True,
    show_overlay=True,
    dpi=150,
):
    """
    Exibe e/ou salva figuras com:
    Imagem | Ground truth | Predição | Sobreposição

    As métricas Dice, IoU, precisão e recall são mostradas no título
    de cada caso.
    """
    indices = np.asarray(
        indices,
        dtype=int,
    )

    if len(indices) == 0:
        print("Nenhuma imagem selecionada para visualização.")
        return []

    samples_per_figure = max(
        1,
        int(samples_per_figure),
    )

    output_dir = Path(output_dir)

    if save_figures:
        output_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

    saved_files = []

    n_cols = 4 if show_overlay else 3

    for page_start in range(
        0,
        len(indices),
        samples_per_figure,
    ):
        page_indices = indices[
            page_start:
            page_start + samples_per_figure
        ]

        n_rows = len(page_indices)

        fig, axes = plt.subplots(
            n_rows,
            n_cols,
            figsize=(4.0 * n_cols, 3.6 * n_rows),
            squeeze=False,
        )

        for row, idx in enumerate(page_indices):
            image = resultados["images"][idx].squeeze()
            ground_truth = resultados["masks"][idx].squeeze()
            prediction = resultados["predictions"][idx].squeeze()

            dice = float(resultados["dice"][idx])
            iou = float(resultados["iou"][idx])
            precision = float(resultados["precision"][idx])
            recall = float(resultados["recall"][idx])

            image_name = Path(
                resultados["image_paths"][idx]
            ).name

            axes[row, 0].imshow(
                image,
                cmap="gray",
                vmin=0.0,
                vmax=1.0,
            )
            axes[row, 0].set_title(
                f"Imagem #{idx}\n{image_name}"
            )
            axes[row, 0].axis("off")

            axes[row, 1].imshow(
                ground_truth,
                cmap="gray",
                vmin=0,
                vmax=1,
            )
            axes[row, 1].set_title(
                "Ground truth"
            )
            axes[row, 1].axis("off")

            axes[row, 2].imshow(
                prediction,
                cmap="gray",
                vmin=0,
                vmax=1,
            )
            axes[row, 2].set_title(
                "Predição\n"
                f"Dice={dice:.4f} | IoU={iou:.4f}\n"
                f"Prec={precision:.4f} | Rec={recall:.4f}"
            )
            axes[row, 2].axis("off")

            if show_overlay:
                _plot_overlay(
                    ax=axes[row, 3],
                    image=image,
                    ground_truth=ground_truth,
                    prediction=prediction,
                )

        fig.suptitle(
            "Segmentação de nódulos da tireoide",
            fontsize=14,
        )
        fig.tight_layout(
            rect=[0, 0, 1, 0.98]
        )

        page_number = (
            page_start // samples_per_figure
        ) + 1

        if save_figures:
            file_path = output_dir / (
                f"predicoes_pagina_{page_number:02d}.png"
            )
            fig.savefig(
                file_path,
                dpi=dpi,
                bbox_inches="tight",
            )
            saved_files.append(
                str(file_path)
            )

        if show_figures:
            plt.show()
        else:
            plt.close(fig)

    return saved_files
