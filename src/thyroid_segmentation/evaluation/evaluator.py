import os

import pandas as pd
import tensorflow as tf

from thyroid_segmentation.data.dataset import build_dataset

from thyroid_segmentation.training.losses import (
    dice_loss,
    dice_coef,
    iou_coef,
)

from thyroid_segmentation.evaluation.metrics import (
    calcular_metricas_batch,
)


CUSTOM_OBJECTS = {
    "dice_loss": dice_loss,
    "dice_coef": dice_coef,
    "iou_coef": iou_coef,
}


def obter_caminho_modelo(
    models_dir,
    fold
):
    """
    Retorna o caminho esperado do modelo de um fold.
    """

    return os.path.join(
        str(models_dir),
        f"fold_{fold}",
        f"modelo_fold_{fold}.keras"
    )


def carregar_modelo_fold(
    models_dir,
    fold
):
    """
    Carrega o melhor modelo salvo para determinado fold.
    """

    model_path = obter_caminho_modelo(
        models_dir=models_dir,
        fold=fold
    )

    if not os.path.isfile(model_path):
        raise FileNotFoundError(
            f"Modelo do fold {fold} não encontrado: "
            f"{model_path}"
        )

    print(f"Carregando modelo: {model_path}")

    model = tf.keras.models.load_model(
        model_path,
        custom_objects=CUSTOM_OBJECTS,
        compile=False
    )

    return model, model_path


def construir_dataset_teste(
    test_images,
    test_masks,
    target_size,
    batch_size,
    use_clahe=False,
    clahe_clip_limit=2.0,
    clahe_tile_grid_size=(8, 8)
):
    """
    Constrói o dataset de teste sem augmentation.
    """

    return build_dataset(
        image_paths=test_images,
        mask_paths=test_masks,
        target_size=target_size,
        batch_size=batch_size,
        training=False,
        augmentation_fn=None,
        use_clahe=use_clahe,
        clahe_clip_limit=clahe_clip_limit,
        clahe_tile_grid_size=clahe_tile_grid_size
    )


def avaliar_modelo_no_teste(
    model,
    test_dataset,
    test_image_paths,
    fold,
    threshold=0.5
):
    """
    Executa a predição e calcula as métricas de cada
    imagem do conjunto de teste.

    Retorna um DataFrame com uma linha por imagem.
    """

    registros = []
    image_index = 0

    for images, masks in test_dataset:
        probabilities = model(
            images,
            training=False
        ).numpy()

        metricas = calcular_metricas_batch(
            y_true=masks.numpy(),
            y_prob=probabilities,
            threshold=threshold
        )

        batch_size_atual = int(
            images.shape[0]
        )

        for batch_index in range(batch_size_atual):
            if image_index >= len(test_image_paths):
                raise RuntimeError(
                    "O modelo produziu mais predições do que "
                    "o número de imagens de teste."
                )

            registros.append({
                "fold": int(fold),
                "image_path": test_image_paths[image_index],
                "threshold": float(threshold),

                "dice": float(
                    metricas["dice"][batch_index]
                ),

                "iou": float(
                    metricas["iou"][batch_index]
                ),

                "precision": float(
                    metricas["precision"][batch_index]
                ),

                "recall": float(
                    metricas["recall"][batch_index]
                ),

                "tp": int(
                    metricas["tp"][batch_index]
                ),

                "fp": int(
                    metricas["fp"][batch_index]
                ),

                "fn": int(
                    metricas["fn"][batch_index]
                ),
            })

            image_index += 1

    if image_index != len(test_image_paths):
        raise RuntimeError(
            "A quantidade de predições não corresponde à "
            "quantidade de imagens de teste. "
            f"Predições: {image_index}; "
            f"imagens: {len(test_image_paths)}."
        )

    return pd.DataFrame(registros)