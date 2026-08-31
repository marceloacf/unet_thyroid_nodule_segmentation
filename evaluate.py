import json
from pathlib import Path

import tensorflow as tf

from thyroid_segmentation.config import (
    DATASET_PATH,
    TARGET_SIZE,
    USE_CLAHE,
    CLAHE_CLIP_LIMIT,
    CLAHE_TILE_GRID_SIZE,
    MODELS_DIR,
    OPTUNA_DIR,
    THRESHOLD
)

from thyroid_segmentation.data import (
    get_dataset_paths,
    build_dataset
)

from thyroid_segmentation.training import (
    bce_dice_loss,
    dice_coef,
    iou_coef
)

from thyroid_segmentation.evaluation import (
    avaliar_segmentacao,
    gerar_resultados_por_imagem,
    encontrar_extremos_metricas
)

from thyroid_segmentation.visualization import (
    mostrar_predicoes,
    mostrar_extremos_metricas
)


def main():
    best_trial_path = Path(OPTUNA_DIR) / "best_trial.json"

    with open(best_trial_path, "r") as f:
        best_trial = json.load(f)

    trial_number = best_trial["number"]
    batch_size = best_trial["params"]["batch_size"]

    model_path = Path(MODELS_DIR) / f"modelo_trial_{trial_number}.keras"

    model = tf.keras.models.load_model(
        model_path,
        custom_objects={
            "bce_dice_loss": bce_dice_loss,
            "dice_coef": dice_coef,
            "iou_coef": iou_coef
        }
    )

    test_images, test_masks = get_dataset_paths(
        DATASET_PATH,
        split="test"
    )

    test_dataset = build_dataset(
        test_images,
        test_masks,
        target_size=TARGET_SIZE,
        batch_size=batch_size,
        training=False,
        augmentation_fn=None,
        use_clahe=USE_CLAHE,
        clahe_clip_limit=CLAHE_CLIP_LIMIT,
        clahe_tile_grid_size=CLAHE_TILE_GRID_SIZE
    )

    resultado = avaliar_segmentacao(
        model=model,
        dataset=test_dataset,
        threshold=THRESHOLD
    )

    print("Métricas globais:")
    for nome, valor in resultado["global"].items():
        print(f"{nome}: {valor:.4f}")

    print("\nMétricas por imagem:")
    for nome, valor in resultado["por_imagem"].items():
        print(f"{nome}: {valor:.4f}")

    mostrar_predicoes(
        model=model,
        dataset=test_dataset,
        num_samples=5,
        threshold=THRESHOLD
    )

    resultados_por_imagem = gerar_resultados_por_imagem(
        model=model,
        dataset=test_dataset,
        threshold=THRESHOLD
    )

    extremos = encontrar_extremos_metricas(resultados_por_imagem)

    mostrar_extremos_metricas(extremos)


if __name__ == "__main__":
    main()