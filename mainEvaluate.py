import gc
from pathlib import Path

import pandas as pd
import tensorflow as tf

from thyroid_segmentation.gpu import configurar_gpu

configurar_gpu()

from thyroid_segmentation.config import (
    DATASET_PATH,
    TARGET_SIZE,
    BATCH_SIZE,
    USE_CLAHE,
    CLAHE_CLIP_LIMIT,
    CLAHE_TILE_GRID_SIZE,
    N_SPLITS,
    MODELS_DIR,
    RESULTS_DIR,
    THRESHOLD,
)

from thyroid_segmentation.data.paths import (
    get_dataset_paths,
)

from thyroid_segmentation.evaluation.evaluator import (
    carregar_modelo_fold,
    construir_dataset_teste,
    avaliar_modelo_no_teste,
)

from thyroid_segmentation.evaluation.metrics import (
    resumir_metricas_fold,
    resumir_resultados_folds,
)


def main():
    print("=" * 80)
    print("Avaliação simples dos modelos K-Fold")
    print("=" * 80)

    print(f"Threshold: {THRESHOLD}")
    print(f"CLAHE: {USE_CLAHE}")
    print(f"Entrada: {TARGET_SIZE} x {TARGET_SIZE}")
    print(f"Número de folds: {N_SPLITS}")

    evaluation_dir = Path(
        RESULTS_DIR,
        "evaluation_simple"
    )

    evaluation_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    # ============================================================
    # Dataset de teste
    # ============================================================

    test_images, test_masks = get_dataset_paths(
        DATASET_PATH,
        split="test"
    )

    print(f"Imagens de teste: {len(test_images)}")
    print(f"Máscaras de teste: {len(test_masks)}")

    test_dataset = construir_dataset_teste(
        test_images=test_images,
        test_masks=test_masks,
        target_size=TARGET_SIZE,
        batch_size=BATCH_SIZE,
        use_clahe=USE_CLAHE,
        clahe_clip_limit=CLAHE_CLIP_LIMIT,
        clahe_tile_grid_size=CLAHE_TILE_GRID_SIZE
    )

    resultados_folds = []

    # ============================================================
    # Avaliação dos modelos
    # ============================================================

    for fold in range(1, N_SPLITS + 1):
        print("\n" + "=" * 80)
        print(f"Avaliando modelo do fold {fold}")
        print("=" * 80)

        tf.keras.backend.clear_session()
        gc.collect()

        model, model_path = carregar_modelo_fold(
            models_dir=MODELS_DIR,
            fold=fold
        )

        df_metricas = avaliar_modelo_no_teste(
            model=model,
            test_dataset=test_dataset,
            test_image_paths=test_images,
            fold=fold,
            threshold=THRESHOLD
        )

        df_metricas.to_csv(
            evaluation_dir
            / f"metricas_teste_fold_{fold}.csv",
            index=False
        )

        resultado_fold = resumir_metricas_fold(
            df_metricas=df_metricas,
            fold=fold,
            model_path=model_path,
            threshold=THRESHOLD
        )

        resultados_folds.append(
            resultado_fold
        )

        print(
            f"Dice médio:     "
            f"{resultado_fold['dice_mean']:.4f}"
        )

        print(
            f"IoU média:      "
            f"{resultado_fold['iou_mean']:.4f}"
        )

        print(
            f"Precisão média: "
            f"{resultado_fold['precision_mean']:.4f}"
        )

        print(
            f"Recall médio:   "
            f"{resultado_fold['recall_mean']:.4f}"
        )

        del model
        del df_metricas

        tf.keras.backend.clear_session()
        gc.collect()

    # ============================================================
    # Resultado de cada fold
    # ============================================================

    df_folds = pd.DataFrame(
        resultados_folds
    ).sort_values("fold")

    df_folds.to_csv(
        evaluation_dir / "metricas_por_fold.csv",
        index=False
    )

    # ============================================================
    # Média e desvio padrão entre folds
    # ============================================================

    df_resumo = resumir_resultados_folds(
        df_folds=df_folds
    )

    df_resumo.to_csv(
        evaluation_dir / "resumo_folds.csv",
        index=False
    )

    print("\n" + "=" * 80)
    print("Métricas de cada modelo")
    print("=" * 80)

    print(
        df_folds[
            [
                "fold",
                "dice_mean",
                "iou_mean",
                "precision_mean",
                "recall_mean",
            ]
        ].to_string(
            index=False,
            float_format=lambda value: f"{value:.4f}"
        )
    )

    print("\n" + "=" * 80)
    print("Média ± desvio padrão entre os folds")
    print("=" * 80)

    for _, linha in df_resumo.iterrows():
        print(
            f"{linha['metrica'].capitalize():10s}: "
            f"{linha['mean']:.4f} ± "
            f"{linha['std']:.4f}"
        )

    print("\nResultados salvos em:")
    print(evaluation_dir.resolve())


if __name__ == "__main__":
    main()