import gc
from pathlib import Path

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
    MODELS_DIR,
    RESULTS_DIR,
    THRESHOLD,
    SEED,
    VISUALIZATION_FOLD,
    VISUALIZATION_NUM_SAMPLES,
    VISUALIZATION_MODE,
    VISUALIZATION_INDICES,
    VISUALIZATION_SAMPLES_PER_FIGURE,
    VISUALIZATION_SAVE,
    VISUALIZATION_SHOW,
    VISUALIZATION_SHOW_OVERLAY,
)

from thyroid_segmentation.data.paths import get_dataset_paths

from thyroid_segmentation.evaluation.evaluator import (
    carregar_modelo_fold,
    construir_dataset_teste,
)

from thyroid_segmentation.visualization import (
    gerar_predicoes_visualizacao,
    selecionar_indices_visualizacao,
    visualizar_predicoes,
)


def main():
    print("=" * 80)
    print("Visualização das predições do modelo")
    print("=" * 80)

    print(f"Fold: {VISUALIZATION_FOLD}")
    print(f"Entrada: {TARGET_SIZE} x {TARGET_SIZE}")
    print(f"Threshold: {THRESHOLD}")
    print(f"CLAHE: {USE_CLAHE}")
    print(f"Modo: {VISUALIZATION_MODE}")
    print(f"Quantidade: {VISUALIZATION_NUM_SAMPLES}")

    test_images, test_masks = get_dataset_paths(
        DATASET_PATH,
        split="test",
    )

    print(f"Imagens de teste disponíveis: {len(test_images)}")

    test_dataset = construir_dataset_teste(
        test_images=test_images,
        test_masks=test_masks,
        target_size=TARGET_SIZE,
        batch_size=BATCH_SIZE,
        use_clahe=USE_CLAHE,
        clahe_clip_limit=CLAHE_CLIP_LIMIT,
        clahe_tile_grid_size=CLAHE_TILE_GRID_SIZE,
    )

    tf.keras.backend.clear_session()
    gc.collect()

    model, model_path = carregar_modelo_fold(
        models_dir=MODELS_DIR,
        fold=VISUALIZATION_FOLD,
    )

    print(f"Modelo carregado: {model_path}")

    resultados = gerar_predicoes_visualizacao(
        model=model,
        dataset=test_dataset,
        image_paths=test_images,
        threshold=THRESHOLD,
    )

    indices = selecionar_indices_visualizacao(
        resultados=resultados,
        num_samples=VISUALIZATION_NUM_SAMPLES,
        mode=VISUALIZATION_MODE,
        seed=SEED,
        indices_especificos=VISUALIZATION_INDICES,
    )

    print(f"Índices selecionados: {indices.tolist()}")

    output_dir = Path(
        RESULTS_DIR,
        "visualization",
        f"fold_{VISUALIZATION_FOLD}",
    )

    saved_files = visualizar_predicoes(
        resultados=resultados,
        indices=indices,
        output_dir=output_dir,
        samples_per_figure=VISUALIZATION_SAMPLES_PER_FIGURE,
        save_figures=VISUALIZATION_SAVE,
        show_figures=VISUALIZATION_SHOW,
        show_overlay=VISUALIZATION_SHOW_OVERLAY,
    )

    if saved_files:
        print("\nFiguras salvas:")
        for file_path in saved_files:
            print(f"- {file_path}")

    del model
    tf.keras.backend.clear_session()
    gc.collect()


if __name__ == "__main__":
    main()
