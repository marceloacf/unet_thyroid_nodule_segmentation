import random
from pathlib import Path

import numpy as np
import tensorflow as tf

from thyroid_segmentation.gpu import configurar_gpu

configurar_gpu()

from thyroid_segmentation.config import (
    DATASET_PATH,
    TARGET_SIZE,
    SEED,
    MODELS_DIR,
    HISTORIES_DIR,
    OPTUNA_DIR,
    KFOLD_DIR,
    RESULTS_DIR,
    N_SPLITS,
    FOLDS_CSV_PATH,
    AREA_SMALL_MAX,
    AREA_MEDIUM_MAX,
)

from thyroid_segmentation.data.paths import get_dataset_paths

from thyroid_segmentation.data.kfold import (
    carregar_ou_criar_folds,
    imprimir_distribuicao_folds,
    verificar_integridade_fold
)

from thyroid_segmentation.training.train_kfold import treinar_kfold


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)


def criar_pastas_experimento():
    Path(MODELS_DIR).mkdir(parents=True, exist_ok=True)
    Path(HISTORIES_DIR).mkdir(parents=True, exist_ok=True)
    Path(OPTUNA_DIR).mkdir(parents=True, exist_ok=True)
    Path(KFOLD_DIR).mkdir(parents=True, exist_ok=True)
    Path(RESULTS_DIR).mkdir(parents=True, exist_ok=True)


def main():
    set_seed(SEED)
    criar_pastas_experimento()

    print("=" * 80)
    print("Carregando dataset")
    print("=" * 80)

    trainval_images, trainval_masks = get_dataset_paths(
        DATASET_PATH,
        split="trainval"
    )

    test_images, test_masks = get_dataset_paths(
        DATASET_PATH,
        split="test"
    )

    print(f"Trainval images: {len(trainval_images)}")
    print(f"Trainval masks:  {len(trainval_masks)}")
    print(f"Test images:     {len(test_images)}")
    print(f"Test masks:      {len(test_masks)}")

    print("=" * 80)
    print("Criando/carregando folds estratificados por área da máscara")
    print("=" * 80)

    trainval_df = carregar_ou_criar_folds(
        image_paths=trainval_images,
        mask_paths=trainval_masks,
        folds_csv_path=FOLDS_CSV_PATH,
        target_size=TARGET_SIZE,
        n_splits=N_SPLITS,
        seed=SEED,
        small_max=AREA_SMALL_MAX,
        medium_max=AREA_MEDIUM_MAX
    )

    imprimir_distribuicao_folds(trainval_df)

    print("=" * 80)
    print("Verificando integridade dos folds")
    print("=" * 80)

    for fold in sorted(trainval_df["fold"].unique()):
        resultado = verificar_integridade_fold(
            df=trainval_df,
            fold=fold
        )

        print(
            f"Fold {resultado['fold']} | "
            f"Treino: {resultado['n_train']} | "
            f"Validação: {resultado['n_val']} | "
            f"Interseção imagens: {resultado['intersecao_imagens']} | "
            f"Interseção máscaras: {resultado['intersecao_mascaras']}"
        )

        if (
            resultado["intersecao_imagens"] != 0
            or resultado["intersecao_mascaras"] != 0
        ):
            raise RuntimeError(
                f"Vazamento detectado no fold {fold}. "
                "Há imagens ou máscaras repetidas entre treino e validação."
            )

    print("=" * 80)
    print("Treinamento K-Fold sem Optuna")
    print("=" * 80)

    treinar_kfold(trainval_df)


if __name__ == "__main__":
    main()