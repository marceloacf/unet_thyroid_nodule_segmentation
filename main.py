import json
from pathlib import Path

import optuna
from thyroid_segmentation.gpu import configurar_gpu

configurar_gpu()
from thyroid_segmentation.config import (
    DATASET_PATH,
    TARGET_SIZE,
    VALIDATION_SIZE,
    SEED,
    USE_CLAHE,
    CLAHE_CLIP_LIMIT,
    CLAHE_TILE_GRID_SIZE,
    FLIP_LEFT_RIGHT,
    FLIP_UP_DOWN,
    BRIGHTNESS_DELTA,
    CONTRAST_RANGE,
    GAMMA_RANGE,
    SPECKLE_STD,
    EPOCHS,
    N_TRIALS,
    MODELS_DIR,
    HISTORIES_DIR,
    OPTUNA_DIR,
    STUDY_NAME,
    OPTUNA_STORAGE
)

from thyroid_segmentation.data import (
    get_dataset_paths,
    split_train_val,
    build_augmenter
)

from thyroid_segmentation.training import build_objective


def criar_pastas_experimento():
    Path(MODELS_DIR).mkdir(parents=True, exist_ok=True)
    Path(HISTORIES_DIR).mkdir(parents=True, exist_ok=True)
    Path(OPTUNA_DIR).mkdir(parents=True, exist_ok=True)


def salvar_resultados_optuna(study):
    best_params_path = Path(OPTUNA_DIR) / "best_params.json"
    best_trial_path = Path(OPTUNA_DIR) / "best_trial.json"
    trials_path = Path(OPTUNA_DIR) / "trials.csv"

    with open(best_params_path, "w") as f:
        json.dump(study.best_params, f, indent=4)

    with open(best_trial_path, "w") as f:
        json.dump(
            {
                "number": study.best_trial.number,
                "value": study.best_value,
                "params": study.best_params
            },
            f,
            indent=4
        )

    study.trials_dataframe().to_csv(trials_path, index=False)


def main():
    criar_pastas_experimento()

    trainval_images, trainval_masks = get_dataset_paths(
        DATASET_PATH,
        split="trainval"
    )

    train_images, val_images, train_masks, val_masks = split_train_val(
        trainval_images,
        trainval_masks,
        validation_size=VALIDATION_SIZE,
        seed=SEED
    )

    augmentation_fn = build_augmenter(
        target_size=TARGET_SIZE,
        flip_left_right=FLIP_LEFT_RIGHT,
        flip_up_down=FLIP_UP_DOWN,
        brightness_delta=BRIGHTNESS_DELTA,
        contrast_range=CONTRAST_RANGE,
        gamma_range=GAMMA_RANGE,
        speckle_std=SPECKLE_STD
    )

    
    # augmentation_fn = build_augmenter(
    #     target_size=TARGET_SIZE,
    #     flip_left_right=FLIP_LEFT_RIGHT,
    #     flip_up_down=FLIP_UP_DOWN,
    #     brightness_delta=BRIGHTNESS_DELTA,
    #     contrast_range=CONTRAST_RANGE
    # )

    objective = build_objective(
        train_images=train_images,
        train_masks=train_masks,
        val_images=val_images,
        val_masks=val_masks,
        augmentation_fn=augmentation_fn,
        checkpoint_path=MODELS_DIR,
        histories_path=HISTORIES_DIR,
        target_size=TARGET_SIZE,
        epochs=EPOCHS,
        use_clahe=USE_CLAHE
    )

    study = optuna.create_study(
        study_name=STUDY_NAME,
        storage=OPTUNA_STORAGE,
        direction="minimize",
        load_if_exists=True
    )
    study.optimize(objective, n_trials=N_TRIALS)

    salvar_resultados_optuna(study)

    print("Melhor trial:", study.best_trial.number)
    print("Melhor valor:", study.best_value)
    print("Melhores parâmetros:", study.best_params)


if __name__ == "__main__":
    main()