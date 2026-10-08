import os
import json
import gc

import numpy as np
import pandas as pd
import tensorflow as tf

from tensorflow.keras.optimizers import Adam

from thyroid_segmentation.config import (
    TARGET_SIZE,
    INPUT_SHAPE,
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
    N_FILTERS,
    BATCH_SIZE,
    LEARNING_RATE,
    DROPOUT_RATE,
    USE_ATTENTION,
    BLOCKS_PER_STAGE,
    MODELS_DIR,
    HISTORIES_DIR,
    RESULTS_DIR,
    FOLDS_TO_RUN,
    EARLY_STOPPING_PATIENCE,
    EARLY_STOPPING_MIN_DELTA,
    REDUCE_LR_PATIENCE,
    REDUCE_LR_FACTOR,
    REDUCE_LR_MIN_DELTA,
    REDUCE_LR_COOLDOWN,
)

from thyroid_segmentation.data.dataset import build_dataset
from thyroid_segmentation.data.augmentation import build_augmenter
from thyroid_segmentation.data.kfold import obter_paths_do_fold

#from thyroid_segmentation.models.res_unet import build_residual_attention_unet
from thyroid_segmentation.models.unet import build_unet

from thyroid_segmentation.training.losses import (
    bce_dice_loss,
    dice_loss,
    dice_coef,
    iou_coef
)

from thyroid_segmentation.training.callbacks import build_kfold_callbacks


def salvar_history_json(history, history_path):
    """
    Salva o histórico do Keras em JSON.

    O history.history contém listas com tipos numpy/float32.
    A conversão para float evita erro de serialização.
    """

    history_serializavel = {
        key: [float(value) for value in values]
        for key, values in history.history.items()
    }

    with open(history_path, "w") as f:
        json.dump(history_serializavel, f, indent=4)


def extrair_metricas_history(history):
    """
    Extrai métricas principais do histórico de treino.

    Como o ModelCheckpoint monitora val_loss, o melhor modelo salvo é o da
    menor val_loss. Por isso registramos:
        - métricas na melhor val_loss;
        - maior Dice de validação observado;
        - maior IoU de validação observado.
    """

    val_losses = history.history["val_loss"]

    best_epoch_idx = int(np.argmin(val_losses))
    best_epoch = best_epoch_idx + 1

    metricas = {
        "best_epoch": best_epoch,
        "best_val_loss": float(history.history["val_loss"][best_epoch_idx]),

        "best_val_accuracy": (
            float(history.history["val_accuracy"][best_epoch_idx])
            if "val_accuracy" in history.history
            else np.nan
        ),

        "best_val_dice_coef": (
            float(history.history["val_dice_coef"][best_epoch_idx])
            if "val_dice_coef" in history.history
            else np.nan
        ),

        "best_val_iou_coef": (
            float(history.history["val_iou_coef"][best_epoch_idx])
            if "val_iou_coef" in history.history
            else np.nan
        ),

        "max_val_dice_coef": (
            float(np.max(history.history["val_dice_coef"]))
            if "val_dice_coef" in history.history
            else np.nan
        ),

        "max_val_iou_coef": (
            float(np.max(history.history["val_iou_coef"]))
            if "val_iou_coef" in history.history
            else np.nan
        ),
    }

    return metricas


def carregar_resultados_existentes(results_path):
    """
    Carrega resultados já salvos.

    Isso permite continuar o K-Fold sem repetir folds já finalizados.
    """

    if os.path.exists(results_path):
        return pd.read_csv(results_path)

    return pd.DataFrame()


def fold_ja_concluido(resultados_df, fold):
    """
    Verifica se determinado fold já está registrado no CSV de resultados.
    """

    if resultados_df.empty:
        return False

    return int(fold) in resultados_df["fold"].astype(int).tolist()


def construir_augmentation():
    """
    Constrói a função de augmentation usando as configurações globais.
    """

    augmentation_fn = build_augmenter(
        target_size=TARGET_SIZE,
        flip_left_right=FLIP_LEFT_RIGHT,
        flip_up_down=FLIP_UP_DOWN,
        brightness_delta=BRIGHTNESS_DELTA,
        contrast_range=CONTRAST_RANGE,
        gamma_range=GAMMA_RANGE,
        speckle_std=SPECKLE_STD
    )

    return augmentation_fn


def treinar_kfold(trainval_df):
    """
    Executa treinamento K-Fold sem Optuna.

    Para cada fold:
        1. cria split treino/validação;
        2. cria datasets;
        3. cria modelo novo do zero;
        4. treina;
        5. salva modelo, CSV, JSON e métricas;
        6. acumula resultado para média ± desvio padrão.
    """

    print("=" * 80)
    print("Iniciando treinamento K-Fold sem Optuna")
    print("=" * 80)

    os.makedirs(str(MODELS_DIR), exist_ok=True)
    os.makedirs(str(HISTORIES_DIR), exist_ok=True)
    os.makedirs(str(RESULTS_DIR), exist_ok=True)

    results_path = os.path.join(
        str(RESULTS_DIR),
        "resultados_kfold.csv"
    )

    resumo_path = os.path.join(
        str(RESULTS_DIR),
        "resumo_kfold.csv"
    )

    resultados_df = carregar_resultados_existentes(results_path)

    if FOLDS_TO_RUN is None:
        folds = sorted(trainval_df["fold"].unique().tolist())
    else:
        folds = FOLDS_TO_RUN

    print(f"Folds a executar: {folds}")

    augmentation_fn = construir_augmentation()

    resultados = []

    if not resultados_df.empty:
        resultados = resultados_df.to_dict("records")

    for fold in folds:
        if fold_ja_concluido(resultados_df, fold):
            print(f"Fold {fold} já concluído. Pulando...")
            continue

        print("=" * 80)
        print(f"Iniciando Fold {fold}")
        print("=" * 80)

        tf.keras.backend.clear_session()
        gc.collect()

        fold_train_images, fold_train_masks, fold_val_images, fold_val_masks = obter_paths_do_fold(
            df=trainval_df,
            fold=fold
        )

        print(f"Treino: {len(fold_train_images)}")
        print(f"Validação: {len(fold_val_images)}")

        intersecao = len(
            set(fold_train_images).intersection(set(fold_val_images))
        )

        print(f"Interseção treino/validação: {intersecao}")

        if intersecao != 0:
            raise RuntimeError(
                f"Vazamento detectado no fold {fold}: "
                f"{intersecao} imagem(ns) repetida(s)."
            )
        
        train_dataset = build_dataset(
            image_paths=fold_train_images,
            mask_paths=fold_train_masks,
            target_size=TARGET_SIZE,
            batch_size=BATCH_SIZE,
            training=True,
            augmentation_fn=augmentation_fn,
            seed=SEED + int(fold),
            use_clahe=USE_CLAHE,
            clahe_clip_limit=CLAHE_CLIP_LIMIT,
            clahe_tile_grid_size=CLAHE_TILE_GRID_SIZE
        )

        val_dataset = build_dataset(
            image_paths=fold_val_images,
            mask_paths=fold_val_masks,
            target_size=TARGET_SIZE,
            batch_size=BATCH_SIZE,
            training=False,
            augmentation_fn=None,
            seed=SEED + int(fold),
            use_clahe=USE_CLAHE,
            clahe_clip_limit=CLAHE_CLIP_LIMIT,
            clahe_tile_grid_size=CLAHE_TILE_GRID_SIZE
        )

        # model = build_residual_attention_unet(
        #     input_shape=INPUT_SHAPE,
        #     num_filters=N_FILTERS,
        #     dropout_rate=DROPOUT_RATE,
        #     blocks_per_stage=BLOCKS_PER_STAGE,
        #     use_attention=USE_ATTENTION
        # )

        model = build_unet(
            input_shape=INPUT_SHAPE,
            num_filters=N_FILTERS,
            dropout_rate=DROPOUT_RATE,
            use_attention=USE_ATTENTION
        )

        model.compile(
            optimizer=Adam(learning_rate=LEARNING_RATE),
            loss=bce_dice_loss,
            metrics=[
                "accuracy",
                dice_coef,
                iou_coef
            ]
        )

        callbacks, model_path, csv_log_path = build_kfold_callbacks(
            checkpoint_path=MODELS_DIR,
            fold=fold,
            learning_rate=LEARNING_RATE,
            early_stopping_patience=EARLY_STOPPING_PATIENCE,
            early_stopping_min_delta=EARLY_STOPPING_MIN_DELTA,
            reduce_lr_patience=REDUCE_LR_PATIENCE,
            reduce_lr_factor=REDUCE_LR_FACTOR,
            reduce_lr_min_delta=REDUCE_LR_MIN_DELTA,
            reduce_lr_cooldown=REDUCE_LR_COOLDOWN,
            use_backup=True
        )

        history = model.fit(
            train_dataset,
            epochs=EPOCHS,
            validation_data=val_dataset,
            callbacks=callbacks,
            verbose=2
        )

        fold_history_dir = os.path.join(
            str(HISTORIES_DIR),
            f"fold_{fold}"
        )

        os.makedirs(fold_history_dir, exist_ok=True)

        history_path = os.path.join(
            fold_history_dir,
            f"history_fold_{fold}.json"
        )

        salvar_history_json(
            history=history,
            history_path=history_path
        )

        metricas = extrair_metricas_history(history)

        resultado_fold = {
            "fold": int(fold),
            "n_train": len(fold_train_images),
            "n_val": len(fold_val_images),

            "n_filters": N_FILTERS,
            "batch_size": BATCH_SIZE,
            "learning_rate": LEARNING_RATE,
            "dropout_rate": DROPOUT_RATE,
            "epochs_configuradas": EPOCHS,
            "epochs_executadas": len(history.history["loss"]),

            **metricas,

            "model_path": model_path,
            "history_path": history_path,
            "csv_log_path": csv_log_path,
        }

        resultados = [
            r for r in resultados
            if int(r["fold"]) != int(fold)
        ]

        resultados.append(resultado_fold)

        resultados_df = pd.DataFrame(resultados).sort_values("fold")
        resultados_df.to_csv(results_path, index=False)

        print(f"Fold {fold} finalizado.")
        print(resultado_fold)

        tf.keras.backend.clear_session()
        gc.collect()

    resultados_df = pd.DataFrame(resultados).sort_values("fold")
    resultados_df.to_csv(results_path, index=False)

    metricas_para_resumo = [
        "best_val_loss",
        "best_val_dice_coef",
        "best_val_iou_coef",
        "max_val_dice_coef",
        "max_val_iou_coef",
    ]

    metricas_disponiveis = [
        metrica for metrica in metricas_para_resumo
        if metrica in resultados_df.columns
    ]

    metricas_resumo = resultados_df[metricas_disponiveis].agg(
        ["mean", "std"]
    )

    metricas_resumo.to_csv(resumo_path)

    print("=" * 80)
    print("Resultados K-Fold")
    print("=" * 80)
    print(resultados_df)

    print("=" * 80)
    print("Resumo K-Fold")
    print("=" * 80)
    print(metricas_resumo)

    print(f"Resultados salvos em: {results_path}")
    print(f"Resumo salvo em: {resumo_path}")

    return resultados_df, metricas_resumo