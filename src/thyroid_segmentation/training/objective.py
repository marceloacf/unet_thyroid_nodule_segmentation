import os
import gc
import json
import tensorflow as tf

from tensorflow.keras.optimizers import Adam

from thyroid_segmentation.config import (
    N_FILTERS_OPTIONS,
    BATCH_SIZE_OPTIONS,
    LEARNING_RATE_MIN,
    LEARNING_RATE_MAX,
    DROPOUT_RATE_MIN,
    DROPOUT_RATE_MAX,
    BLOCKS_PER_STAGE
)

from thyroid_segmentation.data import build_dataset
from thyroid_segmentation.models import build_unet
from thyroid_segmentation.training.losses import (
    bce_dice_loss,
    dice_loss,
    dice_coef,
    iou_coef
)
from thyroid_segmentation.training.callbacks import build_callbacks


def build_objective(
    train_images,
    train_masks,
    val_images,
    val_masks,
    augmentation_fn,
    checkpoint_path,
    histories_path,
    target_size=192,
    epochs=80,
    use_clahe=True
):
    def objective(trial):
        tf.keras.backend.clear_session()
        gc.collect()

        n_filters = trial.suggest_categorical(
            "n_filters",
            N_FILTERS_OPTIONS
        )

        learning_rate = trial.suggest_float(
            "learning_rate",
            LEARNING_RATE_MIN,
            LEARNING_RATE_MAX,
            log=True
        )

        batch_size = trial.suggest_categorical(
            "batch_size",
            BATCH_SIZE_OPTIONS
        )

        dropout_rate = trial.suggest_float(
            "dropout_rate",
            DROPOUT_RATE_MIN,
            DROPOUT_RATE_MAX
        )

        try:
            train_dataset = build_dataset(
                train_images,
                train_masks,
                target_size=target_size,
                batch_size=batch_size,
                training=True,
                augmentation_fn=augmentation_fn,
                use_clahe=use_clahe
            )

            val_dataset = build_dataset(
                val_images,
                val_masks,
                target_size=target_size,
                batch_size=batch_size,
                training=False,
                augmentation_fn=None,
                use_clahe=use_clahe
            )

            model = build_residual_attention_unet(
                input_shape=(target_size, target_size, 1),
                num_filters=n_filters,
                dropout_rate=dropout_rate,
                blocks_per_stage=BLOCKS_PER_STAGE
            )

            # model = build_resnet50_attention_unet(
            #     input_shape=(target_size, target_size, 1),
            #     num_filters=n_filters,
            #     dropout_rate=dropout_rate,
            #     encoder_trainable=False,
            #     weights="imagenet"
            # )

            model.compile(
                optimizer=Adam(learning_rate=learning_rate),
                loss=dice_loss,
                metrics=["accuracy", dice_coef, iou_coef]
            )

            callbacks = build_callbacks(
                checkpoint_path=checkpoint_path,
                trial_number=trial.number,
                learning_rate=learning_rate
            )

            if trial.number == 0:
                model.summary()

                n_train_images = len(train_images)
                n_val_images = len(val_images)

                train_batches = tf.data.experimental.cardinality(train_dataset).numpy()
                val_batches = tf.data.experimental.cardinality(val_dataset).numpy()

                print("=" * 60)
                print("INFORMAÇÕES DO DATASET")
                print(f"Imagens de treino: {n_train_images}")
                print(f"Imagens de validação: {n_val_images}")
                print(f"Batch size: {batch_size}")
                print(f"Batches de treino por época: {train_batches}")
                print(f"Batches de validação por época: {val_batches}")
                print(f"Imagens processadas no treino por época: {n_train_images}")
                print(f"Imagens processadas no treino em {epochs} épocas: {n_train_images * epochs}")
                print("=" * 60)

            history = model.fit(
                train_dataset,
                epochs=epochs,
                validation_data=val_dataset,
                callbacks=callbacks,
                verbose=0
            )

            history_path = os.path.join(
                histories_path,
                f"history_trial_{trial.number}.json"
            )

            with open(history_path, "w") as f:
                json.dump(history.history, f, indent=4, default=float)

            val_loss = min(history.history["val_loss"])

            return val_loss

        except (tf.errors.ResourceExhaustedError, tf.errors.InternalError) as e:
            print(f"Trial {trial.number} interrompido por falta de memória ou erro CUDA.")
            print(e)
            return float("inf")

        finally:
            tf.keras.backend.clear_session()
            gc.collect()

    return objective