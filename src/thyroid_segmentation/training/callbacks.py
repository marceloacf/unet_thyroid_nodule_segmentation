import os

from tensorflow.keras.callbacks import (
    ModelCheckpoint,
    EarlyStopping,
    ReduceLROnPlateau,
    CSVLogger,
    BackupAndRestore
)


def build_callbacks(
    checkpoint_path,
    trial_number,
    learning_rate,
    fold=None,
    monitor="val_loss",
    mode="min",
    early_stopping_patience=25,
    early_stopping_min_delta=1e-4,
    reduce_lr_patience=7,
    reduce_lr_factor=0.5,
    reduce_lr_min_delta=1e-4,
    reduce_lr_cooldown=2,
    use_backup=True
):
    """
    Cria callbacks para treinamento com Optuna.

    Compatível com:
        1. Optuna com split fixo:
            build_callbacks(checkpoint_path, trial_number, learning_rate)

        2. Optuna + K-Fold:
            build_callbacks(checkpoint_path, trial_number, learning_rate, fold=fold)
    """

    if fold is None:
        run_dir = os.path.join(
            str(checkpoint_path),
            f"trial_{trial_number}"
        )

        model_filename = f"modelo_trial_{trial_number}.keras"
        csv_filename = f"history_trial_{trial_number}.csv"

    else:
        run_dir = os.path.join(
            str(checkpoint_path),
            f"trial_{trial_number}",
            f"fold_{fold}"
        )

        model_filename = f"modelo_trial_{trial_number}_fold_{fold}.keras"
        csv_filename = f"history_trial_{trial_number}_fold_{fold}.csv"

    os.makedirs(run_dir, exist_ok=True)

    model_path = os.path.join(run_dir, model_filename)
    csv_log_path = os.path.join(run_dir, csv_filename)
    backup_dir = os.path.join(run_dir, "backup")

    callbacks = []

    if use_backup:
        callbacks.append(
            BackupAndRestore(
                backup_dir=backup_dir
            )
        )

    callbacks.append(
        ModelCheckpoint(
            filepath=model_path,
            monitor=monitor,
            mode=mode,
            verbose=1,
            save_best_only=True,
            save_weights_only=False
        )
    )

    callbacks.append(
        EarlyStopping(
            monitor=monitor,
            mode=mode,
            restore_best_weights=True,
            patience=early_stopping_patience,
            min_delta=early_stopping_min_delta,
            verbose=1
        )
    )

    callbacks.append(
        ReduceLROnPlateau(
            monitor=monitor,
            mode=mode,
            factor=reduce_lr_factor,
            patience=reduce_lr_patience,
            min_delta=reduce_lr_min_delta,
            cooldown=reduce_lr_cooldown,
            min_lr=learning_rate * 0.01,
            verbose=1
        )
    )

    callbacks.append(
        CSVLogger(
            filename=csv_log_path,
            separator=",",
            append=os.path.exists(csv_log_path)
        )
    )

    return callbacks


def build_kfold_callbacks(
    checkpoint_path,
    fold,
    learning_rate,
    monitor="val_loss",
    mode="min",
    early_stopping_patience=25,
    early_stopping_min_delta=1e-4,
    reduce_lr_patience=7,
    reduce_lr_factor=0.5,
    reduce_lr_min_delta=1e-4,
    reduce_lr_cooldown=2,
    use_backup=True
):
    """
    Cria callbacks para K-Fold sem Optuna.

    Estrutura gerada:
        checkpoint_path/
            fold_1/
                modelo_fold_1.keras
                history_fold_1.csv
                backup/

            fold_2/
                modelo_fold_2.keras
                history_fold_2.csv
                backup/
    """

    fold_dir = os.path.join(
        str(checkpoint_path),
        f"fold_{fold}"
    )

    os.makedirs(fold_dir, exist_ok=True)

    model_path = os.path.join(
        fold_dir,
        f"modelo_fold_{fold}.keras"
    )

    csv_log_path = os.path.join(
        fold_dir,
        f"history_fold_{fold}.csv"
    )

    backup_dir = os.path.join(
        fold_dir,
        "backup"
    )

    callbacks = []

    if use_backup:
        callbacks.append(
            BackupAndRestore(
                backup_dir=backup_dir
            )
        )

    callbacks.append(
        ModelCheckpoint(
            filepath=model_path,
            monitor=monitor,
            mode=mode,
            verbose=1,
            save_best_only=True,
            save_weights_only=False
        )
    )

    callbacks.append(
        EarlyStopping(
            monitor=monitor,
            mode=mode,
            restore_best_weights=True,
            patience=early_stopping_patience,
            min_delta=early_stopping_min_delta,
            verbose=1
        )
    )

    callbacks.append(
        ReduceLROnPlateau(
            monitor=monitor,
            mode=mode,
            factor=reduce_lr_factor,
            patience=reduce_lr_patience,
            min_delta=reduce_lr_min_delta,
            cooldown=reduce_lr_cooldown,
            min_lr=learning_rate * 0.01,
            verbose=1
        )
    )

    callbacks.append(
        CSVLogger(
            filename=csv_log_path,
            separator=",",
            append=os.path.exists(csv_log_path)
        )
    )

    return callbacks, model_path, csv_log_path