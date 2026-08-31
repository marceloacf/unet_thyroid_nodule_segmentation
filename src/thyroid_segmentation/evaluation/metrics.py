import numpy as np
import pandas as pd


def divisao_segura(
    numerador,
    denominador,
    valor_se_zero=0.0
):
    """
    Realiza divisão elemento a elemento evitando divisões por zero.
    """

    numerador = np.asarray(
        numerador,
        dtype=np.float64
    )

    denominador = np.asarray(
        denominador,
        dtype=np.float64
    )

    resultado = np.full(
        numerador.shape,
        valor_se_zero,
        dtype=np.float64
    )

    np.divide(
        numerador,
        denominador,
        out=resultado,
        where=denominador != 0
    )

    return resultado


def calcular_metricas_batch(
    y_true,
    y_prob,
    threshold=0.5
):
    """
    Calcula métricas binárias por imagem.

    Parâmetros
    ----------
    y_true:
        Máscaras verdadeiras com shape (batch, H, W, 1).

    y_prob:
        Probabilidades produzidas pelo modelo com shape
        (batch, H, W, 1).

    threshold:
        Threshold aplicado às probabilidades.

    Retorno
    -------
    Dicionário contendo Dice, IoU, precisão, recall,
    verdadeiros positivos, falsos positivos e falsos negativos.
    """

    y_true_bin = np.asarray(y_true) >= 0.5
    y_pred_bin = np.asarray(y_prob) >= threshold

    # Soma H, W e canais separadamente para cada imagem.
    axes = tuple(
        range(1, y_true_bin.ndim)
    )

    tp = np.sum(
        y_true_bin & y_pred_bin,
        axis=axes
    ).astype(np.float64)

    fp = np.sum(
        (~y_true_bin) & y_pred_bin,
        axis=axes
    ).astype(np.float64)

    fn = np.sum(
        y_true_bin & (~y_pred_bin),
        axis=axes
    ).astype(np.float64)

    dice = divisao_segura(
        numerador=2.0 * tp,
        denominador=2.0 * tp + fp + fn,
        valor_se_zero=1.0
    )

    iou = divisao_segura(
        numerador=tp,
        denominador=tp + fp + fn,
        valor_se_zero=1.0
    )

    precision = divisao_segura(
        numerador=tp,
        denominador=tp + fp,
        valor_se_zero=0.0
    )

    recall = divisao_segura(
        numerador=tp,
        denominador=tp + fn,
        valor_se_zero=1.0
    )

    return {
        "dice": dice,
        "iou": iou,
        "precision": precision,
        "recall": recall,
        "tp": tp,
        "fp": fp,
        "fn": fn,
    }


def resumir_metricas_fold(
    df_metricas,
    fold,
    model_path,
    threshold
):
    """
    Calcula a média das métricas das imagens de teste
    para um modelo de determinado fold.
    """

    return {
        "fold": int(fold),
        "n_test": int(len(df_metricas)),
        "threshold": float(threshold),

        "dice_mean": float(
            df_metricas["dice"].mean()
        ),

        "iou_mean": float(
            df_metricas["iou"].mean()
        ),

        "precision_mean": float(
            df_metricas["precision"].mean()
        ),

        "recall_mean": float(
            df_metricas["recall"].mean()
        ),

        "model_path": str(model_path),
    }


def resumir_resultados_folds(df_folds):
    """
    Calcula média e desvio padrão das métricas médias
    obtidas pelos modelos dos folds.

    O desvio padrão é calculado entre os folds usando ddof=1.
    """

    metricas = [
        "dice_mean",
        "iou_mean",
        "precision_mean",
        "recall_mean",
    ]

    registros = []

    for metrica in metricas:
        valores = df_folds[metrica].to_numpy(
            dtype=np.float64
        )

        media = float(
            np.mean(valores)
        )

        desvio_padrao = float(
            np.std(valores, ddof=1)
        )

        registros.append({
            "metrica": metrica.replace("_mean", ""),
            "n_folds": int(len(valores)),
            "mean": media,
            "std": desvio_padrao,
            "mean_std": (
                f"{media:.4f} ± {desvio_padrao:.4f}"
            ),
        })

    return pd.DataFrame(registros)