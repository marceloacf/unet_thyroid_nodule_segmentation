import os
import cv2
import numpy as np
import pandas as pd

from sklearn.model_selection import StratifiedKFold


def calcular_area_mascara(mask_path, target_size=192, threshold=127):
    """
    Calcula a área da máscara binária após resize.

    Retorna:
        area_pixels: número de pixels positivos.
        area_ratio: proporção da imagem ocupada pelo nódulo.
    """

    mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)

    if mask is None:
        raise FileNotFoundError(f"Não foi possível carregar a máscara: {mask_path}")

    mask = cv2.resize(
        mask,
        (target_size, target_size),
        interpolation=cv2.INTER_NEAREST
    )

    mask_bin = (mask > threshold).astype(np.uint8)

    area_pixels = int(mask_bin.sum())
    area_ratio = area_pixels / float(target_size * target_size)

    return area_pixels, area_ratio


def classificar_tamanho_nodulo(
    area_ratio,
    small_max=0.01,
    medium_max=0.05
):
    """
    Classifica o nódulo pela área relativa da máscara.

    pequeno: área <= 1%
    medio:   1% < área <= 5%
    grande:  área > 5%
    """

    if area_ratio <= small_max:
        return "pequeno"

    if area_ratio <= medium_max:
        return "medio"

    return "grande"


def criar_dataframe_com_areas(
    image_paths,
    mask_paths,
    target_size=192,
    small_max=0.01,
    medium_max=0.05
):
    """
    Cria DataFrame com os paths e estatísticas de área das máscaras.
    """

    registros = []

    for idx, (image_path, mask_path) in enumerate(zip(image_paths, mask_paths)):
        area_pixels, area_ratio = calcular_area_mascara(
            mask_path=mask_path,
            target_size=target_size
        )

        classe_area = classificar_tamanho_nodulo(
            area_ratio=area_ratio,
            small_max=small_max,
            medium_max=medium_max
        )

        registros.append({
            "idx": idx,
            "image_path": str(image_path),
            "mask_path": str(mask_path),
            "area_pixels": area_pixels,
            "area_ratio": area_ratio,
            "classe_area": classe_area
        })

    return pd.DataFrame(registros)


def adicionar_folds_estratificados(
    df,
    n_splits=5,
    seed=42,
    stratify_col="classe_area"
):
    """
    Adiciona coluna 'fold' ao DataFrame usando StratifiedKFold.
    """

    df = df.copy().reset_index(drop=True)
    df["fold"] = -1

    skf = StratifiedKFold(
        n_splits=n_splits,
        shuffle=True,
        random_state=seed
    )

    X = df["image_path"].values
    y = df[stratify_col].values

    for fold_idx, (_, val_idx) in enumerate(skf.split(X, y), start=1):
        df.loc[val_idx, "fold"] = fold_idx

    return df


def carregar_ou_criar_folds(
    image_paths,
    mask_paths,
    folds_csv_path,
    target_size=192,
    n_splits=5,
    seed=42,
    small_max=0.01,
    medium_max=0.05
):
    """
    Carrega folds existentes ou cria uma nova divisão estratificada por área.
    """

    folds_csv_path = str(folds_csv_path)

    if os.path.exists(folds_csv_path):
        print(f"Carregando folds existentes: {folds_csv_path}")
        return pd.read_csv(folds_csv_path)

    print("Criando folds estratificados por área da máscara...")

    df = criar_dataframe_com_areas(
        image_paths=image_paths,
        mask_paths=mask_paths,
        target_size=target_size,
        small_max=small_max,
        medium_max=medium_max
    )

    df = adicionar_folds_estratificados(
        df=df,
        n_splits=n_splits,
        seed=seed,
        stratify_col="classe_area"
    )

    os.makedirs(os.path.dirname(folds_csv_path), exist_ok=True)
    df.to_csv(folds_csv_path, index=False)

    print(f"Folds salvos em: {folds_csv_path}")

    return df


def obter_paths_do_fold(df, fold):
    """
    Retorna listas de treino e validação para um fold específico.
    """

    train_df = df[df["fold"] != fold].reset_index(drop=True)
    val_df = df[df["fold"] == fold].reset_index(drop=True)

    train_images = train_df["image_path"].tolist()
    train_masks = train_df["mask_path"].tolist()

    val_images = val_df["image_path"].tolist()
    val_masks = val_df["mask_path"].tolist()

    return train_images, train_masks, val_images, val_masks


def imprimir_distribuicao_folds(df):
    """
    Imprime a distribuição de pequeno/médio/grande por fold.
    """

    print("\nDistribuição absoluta por fold:")
    print(
        pd.crosstab(
            df["fold"],
            df["classe_area"],
            margins=True
        )
    )

    print("\nDistribuição percentual por fold:")
    print(
        pd.crosstab(
            df["fold"],
            df["classe_area"],
            normalize="index"
        ) * 100
    )


def verificar_integridade_fold(df, fold):
    """
    Verifica se há interseção entre treino e validação dentro de um fold.
    """

    train_images, train_masks, val_images, val_masks = obter_paths_do_fold(
        df=df,
        fold=fold
    )

    intersecao_imagens = set(train_images).intersection(set(val_images))
    intersecao_mascaras = set(train_masks).intersection(set(val_masks))

    resultado = {
        "fold": fold,
        "n_train": len(train_images),
        "n_val": len(val_images),
        "intersecao_imagens": len(intersecao_imagens),
        "intersecao_mascaras": len(intersecao_mascaras)
    }

    return resultado