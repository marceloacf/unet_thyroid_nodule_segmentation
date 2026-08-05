import os

from sklearn.model_selection import train_test_split


def get_dataset_paths(base_path, split="trainval"):
    if split == "trainval":
        image_dir = os.path.join(base_path, "trainval-image")
        mask_dir = os.path.join(base_path, "trainval-mask")

    elif split == "test":
        image_dir = os.path.join(base_path, "test-image")
        mask_dir = os.path.join(base_path, "test-mask")

    else:
        raise ValueError("split deve ser 'trainval' ou 'test'")

    if not os.path.exists(image_dir):
        raise FileNotFoundError(f"Diretório de imagens não encontrado: {image_dir}")

    if not os.path.exists(mask_dir):
        raise FileNotFoundError(f"Diretório de máscaras não encontrado: {mask_dir}")

    image_files = sorted(os.listdir(image_dir))

    image_paths = []
    mask_paths = []

    for filename in image_files:
        image_path = os.path.join(image_dir, filename)
        mask_path = os.path.join(mask_dir, filename)

        if not os.path.exists(mask_path):
            raise FileNotFoundError(
                f"Máscara correspondente não encontrada para a imagem {filename}: {mask_path}"
            )

        image_paths.append(image_path)
        mask_paths.append(mask_path)

    return image_paths, mask_paths


def split_train_val(image_paths, mask_paths, validation_size=0.15, seed=42):
    """
    Mantido por compatibilidade com experimentos antigos usando split fixo.

    No treinamento com K-Fold, esta função não será usada.
    """

    train_images, val_images, train_masks, val_masks = train_test_split(
        image_paths,
        mask_paths,
        test_size=validation_size,
        random_state=seed,
        shuffle=True
    )

    return train_images, val_images, train_masks, val_masks