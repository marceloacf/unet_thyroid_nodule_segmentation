import tensorflow as tf

from thyroid_segmentation.data.preprocessing import apply_clahe_tf


def build_parser(
    target_size,
    use_clahe=False,
    clahe_clip_limit=2.0,
    clahe_tile_grid_size=(8, 8)
):
    def parse_image_mask(image_path, mask_path):
        image = tf.io.read_file(image_path)
        image = tf.image.decode_jpeg(image, channels=1)
        image = tf.image.resize(image, [target_size, target_size])
        image = tf.cast(image, tf.float32) / 255.0
        image.set_shape([target_size, target_size, 1])

        if use_clahe:
            image = apply_clahe_tf(
                image,
                target_size=target_size,
                clip_limit=clahe_clip_limit,
                tile_grid_size=clahe_tile_grid_size
            )

        mask = tf.io.read_file(mask_path)
        mask = tf.image.decode_jpeg(mask, channels=1)
        mask = tf.image.resize(
            mask,
            [target_size, target_size],
            method=tf.image.ResizeMethod.NEAREST_NEIGHBOR
        )

        mask = tf.cast(mask, tf.float32) / 255.0
        mask = tf.where(mask > 0.5, 1.0, 0.0)
        mask.set_shape([target_size, target_size, 1])

        return image, mask

    return parse_image_mask


def build_dataset(
    image_paths,
    mask_paths,
    target_size,
    batch_size,
    training=True,
    augmentation_fn=None,
    seed=42,
    use_clahe=False,
    clahe_clip_limit=2.0,
    clahe_tile_grid_size=(8, 8)
):
    dataset = tf.data.Dataset.from_tensor_slices((image_paths, mask_paths))

    if training:
        dataset = dataset.shuffle(
            len(image_paths),
            seed=seed,
            reshuffle_each_iteration=True
        )

    parser = build_parser(
        target_size=target_size,
        use_clahe=use_clahe,
        clahe_clip_limit=clahe_clip_limit,
        clahe_tile_grid_size=clahe_tile_grid_size
    )

    dataset = dataset.map(
        parser,
        num_parallel_calls=tf.data.AUTOTUNE
    )

    if training and augmentation_fn is not None:
        dataset = dataset.map(
            augmentation_fn,
            num_parallel_calls=tf.data.AUTOTUNE
        )

    dataset = dataset.batch(batch_size)
    dataset = dataset.prefetch(tf.data.AUTOTUNE)

    return dataset