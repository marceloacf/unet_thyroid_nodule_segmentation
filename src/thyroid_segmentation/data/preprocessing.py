import cv2
import numpy as np
import tensorflow as tf


def apply_clahe_np(image, clip_limit=2.0, tile_grid_size=(8, 8)):
    image = np.squeeze(image)

    image = np.clip(image, 0.0, 1.0)
    image_uint8 = (image * 255).astype(np.uint8)

    clahe = cv2.createCLAHE(
        clipLimit=clip_limit,
        tileGridSize=tile_grid_size
    )

    image_clahe = clahe.apply(image_uint8)

    image_clahe = image_clahe.astype(np.float32) / 255.0
    image_clahe = np.expand_dims(image_clahe, axis=-1)

    return image_clahe


def apply_clahe_tf(image, target_size, clip_limit=2.0, tile_grid_size=(8, 8)):
    image = tf.numpy_function(
        func=lambda img: apply_clahe_np(
            img,
            clip_limit=clip_limit,
            tile_grid_size=tile_grid_size
        ),
        inp=[image],
        Tout=tf.float32
    )

    image.set_shape([target_size, target_size, 1])

    return image


def load_image_mask_for_visualization(image_path, mask_path, target_size):
    image = tf.io.read_file(image_path)
    image = tf.image.decode_jpeg(image, channels=1)
    image = tf.image.resize(image, [target_size, target_size])
    image = tf.cast(image, tf.float32) / 255.0
    image.set_shape([target_size, target_size, 1])

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

    return image.numpy(), mask.numpy()