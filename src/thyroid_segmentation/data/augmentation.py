import tensorflow as tf


def build_augmenter(
    target_size,
    flip_left_right=True,
    flip_up_down=False,
    brightness_delta=0.04,
    contrast_range=(0.98, 1.02),
    gamma_range=(0.90, 1.10),
    speckle_std=0.03
):
    def augment_image_mask(image, mask):
        # Une imagem e máscara para garantir que transformações geométricas
        # sejam aplicadas de forma sincronizada.
        combined = tf.concat([image, mask], axis=-1)

        # Flip horizontal
        if flip_left_right:
            do_flip = tf.random.uniform(()) > 0.5
            combined = tf.cond(
                do_flip,
                lambda: tf.image.flip_left_right(combined),
                lambda: combined
            )

        # Flip vertical: manter False por padrão em ultrassom de tireoide.
        if flip_up_down:
            do_flip = tf.random.uniform(()) > 0.5
            combined = tf.cond(
                do_flip,
                lambda: tf.image.flip_up_down(combined),
                lambda: combined
            )

        image = combined[..., :1]
        mask = combined[..., 1:]

        # Brilho leve
        if brightness_delta is not None and brightness_delta > 0:
            image = tf.image.random_brightness(
                image,
                max_delta=brightness_delta
            )

        # Contraste leve
        if contrast_range is not None:
            lower, upper = contrast_range

            image = tf.image.random_contrast(
                image,
                lower=lower,
                upper=upper
            )

        # Gamma augmentation leve
        if gamma_range is not None:
            gamma = tf.random.uniform(
                shape=[],
                minval=gamma_range[0],
                maxval=gamma_range[1]
            )

            image = tf.clip_by_value(image, 1e-6, 1.0)
            image = tf.pow(image, gamma)

        # Speckle noise leve, simulando ruído característico do ultrassom.
        if speckle_std is not None and speckle_std > 0:
            noise = tf.random.normal(
                shape=tf.shape(image),
                mean=0.0,
                stddev=speckle_std,
                dtype=tf.float32
            )

            image = image + image * noise

        image = tf.clip_by_value(image, 0.0, 1.0)

        # Garante que a máscara continue binária após qualquer transformação.
        mask = tf.where(mask > 0.5, 1.0, 0.0)

        image.set_shape([target_size, target_size, 1])
        mask.set_shape([target_size, target_size, 1])

        return image, mask

    return augment_image_mask