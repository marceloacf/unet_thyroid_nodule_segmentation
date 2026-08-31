import tensorflow as tf


def dice_coef(y_true, y_pred, smooth=1e-6):
    y_true = tf.cast(y_true, tf.float32)
    y_pred = tf.cast(y_pred, tf.float32)

    axes = [1, 2, 3]

    intersection = tf.reduce_sum(y_true * y_pred, axis=axes)
    total = tf.reduce_sum(y_true, axis=axes) + tf.reduce_sum(y_pred, axis=axes)

    dice = (2.0 * intersection + smooth) / (total + smooth)

    # Retorna um valor por imagem: shape = (batch,)
    return dice


def dice_loss(y_true, y_pred):
    return 1.0 - dice_coef(y_true, y_pred)


def bce_dice_loss(y_true, y_pred):
    y_true = tf.cast(y_true, tf.float32)
    y_pred = tf.cast(y_pred, tf.float32)

    bce = tf.keras.backend.binary_crossentropy(y_true, y_pred)

    # (batch, H, W, 1) -> (batch,)
    bce = tf.reduce_mean(bce, axis=[1, 2, 3])

    dice = dice_loss(y_true, y_pred)

    return 0.3 * bce + 0.7 * dice


def iou_coef(y_true, y_pred, smooth=1e-6):
    y_true = tf.cast(y_true, tf.float32)
    y_pred = tf.cast(y_pred, tf.float32)

    axes = [1, 2, 3]

    intersection = tf.reduce_sum(y_true * y_pred, axis=axes)
    total = tf.reduce_sum(y_true, axis=axes) + tf.reduce_sum(y_pred, axis=axes)
    union = total - intersection

    iou = (intersection + smooth) / (union + smooth)

    return iou