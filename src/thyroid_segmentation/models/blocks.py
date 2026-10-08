from tensorflow.keras.layers import (
    Conv2D,
    MaxPooling2D,
    concatenate,
    Activation,
    BatchNormalization,
    Dropout,
    UpSampling2D,
    Add,
    Multiply
)


def conv_block(input_tensor, num_filters, dropout_rate=0.0):
    conv = Conv2D(
        num_filters,
        (3, 3),
        padding="same",
        kernel_initializer="he_normal"
    )(input_tensor)

    conv = BatchNormalization()(conv)
    conv = Activation("relu")(conv)

    conv = Conv2D(
        num_filters,
        (3, 3),
        padding="same",
        kernel_initializer="he_normal"
    )(conv)

    conv = BatchNormalization()(conv)
    conv = Activation("relu")(conv)

    if dropout_rate > 0:
        conv = Dropout(dropout_rate)(conv)

    return conv


def encoder_block(input_tensor, num_filters, dropout_rate=0.0):
    conv = conv_block(input_tensor, num_filters, dropout_rate)
    pool = MaxPooling2D(pool_size=(2, 2))(conv)

    return conv, pool


def attention_gate(gating, skip, num_filters):
    #antes tinha num_filters//2
    inter_filters = max(num_filters, 1)
    wg = Conv2D(
        inter_filters,
        (1, 1),
        padding="same",
        kernel_initializer="he_normal"
    )(gating)

    wg = BatchNormalization()(wg)

    ws = Conv2D(
        inter_filters,
        (1, 1),
        padding="same",
        kernel_initializer="he_normal"
    )(skip)

    ws = BatchNormalization()(ws)

    x = Add()([wg, ws])
    x = Activation("relu")(x)

    psi = Conv2D(
        1,
        (1, 1),
        padding="same",
        kernel_initializer="he_normal"
    )(x)

    psi = Activation("sigmoid")(psi)

    attended_skip = Multiply()([skip, psi])

    return attended_skip


def decoder_block(input_tensor, skip_connection, num_filters, dropout_rate=0.0, use_attention=True):
    upconv = UpSampling2D(
        interpolation="bilinear",
        size=(2, 2)
    )(input_tensor)

    upconv = Conv2D(
        num_filters,
        (3, 3),
        padding="same",
        kernel_initializer="he_normal"
    )(upconv)

    upconv = BatchNormalization()(upconv)
    upconv = Activation("relu")(upconv)

    if use_attention:
        skip_connection = attention_gate(upconv, skip_connection, num_filters)
    #attention = attention_gate(upconv, skip_connection, num_filters) linha anterior sem a condição

    upconv = concatenate([upconv, skip_connection])
    upconv = conv_block(upconv, num_filters, dropout_rate)

    return upconv