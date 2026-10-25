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

def conv_block(inputs, num_filters, dropout_rate=0.0):
    x = Conv2D(
        num_filters,
        (3, 3),
        padding="same",
        kernel_initializer="he_normal",
        use_bias=False
    )(inputs)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)

    x = Conv2D(
        num_filters,
        (3, 3),
        padding="same",
        kernel_initializer="he_normal",
        use_bias=False
    )(x)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)

    if dropout_rate > 0:
        x = Dropout(dropout_rate)(x)

    return x


def residual_conv_block(inputs, num_filters, dropout_rate=0.0):
    shortcut = inputs

    x = Conv2D(
        num_filters,
        (3, 3),
        padding="same",
        kernel_initializer="he_normal",
        use_bias=False
    )(inputs)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)

    x = Conv2D(
        num_filters,
        (3, 3),
        padding="same",
        kernel_initializer="he_normal",
        use_bias=False
    )(x)
    x = BatchNormalization()(x)

    # Projeção do shortcut quando o número de canais muda
    if shortcut.shape[-1] != num_filters:
        shortcut = Conv2D(
            num_filters,
            (1, 1),
            padding="same",
            kernel_initializer="he_normal",
            use_bias=False
        )(shortcut)
        shortcut = BatchNormalization()(shortcut)

    x = Add()([x, shortcut])
    x = Activation("relu")(x)

    if dropout_rate > 0:
        x = Dropout(dropout_rate)(x)

    return x


def residual_stage(inputs, num_filters, n_blocks=1, dropout_rate=0.0):
    x = inputs

    for i in range(n_blocks):
        # Dropout apenas no último bloco do estágio, se for usado
        current_dropout = dropout_rate if i == n_blocks - 1 else 0.0
        x = residual_conv_block(x, num_filters, current_dropout)

    return x


def residual_encoder_block(inputs, num_filters, n_blocks=1):
    x = residual_stage(
        inputs,
        num_filters=num_filters,
        n_blocks=n_blocks,
        dropout_rate=0.0
    )

    pool = MaxPooling2D(pool_size=(2, 2))(x)

    return x, pool


def attention_gate(gating, skip, num_filters):
    inter_filters = max(num_filters, 1)

    Wg = Conv2D(
        inter_filters,
        (1, 1),
        padding="same",
        kernel_initializer="he_normal",
        use_bias=False
    )(gating)
    Wg = BatchNormalization()(Wg)

    Ws = Conv2D(
        inter_filters,
        (1, 1),
        padding="same",
        kernel_initializer="he_normal",
        use_bias=False
    )(skip)
    Ws = BatchNormalization()(Ws)

    x = Add()([Wg, Ws])
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


def decoder_block(inputs, skip_connection, num_filters, dropout_rate=0.0, use_attention=True):
    x = UpSampling2D(
        size=(2, 2),
        interpolation="bilinear"
    )(inputs)

    x = Conv2D(
        num_filters,
        (3, 3),
        padding="same",
        kernel_initializer="he_normal",
        use_bias=False
    )(x)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)

    if use_attention:
            skip_connection = attention_gate(x, skip_connection, num_filters)
    #attention = attention_gate(x, skip_connection, num_filters) linha anterior sem a condição

    x = concatenate([x, skip_connection])
    x = conv_block(x, num_filters, dropout_rate)

    return x
