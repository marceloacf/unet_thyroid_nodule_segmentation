from tensorflow.keras.layers import (
    Conv2D,
    Activation,
    BatchNormalization,
    Dropout,
    UpSampling2D,
    Add,
    Multiply,
    concatenate
)


def conv_block(inputs, num_filters, dropout_rate=0.0):
    x = Conv2D(
        num_filters,
        kernel_size=(3, 3),
        padding="same",
        kernel_initializer="he_normal",
        use_bias=False
    )(inputs)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)

    x = Conv2D(
        num_filters,
        kernel_size=(3, 3),
        padding="same",
        kernel_initializer="he_normal",
        use_bias=False
    )(x)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)

    if dropout_rate > 0:
        x = Dropout(dropout_rate)(x)

    return x


def attention_gate(gating, skip, num_filters):
    """
    Attention Gate para filtrar o skip connection antes da concatenação.

    gating:
        tensor vindo do decoder, já upsampled.

    skip:
        tensor vindo do encoder/backbone, já projetado para num_filters canais.

    num_filters:
        número de filtros do nível atual do decoder.
    """

    inter_filters = max(num_filters // 2, 1)

    Wg = Conv2D(
        inter_filters,
        kernel_size=(1, 1),
        padding="same",
        kernel_initializer="he_normal",
        use_bias=False
    )(gating)
    Wg = BatchNormalization()(Wg)

    Ws = Conv2D(
        inter_filters,
        kernel_size=(1, 1),
        padding="same",
        kernel_initializer="he_normal",
        use_bias=False
    )(skip)
    Ws = BatchNormalization()(Ws)

    x = Add()([Wg, Ws])
    x = Activation("relu")(x)

    psi = Conv2D(
        1,
        kernel_size=(1, 1),
        padding="same",
        kernel_initializer="he_normal"
    )(x)
    psi = Activation("sigmoid")(psi)

    attended_skip = Multiply()([skip, psi])

    return attended_skip


def decoder_block_backbone(
    inputs,
    skip_connection,
    num_filters,
    dropout_rate=0.0
):
    """
    Decoder block específico para backbones pré-treinados.

    Diferença em relação ao decoder da sua U-Net residual:
        - o skip_connection é projetado para num_filters canais
          antes do attention gate.

    Isso é importante porque os skips da ResNet50 têm muitos canais:
        conv1_relu:        64 canais
        conv2_block3_out: 256 canais
        conv3_block4_out: 512 canais
        conv4_block6_out: 1024 canais
    """

    # Upsampling do decoder.
    x = UpSampling2D(
        size=(2, 2),
        interpolation="bilinear"
    )(inputs)

    x = Conv2D(
        num_filters,
        kernel_size=(3, 3),
        padding="same",
        kernel_initializer="he_normal",
        use_bias=False
    )(x)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)

    # Projeção do skip para controlar custo computacional.
    skip = Conv2D(
        num_filters,
        kernel_size=(1, 1),
        padding="same",
        kernel_initializer="he_normal",
        use_bias=False
    )(skip_connection)
    skip = BatchNormalization()(skip)
    skip = Activation("relu")(skip)

    attention = attention_gate(
        gating=x,
        skip=skip,
        num_filters=num_filters
    )

    x = concatenate([x, attention])
    x = conv_block(x, num_filters, dropout_rate)

    return x