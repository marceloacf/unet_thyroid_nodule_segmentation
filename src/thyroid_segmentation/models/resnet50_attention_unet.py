import tensorflow as tf

from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
    Input,
    Conv2D,
    BatchNormalization,
    Activation
)
from tensorflow.keras.applications import ResNet50

from thyroid_segmentation.models.backbone_preprocessing import (
    ResNet50GrayPreprocess
)
from thyroid_segmentation.models.blocks_backbone import (
    decoder_block_backbone
)


def freeze_backbone_batchnorm(backbone):
    """
    Congela BatchNormalization do backbone.

    Em fine-tuning com batch size pequeno, como 4, 8 ou 16,
    atualizar BatchNorm do backbone pode introduzir estatísticas instáveis.
    """
    for layer in backbone.layers:
        if isinstance(layer, tf.keras.layers.BatchNormalization):
            layer.trainable = False


def build_resnet50_attention_unet(
    input_shape,
    num_filters=32,
    dropout_rate=0.1,
    encoder_trainable=False,
    weights="imagenet"
):
    """
    Attention U-Net com encoder ResNet50 pré-treinado.

    Entrada esperada:
        image: (H, W, 1), float32, escala [0, 1]

    Para input_shape=(192, 192, 1), as resoluções são:
        skip0:              192x192
        conv1_relu:          96x96
        conv2_block3_out:    48x48
        conv3_block4_out:    24x24
        conv4_block6_out:    12x12
        conv5_block3_out:     6x6

    Saída:
        máscara binária probabilística: (H, W, 1)
    """

    inputs = Input(shape=input_shape, name="input_grayscale")

    # =========================================================
    # Skip inicial em resolução original
    # =========================================================
    # A ResNet50 não entrega um skip em 192x192, pois o stem já reduz para 96x96.
    # Este skip raso ajuda a preservar detalhes de borda e textura local.
    skip0 = Conv2D(
        num_filters,
        kernel_size=(3, 3),
        padding="same",
        kernel_initializer="he_normal",
        use_bias=False,
        name="skip0_conv"
    )(inputs)
    skip0 = BatchNormalization(name="skip0_bn")(skip0)
    skip0 = Activation("relu", name="skip0_relu")(skip0)

    # =========================================================
    # Preprocessing específico da ResNet50
    # =========================================================
    x = ResNet50GrayPreprocess(
        name="resnet50_gray_preprocess"
    )(inputs)

    # =========================================================
    # Encoder ResNet50
    # =========================================================
    backbone = ResNet50(
        include_top=False,
        weights=weights,
        input_tensor=x
    )

    backbone.trainable = encoder_trainable
    freeze_backbone_batchnorm(backbone)

    # =========================================================
    # Skip connections da ResNet50
    # =========================================================
    conv1 = backbone.get_layer("conv1_relu").output          # 96x96
    conv2 = backbone.get_layer("conv2_block3_out").output    # 48x48
    conv3 = backbone.get_layer("conv3_block4_out").output    # 24x24
    conv4 = backbone.get_layer("conv4_block6_out").output    # 12x12

    bottleneck = backbone.get_layer("conv5_block3_out").output  # 6x6

    # =========================================================
    # Decoder com Attention Gates
    # =========================================================
    decoder1 = decoder_block_backbone(
        bottleneck,
        conv4,
        num_filters * 8,
        dropout_rate
    )  # 6 -> 12

    decoder2 = decoder_block_backbone(
        decoder1,
        conv3,
        num_filters * 4,
        dropout_rate
    )  # 12 -> 24

    decoder3 = decoder_block_backbone(
        decoder2,
        conv2,
        num_filters * 2,
        dropout_rate
    )  # 24 -> 48

    decoder4 = decoder_block_backbone(
        decoder3,
        conv1,
        num_filters,
        dropout_rate
    )  # 48 -> 96

    decoder5 = decoder_block_backbone(
        decoder4,
        skip0,
        num_filters,
        dropout_rate
    )  # 96 -> 192

    outputs = Conv2D(
        1,
        kernel_size=(1, 1),
        activation="sigmoid",
        name="segmentation_mask"
    )(decoder5)

    model = Model(
        inputs=inputs,
        outputs=outputs,
        name="ResNet50_Attention_UNet"
    )

    return model