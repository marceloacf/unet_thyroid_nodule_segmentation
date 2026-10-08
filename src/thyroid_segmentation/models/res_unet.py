from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D

from thyroid_segmentation.models.blocks_res import (
    residual_stage,
    residual_encoder_block,
    decoder_block
)

def build_residual_attention_unet(
    input_shape,
    num_filters=64,
    dropout_rate=0.0,
    blocks_per_stage=(1, 1, 1, 1),
    use_attention=True
):
    inputs = Input(input_shape)

    # Encoder residual
    conv1, pool1 = residual_encoder_block(
        inputs,
        num_filters,
        n_blocks=blocks_per_stage[0]
    )

    conv2, pool2 = residual_encoder_block(
        pool1,
        num_filters * 2,
        n_blocks=blocks_per_stage[1]
    )

    conv3, pool3 = residual_encoder_block(
        pool2,
        num_filters * 4,
        n_blocks=blocks_per_stage[2]
    )

    conv4, pool4 = residual_encoder_block(
        pool3,
        num_filters * 8,
        n_blocks=blocks_per_stage[3]
    )

    # Gargalo residual
    bottleneck = residual_stage(
        pool4,
        num_filters * 16,
        n_blocks=1,
        dropout_rate=dropout_rate
    )

    # Decoder com Attention Gates
    decoder1 = decoder_block(
        bottleneck,
        conv4,
        num_filters * 8,
        dropout_rate,
        use_attention
    )

    decoder2 = decoder_block(
        decoder1,
        conv3,
        num_filters * 4,
        dropout_rate,
        use_attention
    )

    decoder3 = decoder_block(
        decoder2,
        conv2,
        num_filters * 2,
        dropout_rate,
        use_attention
    )

    decoder4 = decoder_block(
        decoder3,
        conv1,
        num_filters,
        dropout_rate,
        use_attention
    )

    outputs = Conv2D(
        1,
        (1, 1),
        activation="sigmoid"
    )(decoder4)

    model = Model(inputs, outputs)

    return model