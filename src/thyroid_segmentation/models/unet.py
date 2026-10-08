from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D

from thyroid_segmentation.models.blocks import (
    conv_block,
    encoder_block,
    decoder_block
)


def build_unet(input_shape, num_filters=64, dropout_rate=0.0, use_attention=True):
    inputs = Input(input_shape)

    conv1, pool1 = encoder_block(inputs, num_filters)
    conv2, pool2 = encoder_block(pool1, num_filters * 2)
    conv3, pool3 = encoder_block(pool2, num_filters * 4)
    conv4, pool4 = encoder_block(pool3, num_filters * 8)

    bottleneck = conv_block(pool4, num_filters * 16, dropout_rate)

    decoder1 = decoder_block(bottleneck, conv4, num_filters * 8, dropout_rate, use_attention)
    decoder2 = decoder_block(decoder1, conv3, num_filters * 4, dropout_rate, use_attention)
    decoder3 = decoder_block(decoder2, conv2, num_filters * 2, dropout_rate, use_attention)
    decoder4 = decoder_block(decoder3, conv1, num_filters, dropout_rate, use_attention)

    outputs = Conv2D(1, (1, 1), activation="sigmoid")(decoder4)

    model = Model(inputs, outputs)

    return model