import tensorflow as tf

from tensorflow.keras.layers import Layer
from tensorflow.keras.applications.resnet import preprocess_input


@tf.keras.utils.register_keras_serializable(package="thyroid_segmentation")
class ResNet50GrayPreprocess(Layer):
    """
    Preprocessamento para usar ResNet50 ImageNet com imagens grayscale.

    Entrada esperada:
        shape: (B, H, W, 1)
        dtype: float32
        escala: [0, 1]

    Saída:
        shape: (B, H, W, 3)
        dtype: float32
        padrão esperado pela ResNet50 do Keras.

    Ordem aplicada:
        1. garante float32;
        2. limita a imagem em [0, 1];
        3. replica grayscale para pseudo-RGB;
        4. converte escala [0, 1] para [0, 255];
        5. aplica preprocess_input da ResNet50.
    """

    def call(self, inputs):
        x = tf.cast(inputs, tf.float32)

        # Mantém compatibilidade com seu pipeline atual.
        # Seu parser e augmentation trabalham em [0, 1].
        x = tf.clip_by_value(x, 0.0, 1.0)

        # Grayscale -> pseudo-RGB.
        # (B, H, W, 1) -> (B, H, W, 3)
        x = tf.image.grayscale_to_rgb(x)

        # [0, 1] -> [0, 255]
        x = x * 255.0

        # ResNet50 Keras:
        # RGB -> BGR + zero-centering pela média ImageNet.
        x = preprocess_input(x)

        return x

    def get_config(self):
        return super().get_config()