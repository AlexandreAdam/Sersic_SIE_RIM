import tensorflow as tf
import keras.api._v2.keras as keras
from carrim.models.utils import get_activation
from carrim.models.CNN_PerreaultLevasseur2016 import CNNPerreaultLevasseur2016
from tensorflow.keras.applications import ResNet50, ResNet50V2, ResNet101, ResNet101V2, InceptionV3, InceptionResNetV2

MAP = {
    "resnet50": ResNet50,
    "resnet50V2": ResNet50V2,
    "resnet101": ResNet101,
    "resnet101V2": ResNet101,
    "inceptionV3": InceptionV3,
    "inception_resnetV2": InceptionResNetV2,
}


class CNN(tf.keras.Model):
    """
    Incorporate the function F^{-1}(y) in the model
    """
    def __init__(
            self,
            architecture="custom",
            levels=4,
            layer_per_level=2,
            output_features=13,
            kernel_size=3,
            input_kernel_size=11,
            strides=2,
            filters=32,
            filter_scaling=1,
            filter_cap=1024,
            activation="relu",
            dense=512
        ):
        super(CNN, self).__init__()
        if architecture == "custom":
            activation = get_activation(activation)
            self._feature_layers = []
            for i in range(levels):
                if i == 0:  # input layer
                    self._feature_layers.append(
                        keras.layers.Conv2D(
                            kernel_size=input_kernel_size,
                            filters=filters,
                            padding="same",
                            activation=activation
                        )
                    )
                self._feature_layers.extend([
                    keras.layers.Conv2D(
                        kernel_size=kernel_size,
                        filters=min(int(filters * filter_scaling ** i), filter_cap),
                        padding="same",
                        activation=activation
                    )
                    for _ in range(layer_per_level)]
                )
                self._feature_layers.append(
                    keras.layers.Conv2D(
                        kernel_size=kernel_size,
                        filters=min(int(filters * filter_scaling ** i), filter_cap),
                        padding="same",
                        activation=activation,
                        strides=strides
                    )
                )

            def call_custom(y):
                x = tf.identity(y)
                for layer in self._feature_layers:
                    x = layer(x)
                return x
            self.call_method = call_custom

        elif architecture == "perreault_levasseur2016":
            self.call_method = CNNPerreaultLevasseur2016()
        else:
            self.call_method = MAP[architecture](include_top=False, input_shape=[None, None, 1], weights=None)

        self.flatten = keras.layers.Flatten(data_format="channels_last")
        self.dense = keras.layers.Dense(units=dense, activation=activation)
        self.output_layer = keras.layers.Dense(units=output_features, activation="linear")

    def __call__(self, y):
        return self.call(y)

    def call(self, y):
        x = self.call_method(y)
        x = self.flatten(x)
        x = self.dense(x)
        x = self.output_layer(x)
        return x


if __name__ == '__main__':
    y = tf.random.normal(shape=[1, 192, 192, 1])
    # model = CNN(architecture="perreault_levasseur2016")
    model = CNN(architecture="resnet101")
    # model = CNN(architecture="custom", levels=5, layer_per_level=2, filters=128)

    print(model.call(y))
    import numpy as np
    print(np.sum([np.prod(var.shape) for var in model.trainable_variables]))
    model.build(y.shape)
    print(model.summary())
