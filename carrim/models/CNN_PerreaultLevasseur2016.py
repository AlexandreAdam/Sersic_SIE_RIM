import tensorflow as tf
import keras.api._v2.keras as keras
from carrim.models.utils import get_activation


class CNNPerreaultLevasseur2016(tf.keras.Model):
    """
    Incorporate the function F^{-1}(y) in the model
    """
    def __init__(
            self,
            output_features=13,
            activation="swish"
        ):
        super(CNNPerreaultLevasseur2016, self).__init__()
        activation = get_activation(activation)
        filters = [32, 32, 32, 32, 64, 64, 128, 256]
        kernels = [3, 5, 10, 10, 10, 12, 10, 3]
        downsample_index = [2, 4, 6]
        stride = 2
        self.modules = []
        for module in range(8):
            self.modules.append(
                keras.layers.Conv2D(
                    filters=filters[module],
                    kernel_size=kernels[module],
                    padding="same",
                    data_format="channels_last",
                    activation=activation,
                )
            )
            self.modules.append(
                keras.layers.Conv2D(
                    filters=filters[module],
                    kernel_size=1,
                    padding="same",
                    data_format="channels_last",
                    activation=activation
                )
            )
            if module in downsample_index:
                self.modules.append(
                    keras.layers.Conv2D(
                        filters=filters[module],
                        strides=stride,
                        kernel_size=kernels[module],
                        padding="same",
                        data_format="channels_last",
                        activation=activation
                    )
                )
        self.flatten = keras.layers.Flatten(data_format="channels_last")
        self.dense = keras.layers.Dense(units=512, activation=activation)
        self.output_layer = keras.layers.Dense(units=output_features, activation="linear")

    def __call__(self, y):
        return self.call(y)

    def call(self, y):
        x = tf.identity(y)
        for layer in self.modules:
            x = layer(x)
        x = self.flatten(x)
        x = self.dense(x)
        x = self.output_layer(x)
        return x


if __name__ == '__main__':
    y = tf.random.normal(shape=[1, 192, 192, 1])
    model = CNNPerreaultLevasseur2016()
    print(model.call(y))
    import numpy as np
    print(np.sum([np.prod(var.shape) for var in model.trainable_variables]))
    model.build(y.shape)
    print(model.summary())
