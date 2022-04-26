import tensorflow as tf
from .layers import GRU
from .utils import get_activation


class ModelAnalytic(tf.keras.Model):
    def __init__(
            self,
            units=32,
            activation="tanh"
        ):

        super(ModelAnalytic, self).__init__()
        activation = get_activation(activation)
        self.hidden_units = units

        self.gru1 = GRU(hidden_units=units)
        self.gru2 = GRU(hidden_units=units)
        self.input_layer = tf.keras.layers.Dense(units=units, activation=activation)
        self.hidden_layer = tf.keras.layers.Dense(units=units, activation=activation)
        self.output_layer = tf.keras.layers.Dense(units=13, activation="linear")

    def __call__(self, xt, states):
        return self.call(xt, states)

    def call(self, xt, states):
        ht1, ht2 = states
        dxt = tf.identity(xt)
        for layer in self._feature_layers:
            dxt = layer(dxt)
        dxt, new_ht1 = self.gru(dxt, ht1)
        dxt = self.hidden_layer(dxt)
        dxt, new_ht2 = self.gru(dxt, ht2)
        for layer in self._reconstruction_layers:
            dxt = layer(dxt)
        dxt = self.output_layer(dxt)
        return dxt, new_state

    def init_hidden_states(self, batch_size):
        return tf.zeros(shape=(batch_size, self.hidden_units))

