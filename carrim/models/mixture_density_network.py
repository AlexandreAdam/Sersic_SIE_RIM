"""
Mixture Density Network
author: Alexandre Adam

Heavily inspired from pydelfi implementation by Justin Alsing et. al. (tensorflow 2 branch April 2022)
"""

import tensorflow as tf
import keras.api._v2.keras as keras
import tensorflow_probability as tfp
from carrim.models.utils import get_activation

tfd = tfp.distributions
tfpl = tfp.layers


class MixtureDensityNetwork(keras.Model):
    """
    Implementation of a Mixture density Network to model the conditional density p(data | parameters) with
    k gaussian distribution with full covariance.
    """
    def __init__(self, input_shape: list, k=1, layers=1, units=32, activation="relu"):
        super(MixtureDensityNetwork, self).__init__()
        self.k = k
        self.activation = get_activation(activation)
        self.input_dimension = input_shape if isinstance(input_shape, int) else input_shape[-1]

        mean_vector_dimension = k * self.input_dimension
        covariance_dimension = k * int(self.input_dimension * (self.input_dimension + 1) // 2)
        self.output_dimension = mean_vector_dimension + covariance_dimension + k

        self.network_layers = []
        for _ in range(layers):
            self.network_layers.append(keras.layers.Dense(units=units, activation=activation))
        self.output_layer = keras.layers.Dense(units=self.output_dimension, activation=activation)
        self.mixture = tfpl.MixtureSameFamily(self.k, tfpl.MultivariateNormalTriL(self.input_dimension))

    def __call__(self, x, training=None, mask=None):
        """
        x stands for parameters (e.g. compressed statistics).
        """
        return self.call(x, training=training, mask=mask)

    def call(self, x, training=None, mask=None):
        """
        Input: parameters (e.g. compressed statistics)
        Return: Mixture Density Distribution
        """
        for layer in self.network_layers:
            x = layer(x)
        x = self.output_layer(x)
        x = self.mixture(x) # returns a distribution
        return x

    def log_prob(self, data, parameters):
        """
        Evaluate ln p(data | parameters)
        """
        return self.call(parameters).log_prob(data)

    def sample(self, parameters, n):
        """
        Sample n data examples from the likelihood p(data | parameters)
        """
        return self.call(parameters).sample(n)
