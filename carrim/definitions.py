import tensorflow as tf
import numpy as np
import tensorflow_probability as tfp
from scipy.special import exp1
from numpy import euler_gamma

DTYPE = tf.float32
LOG10 = tf.constant(np.log(10.), DTYPE)
SQRT2 = tf.constant(1.41421356, DTYPE)
LOGFLOOR = tf.constant(1e-6, DTYPE)

SIGMOID_MIN = tf.constant(1e-3, DTYPE)
SIGMOIN_MAX = tf.constant(1 - 1e-3, DTYPE)

EULER_GAMMA = tf.constant(euler_gamma, DTYPE)


def exp1_plus_log(x):
    # Although both log and exp1 diverge at x=0, exp1(x) + log(x) is an entire function, real for x >= 0
    return tf.where(condition=(x==0), x=-EULER_GAMMA, y=exp1(x) + tf.math.log(x))


def bipolar_elu(x):
    """Bipolar ELU as in https://arxiv.org/abs/1709.04054."""
    x1, x2 = tf.split(x, 2, axis=-1)
    y1 = tf.nn.elu(x1)
    y2 = -tf.nn.elu(-x2)
    return tf.concat([y1, y2], axis=-1)


def bipolar_leaky_relu(x, alpha=0.2, **kwargs):
    """Bipolar Leaky ReLU as in https://arxiv.org/abs/1709.04054."""
    x1, x2 = tf.split(x, 2, axis=-1)
    y1 = tf.nn.leaky_relu(x1, alpha=alpha)
    y2 = -tf.nn.leaky_relu(-x2, alpha=alpha)
    return tf.concat([y1, y2], axis=-1)


class AutoClipper:
    """
    Prem Seetharaman, Gordon Wichern, Bryan Pardo, Jonathan Le Roux.
    "AutoClip: Adaptive Gradient Clipping for Source Separation Networks."
    2020 IEEE 30th International Workshop on Machine Learning for Signal Processing (MLSP). IEEE, 2020.

    Official implementation: https://github.com/pseeth/autoclip
    """
    def __init__(self, clip_percentile, history_size=1000):
        self.clip_percentile = clip_percentile
        self.grad_history = tf.Variable(tf.zeros(history_size), trainable=False)
        self.i = tf.Variable(0, trainable=False)
        self.history_size = history_size

    def __call__(self, gradients):
        norm = tf.linalg.global_norm(gradients)
        assign_idx = tf.math.mod(self.i, self.history_size)
        self.grad_history = self.grad_history[assign_idx].assign(norm)
        self.i = self.i.assign_add(1)
        clip_value = tfp.stats.percentile(self.grad_history[: self.i], q=self.clip_percentile)
        return tf.clip_by_global_norm(gradients, clip_value)


def bipolar_relu(x):
    x1, x2 = tf.split(x, 2, axis=-1)
    y1 = tf.nn.relu(x1)
    y2 = -tf.nn.relu(-x2)
    return tf.concat([y1, y2], axis=-1)


def lrelu(x, alpha=0.3):
    return tf.maximum(x, tf.multiply(x, alpha))


def endlrelu(x, alpha=0.06):
    return tf.maximum(x, tf.multiply(x, alpha))


def m_softplus(x):
    return tf.keras.activations.softplus(x) - tf.keras.activations.softplus( -x -5.0 )


def xsquared(x):
    return (x/4)**2


def log_10(x):
    return tf.math.log(x) / LOG10


def lrelu4p(x, alpha=0.04):
    return tf.maximum(x, tf.multiply(x, alpha))


def logit(x):
    """
    Computes the logit function, i.e. the logistic sigmoid inverse.
    This function has no gradient, so it cannot be used on model output. It is mainly used
    to link physical labels to model labels.

    We clip values for numerical stability.
    Normally, there should not be any values outside of this range anyway, except maybe for the peak at 1.
    """
    x = tf.math.minimum(x, SIGMOIN_MAX)
    x = tf.math.maximum(x, SIGMOID_MIN)
    return -tf.math.log(1. / x - 1.)


def to_float(x):
    """Cast x to float; created because tf.to_float is deprecated."""
    return tf.cast(x, tf.float32)


def inverse_exp_decay(max_step, min_value=0.01, step=None):
    """Inverse-decay exponentially from min_value to 1.0 reached at max_step."""
    inv_base = tf.exp(tf.math.log(min_value) / float(max_step))
    if step is None:
        step = tf.summary.experimental.get_step()
    if step is None:
        return 1.0
    step = to_float(step)
    return inv_base**tf.maximum(float(max_step) - step, 0.0)


class PolynomialSchedule:
    def __init__(self, initial_value, end_value, power, decay_steps, cyclical=False):
        self.initial_value = initial_value
        self.end_value = end_value
        self.power = power
        self.decay_steps = decay_steps
        self.cyclical = cyclical

    def __call__(self, step=None):
        if step is None:
            step = tf.summary.experimental.get_step()
        if self.cyclical:
            step = min(step % (2 * self.decay_steps), self.decay_steps)
        else:
            step = min(step, self.decay_steps)
        return ((self.initial_value - self.end_value) * (1 - step / self.decay_steps) ** (self.power)) + self.end_value


def inverse_lin_decay(max_step, min_value=0.01, step=None):
    """Inverse-decay linearly from min_value to 1.0 reached at max_step."""
    if step is None:
        step = tf.summary.experimental.get_step()
    if step is None:
        return 1.0
    step = to_float(step)
    progress = tf.minimum(step / float(max_step), 1.0)
    return progress * (1.0 - min_value) + min_value


def inverse_sigmoid_decay(max_step, min_value=0.01, step=None):
    """Inverse-decay linearly from min_value to 1.0 reached at max_step."""
    if step is None:
        step = tf.summary.experimental.get_step()
    if step is None:
        return 1.0
    step = to_float(step)

    def sigmoid(x):
        return 1 / (1 + tf.exp(-x))

    def inv_sigmoid(y):
        return tf.math.log(y / (1 - y))

    assert min_value > 0, (
          "sigmoid's output is always >0 and <1. min_value must respect "
          "these bounds for interpolation to work.")
    assert min_value < 0.5, "Must choose min_value on the left half of sigmoid."

    # Find
    #   x  s.t. sigmoid(x ) = y_min and
    #   x' s.t. sigmoid(x') = y_max
    # We will map [0, max_step] to [x_min, x_max].
    y_min = min_value
    y_max = 1.0 - min_value
    x_min = inv_sigmoid(y_min)
    x_max = inv_sigmoid(y_max)

    x = tf.minimum(step / float(max_step), 1.0)  # [0, 1]
    x = x_min + (x_max - x_min) * x  # [x_min, x_max]
    y = sigmoid(x)  # [y_min, y_max]

    y = (y - y_min) / (y_max - y_min)  # [0, 1]
    y = y * (1.0 - y_min)  # [0, 1-y_min]
    y += y_min  # [y_min, 1]
    return y


def conv2_layers_flops(layer):
    _, _, _, input_channels = layer.input_shape
    _, h, w, output_channels = layer.output_shape
    w_h, w_w = layer.kernel_size
    strides_h, strides_w = layer.strides
    flops = h * w * input_channels * output_channels * w_h * w_w / strides_w / strides_h

    flops_bias = np.prod(layer.output_shape[1:]) if layer.use_bias else 0
    flops = 2 * flops + flops_bias  # times 2 since we must consider multiplications and additions
    return flops


def upsampling2d_layers_flops(layer):
    _, h, w, output_channels = layer.output_shape
    return 50 * h * w * output_channels
