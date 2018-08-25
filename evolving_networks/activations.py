"""
# ==============
# References
# ==============

[1] https://pytorch.org/docs/stable/nn.html#hardtanh
[2] https://pytorch.org/docs/stable/nn.html#relu6
[3] https://pytorch.org/docs/stable/nn.html#elu
[4] https://pytorch.org/docs/stable/nn.html#selu
[5] https://pytorch.org/docs/stable/nn.html#leakyrelu
[6] https://pytorch.org/docs/stable/nn.html#prelu
[7] https://pytorch.org/docs/stable/nn.html#rrelu
[8] https://pytorch.org/docs/stable/nn.html#logsigmoid
[9] https://pytorch.org/docs/stable/nn.html#hardshrink
[10] https://pytorch.org/docs/stable/nn.html#tanhshrink
[11] https://pytorch.org/docs/stable/nn.html#softsign
[12] https://pytorch.org/docs/stable/nn.html#softplus
[13] https://pytorch.org/docs/stable/nn.html#softshrink
[14] https://pytorch.org/docs/stable/nn.html#tanh
[15] https://pytorch.org/docs/stable/nn.html#sigmoid
[16] https://pytorch.org/docs/stable/nn.html#relu

"""

import math
import random

from evolving_networks.errors import InvalidActivationError


def identity_activation(x):
    return x


def hard_tanh_activation(x):
    return 1.0 if x > 1 else -1.0 if x < -1 else x  # [1]


def relu6_activation(x):
    return min(max(0, x), 6)  # [2]


def elu_activation(x, alpha=1.0):
    return max(0.0, x) + min(0.0, alpha * (math.exp(x) - 1.0))  # [3]


def selu_activation(x, alpha=1.6732632423543772848170429916717, scale=1.0507009873554804934193349852946):  # [3], [4]
    return scale * elu_activation(x, alpha)


def leaky_relu_activation(x, negative_slope=0.01):
    return max(0.0, x) + (negative_slope * min(0.0, x))  # [5]


def prelu_activation(x, init=0.25):
    return max(0.0, x) + (init * min(0, x))  # [6]


def rrelu_activation(x, lower=0.125, upper=0.3333333333333333):
    return max(0.0, x) + (random.uniform(lower, upper) * min(0, x))  # [7]


def log_sigmoid_activation(x):
    return math.log(1.0 / (1.0 + math.exp(-1.0 * x)))  # [8]


def hard_shrink_activation(x, _lambda=0.5):
    return x if x > _lambda else x if x < (-1.0 * _lambda) else 0.0  # [9]


def tanh_shrink_activation(x):
    return x - tanh_activation(x)  # [10]


def soft_sign_activation(x):
    return x / (1.0 + abs(x))  # [11]


def soft_plus_activation(x, beta=1.0, threshold=20.0):
    return x if x > threshold else ((1.0 / beta) * math.log(1 + (math.exp(beta * x))))  # [12]


def soft_shrink_activation(x, _lambda=0.5):
    return x - _lambda if x > _lambda else x + _lambda if x < (-1.0 * _lambda) else 0.0  # [13]


def tanh_activation(z):
    return math.tanh(z)  # [14]


def sigmoid_activation(x):
    return 1.0 / (1.0 + math.exp(-1.0 * x))  # [15]


def relu_activation(x):
    return max(0.0, x)  # [16]


class Activations(object):
    def __init__(self):
        self.functions = {}
        self._add('identity', identity_activation)
        self._add('hardtanh', hard_tanh_activation)
        self._add('relu6', relu6_activation)
        self._add('elu', elu_activation)
        self._add('selu', selu_activation)
        self._add('lrelu', leaky_relu_activation)
        self._add('prelu', prelu_activation)
        self._add('rrelu', rrelu_activation)
        self._add('logsigmoid', log_sigmoid_activation)
        self._add('hardshrink', hard_shrink_activation)
        self._add('tanhshirnk', tanh_shrink_activation)
        self._add('softsign', soft_sign_activation)
        self._add('softplus', soft_plus_activation)
        self._add('softshrink', soft_shrink_activation)
        self._add('tanh', tanh_activation)
        self._add('sigmoid', sigmoid_activation)
        self._add('relu', relu_activation)

    def _add(self, name, func):
        self.functions[name] = func

    def get(self, name):
        f = self.functions.get(name)
        if f is None:
            raise InvalidActivationError("NO SUCH ACTIVATION FUNCTION FOUND [{}]".format(name))
        return f

    def is_valid(self, name):
        return name in self.functions
