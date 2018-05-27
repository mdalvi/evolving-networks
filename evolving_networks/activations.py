"""
# ==============
# References
# ==============

[1] https://en.wikipedia.org/wiki/Rectifier_(neural_networks)
[2] https://en.wikipedia.org/wiki/Activation_function
[3] https://arxiv.org/pdf/1710.09967.pdf
[4] https://en.wikipedia.org/wiki/Sinc_function
[5] https://github.com/keras-team/keras/blob/master/keras/activations.py

"""

import math

from evolving_networks.errors import *


def identity_activation(z):
    return z


def binary_step_activation(z):  # [2]
    return 1.0 if z > 0.0 else 0.0


def sigmoid_activation(z):
    return 1.0 / (1.0 + math.exp(-z))


def tanh_activation(z):
    return math.tanh(z)


def arc_tan_activation(z):  # [2]
    return math.atan(z)


def sin_activation(z):
    return math.sin(z)


def soft_sign_activation(z):  # [2]
    return z / (1.0 + abs(z))


def inverse_square_root_unit_activation(z, alpha=3.0):  # [2], [3]
    return z / math.sqrt(1.0 + alpha * (z ** 2))


def relu_activation(z):
    return z if z > 0.0 else 0.0


def leaky_relu_activation(z):
    return z if z > 0.0 else 0.01 * z


def elu_activation(z, alpha=1.0):
    return z if z > 0.0 else alpha * (math.exp(z) - 1.0)


def selu_activation(z, alpha=1.6732632423543772848170429916717, scale=1.0507009873554804934193349852946):  # [2], [5]
    return scale * elu_activation(z, alpha)


def inverse_square_root_linear_unit_activation(z, alpha=3.0):
    return z if z > 0.0 else inverse_square_root_unit_activation(z, alpha)


def soft_plus_activation(z):  # [1], [2]
    return math.log(1 + math.exp(z))


def bent_activation(z):  # [2]
    return ((math.sqrt((z ** 2) + 1.0) - 1.0) / 2.0) + z


def sinc_activation(z):  # [4]
    return 1.0 if z == 0.0 else math.sin(z) / z


def gaussian_activation(z):
    return math.exp(-1.0 * (z ** 2))


def clamped_activation(z):
    return max(-1.0, min(1.0, z))


def inverse_activation(z):
    try:
        z = 1.0 / z
    except ArithmeticError:
        return 0.0
    else:
        return z


def exp_activation(z):
    return math.exp(z)


def abs_activation(z):
    return abs(z)


def hat_activation(z):
    return max(0.0, 1 - abs(z))


class ActivationFunctionSet(object):
    def __init__(self):
        self.functions = {}
        self._add('sigmoid', sigmoid_activation)
        self._add('tanh', tanh_activation)
        self._add('sin', sin_activation)
        self._add('gauss', gaussian_activation)
        self._add('relu', relu_activation)
        self._add('softplus', soft_plus_activation)
        self._add('identity', identity_activation)
        self._add('clamped', clamped_activation)
        self._add('inv', inverse_activation)
        self._add('exp', exp_activation)
        self._add('abs', abs_activation)
        self._add('hat', hat_activation)
        self._add('step', binary_step_activation)
        self._add('arctan', arc_tan_activation)
        self._add('softsign', soft_sign_activation)
        self._add('isrua', inverse_square_root_unit_activation)
        self._add('lrelu', leaky_relu_activation)
        self._add('elu', elu_activation)
        self._add('selu', selu_activation)
        self._add('isrlua', inverse_square_root_linear_unit_activation)
        self._add('bent', bent_activation)
        self._add('sinc', sinc_activation)

    def _add(self, name, func):
        self.functions[name] = func

    def get(self, name):
        f = self.functions.get(name)
        if f is None:
            raise InvalidActivationError("NO SUCH ACTIVATION FUNCTION FOUND [{}]".format(name))
        return f

    def is_valid(self, name):
        return name in self.functions
