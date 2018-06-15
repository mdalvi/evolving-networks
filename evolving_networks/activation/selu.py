"""
# ==============
# References
# ==============

[1] https://en.wikipedia.org/wiki/Activation_function
[2] https://github.com/keras-team/keras/blob/master/keras/activations.py

"""
from evolving_networks.activation.activation import Activation
from evolving_networks.activation.elu import Elu


class Selu(Activation):
    midpoint = 0.0

    def __init__(self):
        super(Selu, self).__init__()
        pass

    @classmethod
    def activate(cls, z, alpha=1.6732632423543772848170429916717, scale=1.0507009873554804934193349852946):  # [1], [2]
        result = scale * Elu.activate(z, alpha)
        return result, result > cls.midpoint
