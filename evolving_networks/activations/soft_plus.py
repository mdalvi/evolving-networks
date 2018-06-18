"""
# ==============
# References
# ==============

[1] https://en.wikipedia.org/wiki/Rectifier_(neural_networks)
[2] https://en.wikipedia.org/wiki/Activation_function

"""
import math

from evolving_networks.activations.activation import Activation


class SoftPlus(Activation):
    midpoint = 0.0

    def __init__(self):
        super(SoftPlus, self).__init__()
        pass

    @classmethod
    def activate(cls, z):
        result = math.log(1 + math.exp(z))  # [1], [2]
        return result, result > cls.midpoint
