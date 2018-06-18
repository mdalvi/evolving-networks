"""
# ==============
# References
# ==============

[1] https://en.wikipedia.org/wiki/Activation_function

"""
import math

from evolving_networks.activations.activation import Activation


class Bent(Activation):
    midpoint = 0.0

    def __init__(self):
        super(Bent, self).__init__()
        pass

    @classmethod
    def activate(cls, z):
        result = ((math.sqrt((z ** 2) + 1.0) - 1.0) / 2.0) + z  # [1]
        return result, result > cls.midpoint
