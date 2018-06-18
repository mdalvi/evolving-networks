"""
# ==============
# References
# ==============

[1] https://en.wikipedia.org/wiki/Sinc_function

"""
import math

from evolving_networks.activations.activation import Activation


class Sinc(Activation):
    midpoint = 0.0

    def __init__(self):
        super(Sinc, self).__init__()
        pass

    @classmethod
    def activate(cls, z):
        result = 1.0 if z == 0.0 else math.sin(z) / z  # [1]
        return result, result > cls.midpoint
