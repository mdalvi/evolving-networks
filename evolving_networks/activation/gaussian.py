import math

from evolving_networks.activation.activation import Activation


class Gaussian(Activation):
    midpoint = 0.0

    def __init__(self):
        super(Gaussian, self).__init__()
        pass

    @classmethod
    def activate(cls, z):
        result = math.exp(-1.0 * (z ** 2))
        return result, result > cls.midpoint
