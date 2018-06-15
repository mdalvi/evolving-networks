import math

from evolving_networks.activation.activation import Activation


class Tanh(Activation):
    midpoint = 0.0

    def __init__(self):
        super(Tanh, self).__init__()
        pass

    @classmethod
    def activate(cls, z):
        result = math.tanh(z)
        return result, result > cls.midpoint
