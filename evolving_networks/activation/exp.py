import math

from evolving_networks.activation.activation import Activation


class Exp(Activation):
    midpoint = 0.0

    def __init__(self):
        super(Exp, self).__init__()
        pass

    @classmethod
    def activate(cls, z):
        result = math.exp(z)
        return result, result > cls.midpoint
