import math

from evolving_networks.activation.activation import Activation


class Sin(Activation):
    midpoint = 0.0

    def __init__(self):
        super(Sin, self).__init__()
        pass

    @classmethod
    def activate(cls, z):
        result = math.sin(z)
        return result, result > cls.midpoint
