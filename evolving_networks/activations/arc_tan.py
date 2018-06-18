import math

from evolving_networks.activations.activation import Activation


class ArcTan(Activation):
    midpoint = 0.0

    def __init__(self):
        super(ArcTan, self).__init__()
        pass

    @classmethod
    def activate(cls, z):
        result = math.atan(z)
        return result, result > cls.midpoint
