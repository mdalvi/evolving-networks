import math

from evolving_networks.activation.activation import Activation


class Elu(Activation):
    midpoint = 0.0

    def __init__(self):
        super(Elu, self).__init__()
        pass

    @classmethod
    def activate(cls, z, alpha=1.0):
        result = z if z > 0.0 else alpha * (math.exp(z) - 1.0)
        return result, result > cls.midpoint
