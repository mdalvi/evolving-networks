import math

from evolving_networks.activations.activation import Activation


class Sigmoid(Activation):
    midpoint = 0.5

    def __init__(self):
        super(Sigmoid, self).__init__()
        pass

    @classmethod
    def activate(cls, z):
        result = 1.0 / (1.0 + math.exp(-z))
        return result, result > cls.midpoint
