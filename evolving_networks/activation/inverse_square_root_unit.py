"""
# ==============
# References
# ==============

[1] https://en.wikipedia.org/wiki/Activation_function
[2] https://arxiv.org/pdf/1710.09967.pdf

"""

import math

from evolving_networks.activation.activation import Activation


class InverseSquareRootUnit(Activation):
    midpoint = 0.0

    def __init__(self):
        super(InverseSquareRootUnit, self).__init__()
        pass

    @classmethod
    def activate(cls, z, alpha=3.0):  # [1], [2]
        result = z / math.sqrt(1.0 + alpha * (z ** 2))
        return result, result > cls.midpoint
