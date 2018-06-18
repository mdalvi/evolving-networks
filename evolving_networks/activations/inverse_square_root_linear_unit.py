from evolving_networks.activations.activation import Activation
from evolving_networks.activations.inverse_square_root_unit import InverseSquareRootUnit


class InverseSquareRootLinearUnit(Activation):
    midpoint = 0.0

    def __init__(self):
        super(InverseSquareRootLinearUnit, self).__init__()
        pass

    @classmethod
    def activate(cls, z, alpha=3.0):
        result = z if z > 0.0 else InverseSquareRootUnit.activate(z, alpha)
        return result, result > cls.midpoint
