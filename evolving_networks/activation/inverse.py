from evolving_networks.activation.activation import Activation


class Inverse(Activation):
    midpoint = 0.0

    def __init__(self):
        super(Inverse, self).__init__()
        pass

    @classmethod
    def activate(cls, z):
        try:
            z = 1.0 / z
        except ArithmeticError:
            result = 0.0
        else:
            result = z
        return result, result < cls.midpoint
