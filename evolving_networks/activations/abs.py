from evolving_networks.activations.activation import Activation


class Abs(Activation):
    midpoint = 0.0

    def __init__(self):
        super(Abs, self).__init__()
        pass

    @classmethod
    def activate(cls, z):
        result = abs(z)
        return result, result > cls.midpoint
