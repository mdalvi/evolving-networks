from evolving_networks.activations.activation import Activation


class SoftSign(Activation):
    midpoint = 0.0

    def __init__(self):
        super(SoftSign, self).__init__()
        pass

    @classmethod
    def activate(cls, z):
        result = z / (1.0 + abs(z))
        return result, result > cls.midpoint
