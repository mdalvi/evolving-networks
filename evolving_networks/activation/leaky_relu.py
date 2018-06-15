from evolving_networks.activation.activation import Activation


class LeakyRelu(Activation):
    midpoint = 0.0

    def __init__(self):
        super(LeakyRelu, self).__init__()
        pass

    @classmethod
    def activate(cls, z):
        result = z if z > 0.0 else 0.01 * z
        return result, result > cls.midpoint
