from evolving_networks.activations.activation import Activation


class Relu(Activation):
    midpoint = 0.0

    def __init__(self):
        super(Relu, self).__init__()
        pass

    @classmethod
    def activate(cls, z):
        result = z if z > 0.0 else 0.0
        return result, result > cls.midpoint
