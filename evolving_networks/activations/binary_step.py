from evolving_networks.activations.activation import Activation


class BinaryStep(Activation):
    midpoint = 0.5

    def __init__(self):
        super(BinaryStep, self).__init__()
        pass

    @classmethod
    def activate(cls, z):
        result = 1.0 if z > 0.0 else 0.0
        return result, result > cls.midpoint
