from evolving_networks.activation.activation import Activation


class Clamped(Activation):
    midpoint = 0.0

    def __init__(self):
        super(Clamped, self).__init__()
        pass

    @classmethod
    def activate(cls, z):
        result = max(-1.0, min(1.0, z))
        return result, result > cls.midpoint
