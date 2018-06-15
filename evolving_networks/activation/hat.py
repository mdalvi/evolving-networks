from evolving_networks.activation.activation import Activation


class Hat(Activation):
    midpoint = 0.0

    def __init__(self):
        super(Hat, self).__init__()
        pass

    @classmethod
    def activate(cls, z):
        result = max(0.0, 1 - abs(z))
        return result, result > cls.midpoint
