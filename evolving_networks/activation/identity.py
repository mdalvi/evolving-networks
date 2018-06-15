from evolving_networks.activation.activation import Activation


class Identity(Activation):
    midpoint = 0.0

    def __init__(self):
        super(Identity, self).__init__()
        pass

    @classmethod
    def activate(cls, z):
        result = z
        return result, result > cls.midpoint
