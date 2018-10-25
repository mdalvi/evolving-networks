class Phenome(object):
    def __init__(self):
        pass

    def initialize(self, activations, aggregations):
        raise NotImplementedError()

    def activate(self, inputs):
        raise NotImplementedError()

    def reset(self):
        raise NotImplementedError()
