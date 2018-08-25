class Phenome(object):
    def __init__(self):
        pass

    def initialize(self, model_class):
        raise NotImplementedError()

    def activate(self, inputs):
        raise NotImplementedError()

    def reset(self):
        raise NotImplementedError()
