class Phenome(object):
    def __init__(self):
        pass

    def initialize(self, act_func_set, agg_func_set):
        raise NotImplementedError()

    def activate(self, inputs):
        raise NotImplementedError()

    def reset(self):
        raise NotImplementedError()
