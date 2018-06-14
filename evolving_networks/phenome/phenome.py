class Phenome(object):
    def __init__(self):
        pass

    def _create(self, genome, config):
        raise NotImplementedError()

    def activate(self):
        raise NotImplementedError()

    def reset(self):
        raise NotImplementedError()
