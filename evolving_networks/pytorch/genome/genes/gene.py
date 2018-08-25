class Gene(object):
    def __init__(self):
        pass

    def crossover(self, other_gene):
        raise NotImplementedError()

    def mutate(self, config):
        raise NotImplementedError()

    def distance(self, other, config):
        raise NotImplementedError()
