class Gene(object):
    def __init__(self):
        pass

    def crossover(self, other_gene):
        raise NotImplementedError()

    def mutate(self):
        raise NotImplementedError()

    def _clamp(self, value, min_value, max_val):
        return max(min(value, max_val), min_value)
