from itertools import count


class Factory(object):
    def __init__(self):
        self._genome_indexer = count(0)

    def populate(self, population_size, config):
        raise NotImplementedError()

    def reproduce(self, species, population_size, generation, config):
        pass

    def _stagnation(self, species, generation):
        pass
