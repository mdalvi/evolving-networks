from itertools import count


class Factory(object):
    def __init__(self):
        self._genome_indexer = count(0)

    def populate(self, population_size, generation, config):
        raise NotImplementedError()

    def reproduce(self, species, generation, regulation, population_size, config):
        pass
