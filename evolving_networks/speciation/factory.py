from itertools import count


class Factory(object):
    def __init__(self):
        self._specie_indexer = count(0)

    def speciate(self, population, generation, config):
        raise NotImplementedError()

    def get_species_id(self, genome_id):
        raise NotImplementedError()

    def get_species(self, genome_id):
        raise NotImplementedError()
