from itertools import count


class Factory(object):
    def __init__(self):
        self._specie_indexer = count(0)

    def reset_specie_stats(self):
        raise NotImplementedError()

    def sort_specie_genomes(self):
        raise NotImplementedError()

    def calc_best_stats(self):
        raise NotImplementedError()

    def _purge_stagnant_species(self, generation, config):
        raise NotImplementedError()

    def calc_specie_stats(self, generation, complexity_regulation, config):
        raise NotImplementedError()

    def speciate(self, population, generation, config):
        raise NotImplementedError()

    def get_species_id(self, genome_id):
        raise NotImplementedError()

    def get_species(self, genome_id):
        raise NotImplementedError()
