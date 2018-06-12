from itertools import count

from evolving_networks.genome import Genome


class Reproduction(object):
    def __init__(self):
        self.ancestors = set()
        self._genome_indexer = count(0)

    def initialize_population(self, population_size, config):
        population = dict()
        for _ in range(population_size):
            g_id = next(self._genome_indexer)
            assert g_id not in self.ancestors
            g = Genome(g_id, config.genome)
            g.initialize(config.node, config.connection)
            population[g_id] = g
            self.ancestors.add(g_id)

        return population
