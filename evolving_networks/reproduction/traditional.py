from evolving_networks.genome import Genome
from evolving_networks.reproduction.factory import Factory


class Traditional(Factory):
    def __init__(self):
        super(Factory, self).__init__()
        self.ancestors = set()

    def populate(self, population_size, config):
        population = {}
        for _ in range(population_size):
            genome_id = next(self._genome_indexer)
            assert genome_id not in self.ancestors
            g = Genome(genome_id, config.genome)
            g.initialize(config.node, config.connection)
            population[genome_id] = g
            self.ancestors.add(genome_id)
        return population
