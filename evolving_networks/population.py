from evolving_networks.species import SpeciesSet


class Population(object):
    def __init__(self, reproduction):
        self.reproduction = reproduction
        self.population = dict()
        self.species_set = None
        self.generation = 0

    def initialize(self, config):
        self.population = self.reproduction.init_population(config.neat.population_size, config)
        self.species_set = SpeciesSet(config)



    def fit(self):
        pass
