from evolving_networks.math_util import stat_functions, mean


class Species(object):
    def __init__(self, specie_id, generation, config):
        self.specie_id = specie_id
        self.created = generation
        self.last_improved = generation

        self.members = None
        self.representative = None
        self.fitness_history = []
        self.is_stagnant = False
        self.adjusted_fitness = None
        self.fitness_criterion = stat_functions.get(config.fitness_criterion)

    @property
    def fitness(self):
        return self.fitness_criterion([genome.fitness for genome in self.members.values()])

    @property
    def fitness_mean(self):
        return mean([genome.fitness for genome in self.members.values()])
