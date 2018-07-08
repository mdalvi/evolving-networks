from evolving_networks.math_util import stat_functions


class SpeciesStatistics(object):
    def __init__(self):
        self.fitness = 0.0
        self.fitness_mean = 0.0
        self.adjusted_fitness = 0.0
        self.target_size_float = 0.0
        self.target_size = 0
        self.elite_size = 0
        self.off_spring_count = 0
        self.off_spring_asexual_count = 0
        self.off_spring_sexual_count = 0


class Species(object):
    def __init__(self, specie_id, generation, config):
        self.specie_id = specie_id
        self.created = generation
        self.last_improved = generation
        self.fitness_criterion = stat_functions.get(config.fitness_criterion)

        self.members = []
        self.representative = None
        self.fitness_history = []
