from evolving_networks.math_util import stat_functions


class Species(object):
    def __init__(self, specie_id, generation, config):
        self.specie_id = specie_id
        self.created = generation
        self.last_improved = generation
        self.fitness_criterion = stat_functions.get(config.fitness_criterion)

        self.members = []
        self.representative = None
        self.fitness_history = []

        self.members_fitness = []
        self.fitness = 0.0
        self.adjusted_fitness = 0.0
        self.adjusted_fitness_mean = 0.0

        self.target_size_float = 0.0
        self.target_size = 0
        self.elites = 0
        self.off_springs = 0
        self.off_spring_asexual = 0
        self.off_spring_sexual = 0
        self.survivors = 0

    def reset_stats(self):
        self.members_fitness = []
        self.fitness = 0.0
        self.adjusted_fitness = 0.0
        self.adjusted_fitness_mean = 0.0

        self.target_size_float = 0.0
        self.target_size = 0
        self.elites = 0
        self.off_springs = 0
        self.off_spring_asexual = 0
        self.off_spring_sexual = 0
        self.survivors = 0

    def __len__(self):
        return len(self.members)
