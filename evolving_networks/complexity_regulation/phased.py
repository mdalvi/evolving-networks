"""
# ==============
# References
# ==============

[1] http://sharpneat.sourceforge.net/phasedsearch.html

"""
from evolving_networks.complexity_regulation.factory import Factory
from evolving_networks.math_util import mean


class Phased(Factory):  # [1]
    def __init__(self, config):
        super(Phased, self).__init__()
        self.mode = 'complexifying'
        self.config = config
        self.node_add_rate = config.genome.node_add_rate
        self.conn_add_rate = config.genome.conn_add_rate
        self.node_delete_rate = 0.0
        self.conn_delete_rate = 0.0
        self.off_spring_asexual_rate = config.species.off_spring_asexual_rate

        self.complexity_type = config.neat.phased_complexity_type
        self.complexity_threshold = config.neat.phased_complexity_threshold

        assert self.complexity_type in ['absolute', 'relative']
        if self.complexity_type == 'relative':
            self.complexity_ceiling = -1.0
        else:
            self.complexity_ceiling = self.complexity_threshold

        self.last_transition = 0
        self.min_fitness_plateau = config.neat.phase_min_fitness_plateau
        self.min_simplification_generations = config.neat.phase_min_simplification_generations

    def determine_mode(self, statistics):
        if self.mode == 'complexifying':
            if self.complexity_ceiling == -1.0:
                self.complexity_ceiling = statistics.mean_complexity[-1] + self.complexity_threshold

            condition_1 = statistics.mean_complexity[-1] > self.complexity_ceiling
            condition_2 = statistics.fitness_plateau_size >= self.min_fitness_plateau
            if condition_1 and condition_2:
                self.mode = 'simplifying'
                self.last_transition = statistics.generation

                self.node_add_rate = 0.0
                self.conn_add_rate = 0.0
                self.node_delete_rate = self.config.genome.node_delete_rate
                self.conn_delete_rate = self.config.genome.conn_delete_rate
                self.off_spring_asexual_rate = 0.0

        elif self.mode == 'simplifying':
            condition_1 = statistics.generation - self.last_transition > self.min_simplification_generations
            condition_2 = statistics.mean_complexity[-1] < self.complexity_ceiling
            condition_3 = mean(statistics.mean_complexity[-100:]) - statistics.mean_complexity_ma >= 0.0
            if condition_1 and condition_2 and condition_3:
                self.mode = 'complexifying'
                self.last_transition = statistics.generation

                if self.complexity_type == 'relative':
                    self.complexity_ceiling = statistics.mean_complexity[-1] + self.complexity_threshold

                self.node_add_rate = self.config.genome.node_add_rate
                self.conn_add_rate = self.config.genome.conn_add_rate
                self.node_delete_rate = 0.0
                self.conn_delete_rate = 0.0
                self.off_spring_asexual_rate = self.config.species.off_spring_asexual_rate
