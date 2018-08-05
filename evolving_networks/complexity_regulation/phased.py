"""
# ==============
# References
# ==============

[1] http://sharpneat.sourceforge.net/phasedsearch.html

"""
from evolving_networks.complexity_regulation.factory import Factory


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
        self.complexity_ceiling = -1.0 if self.complexity_type == 'relative' else self.complexity_threshold

        self.last_transition = 0
        self.fitness_plateau = 0
        self.best_fitness = float('-Infinity')
        self.fitness_plateau_threshold = config.neat.phase_fitness_plateau_threshold
        self.simplification_generations_threshold = config.neat.phase_simplification_generations_threshold

    def determine_mode(self, statistics):
        if self.mode == 'complexifying':

            if statistics.current_best.fitness > self.best_fitness:
                self.best_fitness = statistics.current_best.fitness
                self.fitness_plateau = 0
            else:
                self.fitness_plateau += 1

            if self.complexity_ceiling == -1.0:
                self.complexity_ceiling = statistics.mean_complexity[-1] + self.complexity_threshold

            condition_1 = statistics.mean_complexity[-1] > self.complexity_ceiling
            condition_2 = self.fitness_plateau >= self.fitness_plateau_threshold
            if condition_1 and condition_2:
                self.mode = 'simplifying'
                self.last_transition = statistics.generation

                self.node_add_rate = 0.0
                self.conn_add_rate = 0.0
                self.node_delete_rate = self.config.genome.node_delete_rate
                self.conn_delete_rate = self.config.genome.conn_delete_rate
                # self.off_spring_asexual_rate =  self.config.species.off_spring_asexual_rate
                self.fitness_plateau = 0

        elif self.mode == 'simplifying':
            condition_1 = statistics.generation - self.last_transition > self.simplification_generations_threshold
            condition_2 = statistics.mean_complexity[-1] < self.complexity_ceiling
            # condition_3 = mean(statistics.mean_complexity[-100:]) - statistics.mean_complexity_ma >= 0.0
            if condition_1 and condition_2:
                self.mode = 'complexifying'
                self.last_transition = statistics.generation

                if self.complexity_type == 'relative':
                    self.complexity_ceiling = statistics.mean_complexity[-1] + self.complexity_threshold

                self.node_add_rate = self.config.genome.node_add_rate
                self.conn_add_rate = self.config.genome.conn_add_rate
                self.node_delete_rate = 0.0
                self.conn_delete_rate = 0.0
                # self.off_spring_asexual_rate = self.config.species.off_spring_asexual_rate
                self.fitness_plateau = 0

    def __str__(self):
        return "{0} phased complexity regulation {1} for ceiling {2} with fitness plateau {3}".format(
            self.complexity_type.title(), self.mode, self.complexity_ceiling, self.fitness_plateau)
