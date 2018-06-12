from evolving_networks.math_util import mean
from evolving_networks.speciation.traditional import Traditional as TraditionalSpeciation


class Population(object):
    def __init__(self, reproduction):
        self.reproduction = reproduction
        self.generation = 0
        self.population = None
        self._speciation = None
        self.best_genome = None
        self.fitness_criterion = None

    def initialize(self, config):
        self.generation = 0
        self.best_genome = None

        if config.neat.fitness_criterion == 'max':
            self.fitness_criterion = max
        elif config.neat.fitness_criterion == 'min':
            self.fitness_criterion = min
        elif config.neat.fitness_criterion == 'mean':
            self.fitness_criterion = mean
        elif not config.neat.no_fitness_termination:
            raise RuntimeError('UNEXPECTED FITNESS CRITERION [{}]'.format(config.neat.fitness_criterion))

        self.population = self.reproduction.initialize_population(config.neat.population_size, config)
        self._speciation = TraditionalSpeciation()
        self._speciation.speciate(self.population, self.generation, config)

    def fit(self, fitness_function, config, n=None):
        if config.neat.no_fitness_termination and (n is None):
            raise RuntimeError('CANNOT HAVE NO GENERATIONAL LIMIT WITH NO FITNESS TERMINATION')

        k = 0
        while n is None or k < n:
            k += 1

            fitness_function(list(self.population.items()), config)

            best = None
            for genome in self.population.values():
                if best is None or genome.fitness > best.fitness:
                    best = genome

            if not config.neat.no_fitness_termination:
                fv = self.fitness_criterion(genome.fitness for genome in self.population.values())
                if fv >= config.neat.fitness_threshold:
                    break

            self.population = self.reproduction.reproduce(self.config, self.species,
                                                          self.config.pop_size, self.generation)
