from evolving_networks.math_util import stat_functions


class Population(object):
    def __init__(self, reproduction, speciation):
        self.reproduction = reproduction
        self.speciation = speciation
        self.generation = 0
        self.population = None
        self.best_genome = None
        self.fitness_criterion = None

    def initialize(self, config):
        self.generation = 0
        self.best_genome = None

        self.fitness_criterion = stat_functions.get(config.neat.fitness_criterion)
        if self.fitness_criterion is None and not config.neat.no_fitness_termination:
            raise RuntimeError('UNEXPECTED FITNESS CRITERION [{}]'.format(config.neat.fitness_criterion))

        self.population = self.reproduction.populate(config.neat.population_size, config)
        self.speciation.speciate(self.population, self.generation, config)

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

            self.population = self.reproduction.reproduce(self.speciation.species, config.neat.population_size,
                                                          self.generation, config)
