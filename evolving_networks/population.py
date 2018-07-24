from evolving_networks.math_util import stat_functions, normalize


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
            raise RuntimeError('Unexpected fitness criterion [{}]'.format(config.neat.fitness_criterion))

        self.population = self.reproduction.populate(config.neat.population_size, self.generation, config)
        self.speciation.speciate(self.population, self.generation, config)

    def fit(self, fitness_function, config, n=None):
        if config.neat.no_fitness_termination and (n is None):
            raise RuntimeError('Cannot have no generational limit with no fitness termination')

        k = 0
        while n is None or k < n:
            k += 1

            fitness_function(list(self.population.items()), config)

            best = None
            damaged_members = []
            population_fitness = []
            for g_id, member in self.population.items():
                if member.is_damaged:
                    damaged_members.append(g_id)
                else:
                    population_fitness.append(member.fitness)
                    if best is None or member.fitness > best.fitness:
                        best = member

            for g_id in damaged_members:
                del self.population[g_id]

            if self.best_genome is None or best.fitness > self.best_genome.fitness:
                self.best_genome = best

            if not config.neat.no_fitness_termination:
                fv = self.fitness_criterion(population_fitness)
                if fv >= config.neat.fitness_threshold:
                    break

            min_fitness, max_fitness = min(population_fitness), max(population_fitness)
            if min_fitness == max_fitness:
                for member in self.population.values():
                    member.adjusted_fitness = 0.0
            else:
                for member in self.population.values():
                    member.adjusted_fitness = normalize(min_fitness, max_fitness, member.fitness, 0.0, 1.0)

            self.speciation.reset_specie_stats()
            self.speciation.sort_specie_genomes()
            self.speciation.calc_best_stats()
            self.speciation.calc_specie_stats(self.generation, config)
            self.population = self.reproduction.reproduce(self.speciation.species, config.neat.population_size,
                                                          self.generation, config)
            print(self.generation, len(self.speciation.species))
            self.speciation.speciate(self.population, self.generation, config)
            self.generation += 1
        print(self.best_genome)
