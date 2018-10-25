import time

from tabulate import tabulate

from evolving_networks.math_util import stat_functions, normalize, mean


class Statistics(object):
    def __init__(self):
        self.generation = 0
        self.current_best = None
        self.best_fitness = float('-Infinity')

        self.max_fitness = []
        self.mean_fitness = []
        self.stdev_fitness = []
        self.max_complexity = []
        self.mean_complexity = []
        self.stdev_complexity = []
        self.mean_species_fitness = []
        self.stdev_species_fitness = []
        self.elapsed_generation_time = []

    @property
    def mean_complexity_ma(self):
        if len(self.mean_complexity) == 0:
            return 0.0
        return mean(self.mean_complexity[-100:])

    def describe_stats(self, speciation, regulation, members_complexity, generation):
        self.mean_complexity.append(mean(members_complexity))
        self.generation = generation

        species_details = {'Id': [], 'Size': [], 'Fitness': [], 'AdjFitness': []}
        for s_id, specie in speciation.species.items():
            species_details['Id'].append(s_id)
            species_details['Size'].append(len(specie))
            species_details['Fitness'].append(specie.fitness)
            species_details['AdjFitness'].append(specie.adjusted_fitness)

        if speciation.best_genome.fitness > self.best_fitness:
            self.best_fitness = speciation.best_genome.fitness
        self.current_best = speciation.best_genome

        print("\nTotal number of species {0}".format(len(speciation.species)))
        print("Best fitness {0}".format(self.best_fitness))
        print("Current best fitness {0} @ complexity {1}".format(self.current_best.fitness,
                                                                 self.current_best.complexity))

        print(tabulate(species_details, headers="keys", numalign="right"))
        print(str(regulation))


class Population(object):
    def __init__(self, reproduction, speciation, regulation, reporter):
        self.reproduction = reproduction
        self.speciation = speciation
        self.regulation = regulation
        self.reporter = reporter

        self.config = None
        self.generation = 0
        self.population = None
        self.best_genome = None
        self.population_size = 0
        self.fitness_function = None
        self.fitness_criterion = None

        self.statistics = Statistics()

    def initialize(self, fitness_function, config):
        self.config = config
        self.population_size = config.neat.population_size
        self.fitness_function = fitness_function
        self.fitness_criterion = stat_functions.get(config.neat.fitness_criterion)
        if self.fitness_criterion is None and not config.neat.no_fitness_termination:
            raise RuntimeError('Unexpected fitness criterion [{}]'.format(config.neat.fitness_criterion))

        self.population = self.reproduction.populate(self.population_size, self.generation, config)
        fitness_function(list(self.population.items()), config)

        damaged_members = []
        members_fitness = []
        for g_id, member in self.population.items():
            if member.is_damaged:
                damaged_members.append(g_id)
            else:
                members_fitness.append(member.fitness)

        for g_id in damaged_members:
            del self.population[g_id]

        min_fitness, max_fitness = min(members_fitness), max(members_fitness)
        if min_fitness == max_fitness:
            for member in self.population.values():
                member.adjusted_fitness = 0.0
        else:
            for member in self.population.values():
                member.adjusted_fitness = normalize(min_fitness, max_fitness, member.fitness, 0.0, 1.0)

        self.speciation.speciate(self.population, self.generation, config)
        self.speciation.reset_specie_stats()
        self.speciation.sort_specie_genomes()
        self.speciation.calc_best_stats()
        self.speciation.calc_specie_stats(self.generation, self.population_size, self.config)
        self.generation += 1

    def fit(self, n=None):
        if self.config.neat.no_fitness_termination and (n is None):
            raise RuntimeError('Cannot have no generational limit with no fitness termination')

        k = 0
        while n is None or k < n:
            k += 1
            t0 = time.time()
            self.reporter.start_generation(self.generation)
            self.reporter.pre_reproduction()
            self.population = self.reproduction.reproduce(species=self.speciation.species,
                                                          generation=self.generation, regulation=self.regulation,
                                                          population_size=self.population_size, config=self.config)
            self.reporter.post_reproduction()
            self.reporter.pre_evaluation()
            self.fitness_function(list(self.population.items()), self.config)
            self.reporter.post_evaluation()

            best = None
            damaged_members = []
            members_fitness = []
            members_complexity = []
            for g_id, member in self.population.items():
                if member.is_damaged:
                    damaged_members.append(g_id)
                else:
                    members_fitness.append(member.fitness)
                    members_complexity.append(member.complexity)
                    if best is None or member.fitness > best.fitness:
                        best = member

            for g_id in damaged_members:
                del self.population[g_id]

            if self.best_genome is None or best.fitness > self.best_genome.fitness:
                self.best_genome = best

            min_fitness, max_fitness = min(members_fitness), max(members_fitness)
            if min_fitness == max_fitness:
                for member in self.population.values():
                    member.adjusted_fitness = 0.0
            else:
                for member in self.population.values():
                    member.adjusted_fitness = normalize(min_fitness, max_fitness, member.fitness, 0.0, 1.0)

            self.reporter.pre_speciation()
            self.speciation.speciate(self.population, self.generation, self.config)
            self.speciation.reset_specie_stats()
            self.speciation.sort_specie_genomes()
            self.speciation.calc_best_stats()
            self.speciation.calc_specie_stats(self.generation, self.population_size, self.config)
            self.reporter.post_speciation()

            self.statistics.describe_stats(self.speciation, self.regulation, members_complexity, self.generation)

            if not self.config.neat.no_fitness_termination:
                fv = self.fitness_criterion(members_fitness)
                if fv >= self.config.neat.fitness_threshold:
                    break

            self.regulation.determine_mode(self.statistics)
            self.reporter.end_generation()
            self.generation += 1
        return self.statistics
