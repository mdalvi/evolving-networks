from evolving_networks.math_util import stat_functions, normalize, mean


class Statistics(object):
    def __init__(self):
        self.generation = 0
        self.max_fitness = []
        self.mean_fitness = []
        self.max_complexity = []
        self.mean_complexity = []
        self.mean_species_best_fitness = []
        self.mean_complexity_ma = 0.0
        self.fitness_plateau_size = 0


class Population(object):
    def __init__(self, reproduction, speciation, complexity_regulation):
        self.reproduction = reproduction
        self.speciation = speciation
        self.complexity_regulation = complexity_regulation

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
        self.generation += 1

    def fit(self, n=None):
        if self.config.neat.no_fitness_termination and (n is None):
            raise RuntimeError('Cannot have no generational limit with no fitness termination')

        k = 0
        while n is None or k < n:
            k += 1

            print(self.complexity_regulation.mode)
            self.speciation.calc_specie_stats(self.generation, self.complexity_regulation, self.config)
            self.population = self.reproduction.reproduce(self.speciation.species, self.complexity_regulation,
                                                          self.generation, self.population_size, self.config)
            self.fitness_function(list(self.population.items()), self.config)

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

            min_fitness, max_fitness = min(members_fitness), max(members_fitness)
            if min_fitness == max_fitness:
                for member in self.population.values():
                    member.adjusted_fitness = 0.0
            else:
                for member in self.population.values():
                    member.adjusted_fitness = normalize(min_fitness, max_fitness, member.fitness, 0.0, 1.0)

            if self.best_genome is None or best.fitness > self.best_genome.fitness:
                self.best_genome = best
                self.statistics.fitness_plateau_size = 0
            else:
                self.statistics.fitness_plateau_size += 1

            self.speciation.speciate(self.population, self.generation, self.config)
            self.speciation.reset_specie_stats()
            self.speciation.sort_specie_genomes()
            self.speciation.calc_best_stats()

            self.statistics.max_fitness.append(max(members_fitness))
            self.statistics.mean_fitness.append(mean(members_fitness))
            self.statistics.max_complexity.append(max(members_complexity))
            if len(self.statistics.mean_complexity) != 0:
                self.statistics.mean_complexity_ma = mean(self.statistics.mean_complexity[-100:])
            self.statistics.mean_complexity.append(mean(members_complexity))
            species_best_fitness = []
            for specie in self.speciation.species.values():
                species_best_fitness.append(specie.members[0].fitness)
            self.statistics.mean_species_best_fitness.append(mean(species_best_fitness))

            if not self.config.neat.no_fitness_termination:
                fv = self.fitness_criterion(members_fitness)
                if fv >= self.config.neat.fitness_threshold:
                    break

            self.complexity_regulation.determine_mode(self.statistics)
            self.generation += 1
        return self.statistics
