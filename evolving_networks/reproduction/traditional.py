from evolving_networks.genome.genome import Genome
from evolving_networks.math_util import normalize
from evolving_networks.reproduction.factory import Factory


class Traditional(Factory):
    def __init__(self):
        super(Traditional, self).__init__()
        self.ancestors = set()

    def populate(self, population_size, generation, config):
        population = {}
        for _ in range(population_size):
            genome_id = next(self._genome_indexer)
            assert genome_id not in self.ancestors
            g = Genome(genome_id, generation, config.genome)
            g.initialize(config.node, config.connection)
            population[genome_id] = g
            self.ancestors.add(genome_id)
        return population

    def reproduce_off_springs(self, species):
        pass

    def reproduce(self, species, population_size, generation, config):
        species_data = []
        for specie_id, specie in species.items():
            if specie.fitness_history:
                historical_best_fitness = max(specie.fitness_history)
            else:
                historical_best_fitness = float('-Infinity')

            specie_fitness = specie.fitness
            specie.fitness_history.append(specie_fitness)
            specie.adjusted_fitness = None
            if specie_fitness > historical_best_fitness:
                specie.last_improved = generation

            species_data.append((specie_id, specie, specie_fitness))

        # Sort in ascending fitness order.
        species_data.sort(key=lambda x: x[2])

        nb_members = []
        non_stagnant_fitness = []
        nb_non_stagnants = len(species_data)
        for (specie_id, specie, _) in species_data:
            is_stagnant = False
            stagnant_time = generation - specie.last_improved

            if nb_non_stagnants > config.species.elitism:
                is_stagnant = stagnant_time >= config.species.max_stagnation

            if is_stagnant:
                nb_non_stagnants -= 1

            specie.is_stagnant = is_stagnant

            if not is_stagnant:
                nb_members.append(len(specie.members))
                non_stagnant_fitness.extend(member.fitness for member in specie.members.values())

        assert nb_non_stagnants >= config.species.elitism

        if not non_stagnant_fitness:
            species = {}
            return {}

        min_fitness = min(non_stagnant_fitness)
        max_fitness = max(non_stagnant_fitness)

        if min_fitness == max_fitness:
            min_fitness, max_fitness = 0.0, 1.0

        adjusted_fitness = []
        for (specie_id, specie, _) in species_data:
            specie_fitness_mean = specie.fitness_mean
            specie.adjusted_fitness = normalize(min_fitness, max_fitness, specie_fitness_mean, 0.0, 1.0)


        min_species_size = max(config.species.min_species_size, config.reproduction.elitism)
        # spawn_amounts = self.compute_spawn(adjusted_fitness, nb_members, population_size, min_species_size)