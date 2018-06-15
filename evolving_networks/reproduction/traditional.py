from evolving_networks.genome.genome import Genome
from evolving_networks.reproduction.factory import Factory


class Traditional(Factory):
    def __init__(self):
        super(Traditional, self).__init__()
        self.ancestors = set()

    def populate(self, population_size, config):
        population = {}
        for _ in range(population_size):
            genome_id = next(self._genome_indexer)
            assert genome_id not in self.ancestors
            g = Genome(genome_id, config.genome)
            g.initialize(config.node, config.connection)
            population[genome_id] = g
            self.ancestors.add(genome_id)
        return population

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

        nb_non_stagnants = len(species_data)
        for (specie_id, specie, _) in species_data:
            is_stagnant = False
            stagnant_time = generation - specie.last_improved

            if nb_non_stagnants > config.species.elitism:
                is_stagnant = stagnant_time >= config.species.max_stagnation

            if is_stagnant:
                nb_non_stagnants -= 1

            specie.is_stagnant = is_stagnant

        print('stagnation successful')

