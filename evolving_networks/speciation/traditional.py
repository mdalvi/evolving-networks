import random

from evolving_networks.math_util import mean, probabilistic_round
from evolving_networks.speciation.factory import Factory
from evolving_networks.speciation.helpers import genomic_distance
from evolving_networks.speciation.species import Species, SpeciesStatistics


class Traditional(Factory):
    def __init__(self):
        super(Traditional, self).__init__()
        self.species = {}
        self._genome_to_species = {}
        self.best_genome = None
        self.best_fitness = float('-Infinity')
        self.best_specie_idx = None
        self.min_specie_size = float('+Infinity')
        self.max_specie_size = float('-Infinity')

    def sort_specie_genomes(self):
        min_specie_size, max_specie_size = float('+Infinity'), float('-Infinity')
        for specie in self.species.values():
            sorted(specie.members, reverse=True)
            min_specie_size = min(min_specie_size, len(specie.members))
            max_specie_size = max(max_specie_size, len(specie.members))

        self.min_specie_size = min_specie_size
        self.max_specie_size = max_specie_size

    def calc_best_stats(self):

        best_genome = None
        best_fitness = float('-Infinity')
        best_specie_idx = None

        for specie_id, specie in self.species.items():
            if specie.members[0].fitness > best_fitness:
                best_genome = specie.members[0]
                best_fitness = specie.members[0].fitness
                best_specie_idx = specie_id

        self.best_genome = best_genome
        self.best_fitness = best_fitness
        self.best_specie_idx = best_specie_idx

    def calc_specie_stats(self, config):
        specie_stats = {}
        total_mean_fitness = 0.0
        total_target_size = 0

        for specie_id, specie in self.species.items():
            members_fitness = [member.fitness for member in specie.members.values()]
            stats = SpeciesStatistics()
            stats.fitness = specie.fitness_criterion(members_fitness)
            stats.fitness_mean = mean(members_fitness)
            specie_stats[specie_id] = stats

            total_mean_fitness += stats.fitness_mean

        if total_mean_fitness == 0.0:
            target_size_float = config.neat.population_size / len(self.species)
            for specie_id in self.species.keys():
                specie_stats[specie_id].target_size_float = target_size_float
                specie_stats[specie_id].target_size = probabilistic_round(target_size_float)
                total_target_size += specie_stats[specie_id].target_size
        else:
            for specie_id in self.species.keys():
                specie_stats[specie_id].target_size_float = (specie_stats[
                                                                 specie_id].fitness_mean / total_mean_fitness) * config.neat.population_size
                specie_stats[specie_id].target_size = probabilistic_round(specie_stats[specie_id].target_size_float)
                total_target_size += specie_stats[specie_id].target_size

        target_size_delta = total_target_size - config.neat.population_size
        if target_size_delta < 0:
            pass
        elif target_size_delta > 0:
            pass

    def speciate(self, population, generation, config):

        representatives, members = {}, {}

        # Speciate entire population
        unspeciated = set(population.keys())
        compatibility_threshold = config.species.compatibility_threshold

        # Uniform chance of electing fresh new representative
        species_election = [s_id for s_id in self.species.keys()]

        while len(species_election) > 0:
            specie_id = random.choice(species_election)
            specie = self.species[specie_id]

            specie_distances = []
            for genome_id in unspeciated:
                genome = population[genome_id]
                d = genomic_distance(specie.representative, genome, config)
                specie_distances.append((d, genome))

            _, new_representative = min(specie_distances, key=lambda x: x[0])
            new_representative_id = new_representative.id
            representatives[specie_id] = new_representative_id
            members[specie_id] = [new_representative_id]
            unspeciated.remove(new_representative_id)

            species_election.remove(specie_id)

        while unspeciated:
            genome_id = unspeciated.pop()
            genome = population[genome_id]

            specie_distances = []
            for specie_id, representative_id in representatives.items():
                representative = population[representative_id]
                d = genomic_distance(representative, genome, config)
                if d < compatibility_threshold:
                    specie_distances.append((d, specie_id))

            if specie_distances:
                _, specie_id = min(specie_distances, key=lambda x: x[0])
                members[specie_id].append(genome_id)
            else:
                specie_id = next(self._specie_indexer)
                representatives[specie_id] = genome_id
                members[specie_id] = [genome_id]

        self._genome_to_species = {}
        for specie_id, representative_id in representatives.items():
            s = self.species.get(specie_id)
            if s is None:
                s = Species(specie_id, generation, config.species)

            specie_members = []
            for genome_id in members[specie_id]:
                self._genome_to_species[genome_id] = specie_id
                specie_members.append(population[genome_id])

            s.representative = population[representative_id]
            s.members = specie_members
            self.species[specie_id] = s

    def get_species_id(self, genome_id):
        return self._genome_to_species[genome_id]

    def get_species(self, genome_id):
        specie_id = self._genome_to_species[genome_id]
        return self.species[specie_id]
