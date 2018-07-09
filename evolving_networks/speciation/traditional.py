import random

import numpy as np

from evolving_networks.math_util import mean, probabilistic_round
from evolving_networks.speciation.factory import Factory
from evolving_networks.speciation.helpers import genomic_distance
from evolving_networks.speciation.species import Species


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

    def reset_specie_stats(self):
        for specie in self.species.values():
            specie.reset_stats()

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

        for s_id, specie in self.species.items():
            if specie.members[0].fitness > best_fitness:
                best_genome = specie.members[0]
                best_fitness = specie.members[0].fitness
                best_specie_idx = s_id

        self.best_genome = best_genome
        self.best_fitness = best_fitness
        self.best_specie_idx = best_specie_idx

    def calc_specie_stats(self, config):
        target_size_sum = 0
        mean_fitness_sum = 0.0

        for specie in self.species.values():
            members_fitness = [member.adjusted_fitness for member in specie.members.values()]
            specie.fitness = specie.fitness_criterion(members_fitness)
            specie.fitness_mean = mean(members_fitness)

            mean_fitness_sum += specie.fitness_mean

        if mean_fitness_sum == 0.0:
            target_size_float = config.neat.population_size / len(self.species)
            for specie in self.species.values():
                specie.target_size_float = target_size_float
                specie.target_size = probabilistic_round(target_size_float)
                target_size_sum += specie.target_size
        else:
            for s_id, specie in self.species.items():
                specie.target_size_float = (specie.fitness_mean / mean_fitness_sum) * config.neat.population_size
                target_size = probabilistic_round(specie.target_size_float)
                if target_size == 0 and s_id == self.best_specie_idx:
                    target_size = 1
                specie.target_size = target_size
                target_size_sum += specie.target_size

        target_size_delta = target_size_sum - config.neat.population_size
        if target_size_delta < 0:
            if target_size_delta == -1:
                self.species[self.best_specie_idx].target_size += 1
            else:
                specie_idxs = list(self.species.keys())
                for _ in range(target_size_delta * -1):
                    probabilities = np.array([max(0.0, specie.target_size_float - specie.target_size) for specie in
                                              self.species.values()])
                    probabilities = probabilities / np.sum(probabilities)
                    self.species[np.random.choice(specie_idxs, 1, p=probabilities)].target_size += 1
        elif target_size_delta > 0:
            specie_idxs = list(self.species.keys())
            i = 0
            while i < target_size_delta:
                probabilities = np.array([max(0.0, specie.target_size - specie.target_size_float) for specie in
                                          self.species.keys()])
                probabilities = probabilities / np.sum(probabilities)
                specie_idx = np.random.choice(specie_idxs, 1, p=probabilities)
                if self.species[specie_idx].target_size != 0:
                    if not (specie_idx == self.best_specie_idx and self.species[specie_idx].target_size == 1):
                        self.species[specie_idx].target_size -= 1
                        i += 1

        for s_id, specie in self.species.items():
            if specie.target_size == 0:
                specie.elite_size = 0
                continue

            elite_size = probabilistic_round(len(specie.members) * config.species.elitism)
            specie.elite_size = min(elite_size, specie.target_size)

            if s_id == self.best_specie_idx and elite_size == 0:
                specie.elite_size = 1

            specie.off_springs = specie.target_size - specie.elite_size
            specie.off_spring_asexual = probabilistic_round(specie.off_springs * config.species.off_spring_asexual)
            specie.off_spring_sexual = specie.off_springs - specie.off_spring_asexual

            specie.selection_size = probabilistic_round(len(specie.members) * config.species.selection)

    def speciate(self, population, generation, config):

        representatives, members = {}, {}

        # Speciate entire population
        unspeciated = set(population.keys())
        compatibility_threshold = config.species.compatibility_threshold

        # Uniform chance of electing fresh new representative
        species_election = [s_id for s_id in self.species.keys()]

        while len(species_election) > 0:
            s_id = random.choice(species_election)
            specie = self.species[s_id]

            specie_distances = []
            for genome_id in unspeciated:
                genome = population[genome_id]
                d = genomic_distance(specie.representative, genome, config)
                specie_distances.append((d, genome))

            _, new_representative = min(specie_distances, key=lambda x: x[0])
            new_representative_id = new_representative.id
            representatives[s_id] = new_representative_id
            members[s_id] = [new_representative_id]
            unspeciated.remove(new_representative_id)

            species_election.remove(s_id)

        while unspeciated:
            genome_id = unspeciated.pop()
            genome = population[genome_id]

            specie_distances = []
            for s_id, representative_id in representatives.items():
                representative = population[representative_id]
                d = genomic_distance(representative, genome, config)
                if d < compatibility_threshold:
                    specie_distances.append((d, s_id))

            if specie_distances:
                _, s_id = min(specie_distances, key=lambda x: x[0])
                members[s_id].append(genome_id)
            else:
                s_id = next(self._specie_indexer)
                representatives[s_id] = genome_id
                members[s_id] = [genome_id]

        self._genome_to_species = {}
        for s_id, representative_id in representatives.items():
            s = self.species.get(s_id)
            if s is None:
                s = Species(s_id, generation, config.species)

            specie_members = []
            for genome_id in members[s_id]:
                self._genome_to_species[genome_id] = s_id
                specie_members.append(population[genome_id])

            s.representative = population[representative_id]
            s.members = specie_members
            self.species[s_id] = s

    def get_species_id(self, genome_id):
        return self._genome_to_species[genome_id]

    def get_species(self, genome_id):
        s_id = self._genome_to_species[genome_id]
        return self.species[s_id]
