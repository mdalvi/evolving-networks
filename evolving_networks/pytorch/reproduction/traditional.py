"""
# ==============
# References
# ==============

[1] https://stackoverflow.com/questions/19286657/index-all-except-one-item-in-python

"""

import numpy as np

from evolving_networks.math_util import probabilistic_round
from evolving_networks.pytorch.genome.genome import Genome
from evolving_networks.pytorch.reproduction.factory import Factory


class Traditional(Factory):
    def __init__(self, model_class):
        super(Traditional, self).__init__()
        self.ancestors = set()
        self.model_class = model_class

    def populate(self, population_size, generation, config):
        population = {}
        for _ in range(population_size):
            member_id = next(self._genome_indexer)
            assert member_id not in self.ancestors
            g = Genome(member_id, generation, config.genome)
            g.initialize(self.model_class)
            population[member_id] = g
            self.ancestors.add(member_id)
        return population

    def _reproduce_off_springs(self, species, generation, config):
        off_springs = []
        non_zero_species = 0
        reproduce_probs, species_probs = {}, []
        for idx, (s_id, specie) in enumerate(species.items()):
            if specie.survivors != 0:
                non_zero_species += 1

            members_fitness_sum = sum(specie.members_fitness[:specie.survivors])
            if members_fitness_sum == 0.0:
                p = np.array([1.0] * specie.survivors) / specie.survivors
                reproduce_probs[s_id] = p.tolist()
            else:
                p = np.array(specie.members_fitness[:specie.survivors]) / members_fitness_sum
                reproduce_probs[s_id] = p.tolist()

            species_probs.append(specie.survivors)

        assert non_zero_species != 0
        species_probs_sum = sum(species_probs)
        assert species_probs_sum != 0
        species_probs = np.array(species_probs) / species_probs_sum

        for s_idx, (s_id, specie) in enumerate(species.items()):
            for _ in range(specie.off_spring_asexual):
                member_parent_1 = np.random.choice(specie.members[:specie.survivors], 1, p=reproduce_probs[s_id])[0]
                member_id = next(self._genome_indexer)
                assert member_id not in self.ancestors
                g = Genome(member_id, generation, config.genome)
                g.crossover_asexual(member_parent_1)
                g.mutate(config)
                off_springs.append(g)

            inter_species_matings = 0 if non_zero_species == 1 else probabilistic_round(
                specie.off_spring_sexual * config.species.inter_species_mating_rate)
            intra_species_matings = specie.off_spring_sexual - inter_species_matings

            for _ in range(inter_species_matings):
                s_id_1 = s_id
                member_parent_1 = np.random.choice(species[s_id_1].members[:species[s_id_1].survivors], 1,
                                                   p=reproduce_probs[s_id_1])[0]

                species_probs_revised = np.array([0.0 if i == s_idx else p for i, p in enumerate(species_probs)])
                species_probs_revised = species_probs_revised / np.sum(species_probs_revised)
                s_id_2 = np.random.choice(list(species.keys()), 1, p=species_probs_revised)[0]

                member_parent_2 = np.random.choice(species[s_id_2].members[:species[s_id_2].survivors], 1,
                                                   p=reproduce_probs[s_id_2])[0]

                member_id = next(self._genome_indexer)
                assert member_id not in self.ancestors
                g = Genome(member_id, generation, config.genome)
                g.crossover_sexual(member_parent_1, member_parent_2)
                off_springs.append(g)

            for _ in range(intra_species_matings):
                if specie.survivors == 1:
                    member_parent_1 = np.random.choice(specie.members[:specie.survivors], 1, p=reproduce_probs[s_id])[0]
                    member_id = next(self._genome_indexer)
                    assert member_id not in self.ancestors
                    g = Genome(member_id, generation, config.genome)
                    g.crossover_asexual(member_parent_1)
                    g.mutate(config)
                    off_springs.append(g)
                else:
                    m1_idx = np.random.choice(range(specie.survivors), 1, p=reproduce_probs[s_id])[0]
                    member_parent_1 = specie.members[m1_idx]
                    reproduce_probs_revised = np.array(
                        [0.0 if i == m1_idx else p for i, p in enumerate(reproduce_probs[s_id])])
                    reproduce_probs_revised_sum = np.sum(reproduce_probs_revised)
                    if reproduce_probs_revised_sum == 0.0:
                        member_id = next(self._genome_indexer)
                        assert member_id not in self.ancestors
                        g = Genome(member_id, generation, config.genome)
                        g.crossover_asexual(member_parent_1)
                        g.mutate(config)
                        off_springs.append(g)
                    else:
                        reproduce_probs_revised = reproduce_probs_revised / reproduce_probs_revised_sum
                        m2_idx = np.random.choice(range(specie.survivors), 1, p=reproduce_probs_revised)[0]
                        member_parent_2 = specie.members[m2_idx]
                        member_id = next(self._genome_indexer)
                        assert member_id not in self.ancestors
                        g = Genome(member_id, generation, config.genome)
                        g.crossover_sexual(member_parent_1, member_parent_2)
                        off_springs.append(g)

        return off_springs

    def reproduce(self, species, generation, regulation, population_size, config):
        new_population = {}
        off_springs = self._reproduce_off_springs(species, generation, config)

        for specie in species.values():
            for elite in specie.members[:specie.elites]:
                g = Genome(elite.id, elite.birth_generation, config.genome)
                g.clone(elite)
                new_population[elite.id] = g

        for off_spring in off_springs:
            new_population[off_spring.id] = off_spring

        assert len(new_population) == population_size
        return new_population
