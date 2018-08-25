"""
# ==============
# References
# ==============

[1] https://github.com/CodeReclaimers/neat-python/blob/master/neat/graphs.py
[2] http://nn.cs.utexas.edu/downloads/papers/stanley.ec02.pdf

"""

import copy
import random

import numpy as np

from evolving_networks.math_util import probabilistic_round
from evolving_networks.pytorch.genome.genes.weight import Weight as WeightGene


class Genome(object):
    def __init__(self, g_id, generation, config):
        self.id = g_id
        self.config = config
        self.birth_generation = generation

        self.fitness = 0.0
        self.adjusted_fitness = 0.0
        self.is_damaged = False

        self.structured_weights = []
        self.unstructured_weights = []

    @property
    def complexity(self):
        return len(self.unstructured_weights)

    def __lt__(self, other):
        if self.fitness == other.fitness:
            return self.birth_generation < other.birth_generation
        return self.fitness < other.fitness

    def __le__(self, other):
        if self.fitness == other.fitness:
            return self.birth_generation <= other.birth_generation
        return self.fitness <= other.fitness

    def __gt__(self, other):
        if self.fitness == other.fitness:
            return self.birth_generation > other.birth_generation
        return self.fitness > other.fitness

    def __ge__(self, other):
        if self.fitness == other.fitness:
            return self.birth_generation >= other.birth_generation
        return self.fitness >= other.fitness

    def __str__(self):
        s = "Id: {0}\nFitness: {1}".format(self.id, self.fitness)
        s += "\nWeights:"
        for weight in self.unstructured_weights:
            s += "\n\t{0!s}".format(weight)
        return s

    def distance(self, other_genome, config):
        dist = 0.0
        c1 = config.genome.compatibility_weight_contribution

        c_dist = 0.0
        for w_id in range(len(self.unstructured_weights)):
            c_dist += self.unstructured_weights[w_id].distance(other_genome.unstructured_weights[w_id],
                                                               config.connection)
        c_dist = c_dist / len(self.unstructured_weights)

        dist += (c1 * c_dist)
        assert 0.0 <= dist <= 1.0
        return dist

    def mutate(self, config):

        mutate_weights = []
        nb_mutate_weights = probabilistic_round(random.random() * len(self.unstructured_weights))

        if nb_mutate_weights > 0 and len(self.unstructured_weights) > 0:
            mutate_weights = np.random.choice(range(len(self.unstructured_weights)), nb_mutate_weights, replace=False)

        for w_id in mutate_weights:
            self.unstructured_weights[w_id].mutate(config.connection)

        self._find_and_replace(self.structured_weights, self.unstructured_weights, 0)

    def crossover_sexual(self, parent_1, parent_2):
        for w_idx in range(len(parent_1.unstructured_weights)):
            w1 = parent_1.unstructured_weights[w_idx]
            w2 = parent_2.unstructured_weights[w_idx]
            new_w = w1.crossover(w2)
            assert new_w.id == len(self.unstructured_weights)
            self.unstructured_weights.append(new_w)
        self._find_and_replace(self.structured_weights, self.unstructured_weights, 0)

    def crossover_asexual(self, parent):
        self.structured_weights = copy.deepcopy(parent.structured_weights)
        self.unstructured_weights = [WeightGene(wg.id, wg.weight) for wg in parent.unstructured_weights]

    def clone(self, parent):
        self.structured_weights = copy.deepcopy(parent.structured_weights)
        self.unstructured_weights = [WeightGene(wg.id, wg.weight) for wg in parent.unstructured_weights]

    def initialize(self, model_class):
        model = model_class()
        self.structured_weights = [param.data.numpy().tolist() for param in model.parameters()]
        self._flatten(self.structured_weights, self.unstructured_weights)
        self.unstructured_weights = [WeightGene(w_id, weight) for w_id, weight in enumerate(self.unstructured_weights)]

    def _flatten(self, weights, flattened):
        for w in weights:
            if type(w) == list:
                self._flatten(w, flattened)
            else:
                flattened.append(w)

    def _find_and_replace(self, weights, replacements, replace_idx=0):
        for i, l in enumerate(weights):
            if type(l) == list:
                replace_idx = self._find_and_replace(l, replacements, replace_idx)
            else:
                weights[i] = replacements[replace_idx].weight
                replace_idx += 1
        return replace_idx
