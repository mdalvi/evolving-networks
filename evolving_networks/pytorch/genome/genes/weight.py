"""
# ==============
# References
# ==============

[1] http://nn.cs.utexas.edu/downloads/papers/stanley.ec02.pdf

"""
import random

import numpy as np

from evolving_networks.errors import InvalidConfigurationError
from evolving_networks.initializers import random_normal, random_uniform
from evolving_networks.math_util import clamp, normalize
from evolving_networks.pytorch.genome.genes.gene import Gene


class Weight(Gene):
    def __init__(self, w_id, weight):
        super(Weight, self).__init__()
        self.id = w_id
        self.weight = weight

    def __str__(self):
        attributes = ['id', 'weight']
        attrib = ['{0}={1}'.format(a, getattr(self, a)) for a in attributes]
        return '{0}({1})'.format(self.__class__.__name__, ", ".join(attrib))

    def distance(self, other_weight, config):
        wmin = config.weight_min_value
        wmax = config.weight_max_value
        wdiff_min, wdiff_max = 0.0, abs(wmin - wmax)
        return normalize(wdiff_min, wdiff_max, abs(self.weight - other_weight.weight), 0.0, 1.0)

    def crossover(self, other_weight):
        assert self.id == other_weight.id  # [1][106,109]
        weight = self.weight if random.random() < 0.5 else other_weight.weight
        weight_gene = self.__class__(self.id, weight)
        return weight_gene

    def mutate(self, config):
        wmr = config.weight_mutate_rate
        wms = config.weight_mutate_stdev
        wrr = config.weight_replace_rate
        wim = config.weight_init_mean
        wis = config.weight_init_stdev
        wit = config.weight_init_type
        wmin = config.weight_min_value
        wmax = config.weight_max_value

        if config.single_structural_mutation:
            success = False
            mutation_probs = np.array([wmr, wrr])
            while True:
                mutation_probs = mutation_probs / np.sum(mutation_probs)
                mut_idx = np.random.choice(range(2), 1, p=mutation_probs)[0]

                if mut_idx == 0:
                    self.weight = clamp(self.weight + random_normal(0.0, wms), wmin, wmax)
                    success = True
                else:
                    if wit == 'normal':
                        self.weight = clamp(random_normal(wim, wis), wmin, wmax)
                    elif wit == 'uniform':
                        self.weight = random_uniform(wim, wis, wmin, wmax)
                    else:
                        raise InvalidConfigurationError()
                    success = True

                if success is True:
                    break

                mutation_probs[mut_idx] = 0.0
                if np.sum(mutation_probs) == 0.0:
                    break
        else:
            if random.random() < wmr:
                self.weight = clamp(self.weight + random_normal(0.0, wms), wmin, wmax)

            if random.random() < wrr:
                if wit == 'normal':
                    self.weight = clamp(random_normal(wim, wis), wmin, wmax)
                elif wit == 'uniform':
                    self.weight = random_uniform(wim, wis, wmin, wmax)
                else:
                    raise InvalidConfigurationError()
