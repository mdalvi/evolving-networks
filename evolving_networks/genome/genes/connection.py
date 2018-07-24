"""
# ==============
# References
# ==============

[1] http://nn.cs.utexas.edu/downloads/papers/stanley.ec02.pdf

"""
import random

import numpy as np

from evolving_networks.errors import InvalidConfigurationError
from evolving_networks.genome.genes.gene import Gene
from evolving_networks.initializers import random_normal, random_uniform
from evolving_networks.math_util import clamp, normalize


class Connection(Gene):
    def __init__(self, c_id, source_id, target_id, weight, enabled):
        super(Connection, self).__init__()
        self.id = c_id
        self.source_id = source_id
        self.target_id = target_id
        self.weight = weight
        self.enabled = enabled

    def __lt__(self, other):
        return self.id < other.id

    def __le__(self, other):
        return self.id <= other.id

    def __gt__(self, other):
        return self.id > other.id

    def __ge__(self, other):
        return self.id >= other.id

    def initialize(self, config):
        wit = config.weight_init_type
        wim = config.weight_init_mean
        wis = config.weight_init_stdev
        wmin = config.weight_min_value
        wmax = config.weight_max_value

        if wit == 'normal':
            self.weight = clamp(random_normal(wim, wis), wmin, wmax)
        elif wit == 'uniform':
            self.weight = random_uniform(wim, wis, wmin, wmax)
        else:
            raise InvalidConfigurationError()

        self.enabled = config.enabled_default

    def __str__(self):
        attributes = ['id', 'source_id', 'target_id', 'weight', 'enabled']
        attrib = ['{0}={1}'.format(a, getattr(self, a)) for a in attributes]
        return '{0}({1})'.format(self.__class__.__name__, ", ".join(attrib))

    def _distance(self, other_connection, config):
        wmin = config.weight_min_value
        wmax = config.weight_max_value
        wdiff_min, wdiff_max = 0.0, abs(wmin - wmax)
        wdiff = normalize(wdiff_min, wdiff_max, abs(self.weight - other_connection.weight), 0.0, 1.0)
        ediff = 0.0 if self.enabled == other_connection.enabled else 1.0
        return normalize(0.0, 2.0, wdiff + ediff, 0.0, 1.0) * config.compatibility_weight_contribution

    def distance(self, other_connection, config):
        wmin = config.weight_min_value
        wmax = config.weight_max_value
        wdiff_min, wdiff_max = 0.0, abs(wmin - wmax)
        return normalize(wdiff_min, wdiff_max, abs(self.weight - other_connection.weight), 0.0, 1.0)

    def crossover(self, other_connection):
        assert self.id == other_connection.id  # [1][106,109]
        assert self.source_id == other_connection.source_id  # [1][106,109]
        assert self.target_id == other_connection.target_id  # [1][106,109]

        if random.random() < 0.5:
            weight = self.weight
            enabled = self.enabled
        else:
            weight = other_connection.weight
            enabled = other_connection.enabled
        connection = self.__class__(self.id, self.source_id, self.target_id, weight, enabled)
        return connection

    def mutate(self, config):
        wmr = config.weight_mutate_rate
        wms = config.weight_mutate_stdev
        wrr = config.weight_replace_rate
        emr = config.enabled_mutate_rate
        wim = config.weight_init_mean
        wis = config.weight_init_stdev
        wit = config.weight_init_type
        wmin = config.weight_min_value
        wmax = config.weight_max_value

        if config.single_structural_mutation:
            success = False
            mutation_probs = np.array([wmr, wrr, emr])
            while True:
                mutation_probs = mutation_probs / np.sum(mutation_probs)
                mut_idx = np.random.choice(range(3), 1, p=mutation_probs)[0]

                if mut_idx == 0:
                    self.weight = clamp(self.weight + random_normal(0.0, wms), wmin, wmax)
                    success = True
                elif mut_idx == 1:
                    if wit == 'normal':
                        self.weight = clamp(random_normal(wim, wis), wmin, wmax)
                    elif wit == 'uniform':
                        self.weight = random_uniform(wim, wis, wmin, wmax)
                    else:
                        raise InvalidConfigurationError()
                    success = True
                else:
                    self.enabled = True if self.enabled is False else False
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

            if random.random() < emr:
                self.enabled = True if self.enabled is False else False
