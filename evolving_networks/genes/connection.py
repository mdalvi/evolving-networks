"""
# ==============
# References
# ==============

[1] http://nn.cs.utexas.edu/downloads/papers/stanley.ec02.pdf

"""
import random

from evolving_networks.errors import InvalidConfigurationError
from evolving_networks.genes.gene import Gene
from evolving_networks.initializers import random_normal, random_uniform
from evolving_networks.math_util import clamp, normalize


class Connection(Gene):
    def __init__(self, c_id, source_id, target_id, weight=None, enabled=None):
        super(Connection, self).__init__()
        self.id = c_id
        self.source_id = source_id
        self.target_id = target_id
        self.weight = weight
        self.enabled = enabled

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

    def distance(self, other_connection, config):
        wmin = config.weight_min_value
        wmax = config.weight_max_value
        wdiff_min, wdiff_max = 0.0, abs(wmin - wmax)
        wdiff = normalize(wdiff_min, wdiff_max, abs(self.weight - other_connection.weight), 0.0, 1.0)
        ediff = 0.0 if self.enabled == other_connection.enabled else 1.0
        return normalize(0.0, 2.0, wdiff + ediff, 0.0, 1.0) * config.compatibility_weight_contribution

    def crossover(self, other_connection):
        assert self.id == other_connection.id  # [1][106,109]
        assert self.source_id == other_connection.source_id  # [1][106,109]
        assert self.target_id == other_connection.target_id  # [1][106,109]

        weight = self.weight if random.random() < 0.5 else other_connection.weight
        enabled = self.enabled if random.random() < 0.5 else other_connection.enabled
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
            mutate_rate = wmr + wrr + emr
            r = random.random()
            if r < wmr / mutate_rate:
                self.weight = clamp(self.weight + random_normal(0.0, wms), wmin, wmax)
            elif r < (wmr + wrr) / mutate_rate:
                if wit == 'normal':
                    self.weight = clamp(random_normal(wim, wis), wmin, wmax)
                elif wit == 'uniform':
                    self.weight = random_uniform(wim, wis, wmin, wmax)
                else:
                    raise InvalidConfigurationError()
            else:
                self.enabled = True if self.enabled is False else False
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
