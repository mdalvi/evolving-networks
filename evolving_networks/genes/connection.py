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


class Connection(Gene):
    def __init__(self, innovation_id, source_id, target_id, weight=None, enabled=None, config=None):
        """
        weight_init_type
        weight_init_mean
        weight_init_sigma
        weight_min_value
        weight_max_value
        enabled_default
        weight_mutate_rate
        weight_replace_rate
        enabled_mutate_rate
        single_structural_mutation
        """
        super(Connection, self).__init__()
        self.config = config
        self.innovation_id = innovation_id
        self.source_id = source_id
        self.target_id = target_id

        if weight is None:
            if self.config.weight_init_type == 'normal':
                self.weight = self._clamp(random_normal(self.config.weight_init_mean, self.config.weight_init_sigma),
                                          self.config.weight_min_value, self.config.weight_max_value)
            elif self.config.weight_init_type == 'uniform':
                self.weight = random_uniform(self.config.weight_init_mean, self.config.weight_init_sigma,
                                             self.config.weight_min_value, self.config.weight_max_value)
            else:
                raise InvalidConfigurationError()
        else:
            self.weight = weight

        if enabled is None:
            self.enabled = self.config.enabled_default
        else:
            self.enabled = enabled

    def crossover(self, other_connection):
        if self.config is None:
            raise InvalidConfigurationError()

        assert self.source_id == other_connection.source_id  # [1][106,109]
        assert self.target_id == other_connection.target_id  # [1][106,109]
        assert self.innovation_id == other_connection.innovation_id  # [1][106,109]

        weight = self.weight if random.random() < 0.5 else other_connection.weight
        enabled = self.enabled if random.random() < 0.5 else other_connection.enabled
        connection = self.__class__(self.innovation_id, self.source_id, self.target_id, weight, enabled, self.config)
        return connection

    def mutate(self):
        if self.config is None:
            raise InvalidConfigurationError()

        wmr = self.config.weight_mutate_rate
        wms = self.config.weight_mutate_sigma
        wrr = self.config.weight_replace_rate
        emr = self.config.enabled_mutate_rate
        wim = self.config.weight_init_mean
        wis = self.config.weight_init_sigma
        wmin = self.config.weight_min_value
        wmax = self.config.weight_max_value

        if self.config.single_structural_mutation:
            mutate_rate = wmr + wrr + emr
            r = random.random()
            if r < wmr / mutate_rate:
                self.weight = self._clamp(self.weight + random_normal(0.0, wms), wmin, wmax)
            elif r < (wmr + wrr) / mutate_rate:
                if self.config.weight_init_type == 'normal':
                    self.weight = self._clamp(random_normal(wim, wis), wmin, wmax)
                elif self.config.weight_init_type == 'uniform':
                    self.weight = random_uniform(wim, wis, wmin, wmax)
                else:
                    raise InvalidConfigurationError()
            else:
                self.enabled = True if self.enabled is False else False
        else:
            if random.random() < wmr:
                self.weight = self._clamp(self.weight + random_normal(0.0, wms), wmin, wmax)

            if random.random() < wrr:
                if self.config.weight_init_type == 'normal':
                    self.weight = self._clamp(random_normal(wim, wis), wmin, wmax)
                elif self.config.weight_init_type == 'uniform':
                    self.weight = random_uniform(wim, wis, wmin, wmax)
                else:
                    raise InvalidConfigurationError()

            if random.random() < emr:
                self.enabled = True if self.enabled is False else False
