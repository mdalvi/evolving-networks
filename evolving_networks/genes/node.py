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


class Node(Gene):
    def __init__(self, node_id, type, bias, response, activation, aggregation, config=None):
        super(Node, self).__init__()
        self.id = node_id
        self.type = type
        self.bias = bias
        self.response = response
        self.activation = activation
        self.aggregation = aggregation
        self.config = config

    def crossover(self, other_node):
        if self.config is None:
            raise InvalidConfigurationError()

        assert self.id == other_node.id  # [1][106,109]
        assert self.type == other_node.type

        bias = self.bias if random.random() < 0.5 else other_node.bias
        response = self.response if random.random() < 0.5 else other_node.response
        activation = self.activation if random.random() < 0.5 else other_node.activation
        aggregation = self.aggregation if random.random() < 0.5 else other_node.aggregation
        node = self.__class__(self.id, self.type, bias, response, activation, aggregation, self.config)
        return node

    def mutate(self):
        if self.config is None:
            raise InvalidConfigurationError()

        bmr = getattr(self.config, 'bias_mutate_rate')
        bms = getattr(self.config, 'bias_mutate_sigma')
        brr = getattr(self.config, 'bias_replace_rate')
        bim = getattr(self.config, 'bias_init_mean')
        bis = getattr(self.config, 'bias_init_sigma')
        bit = getattr(self.config, 'bias_init_type')
        bmin = getattr(self.config, 'bias_min_value')
        bmax = getattr(self.config, 'bias_max_value')

        rmr = getattr(self.config, 'response_mutate_rate')
        rms = getattr(self.config, 'response_mutate_sigma')
        rrr = getattr(self.config, 'response_replace_rate')
        rim = getattr(self.config, 'response_init_mean')
        ris = getattr(self.config, 'response_init_sigma')
        rit = getattr(self.config, 'response_init_type')
        rmin = getattr(self.config, 'response_min_value')
        rmax = getattr(self.config, 'response_max_value')

        act_mr = getattr(self.config, 'activation_mutate_rate')
        act_opt = getattr(self.config, 'activation_options')
        agg_mr = getattr(self.config, 'aggregation_mutate_rate')
        agg_opt = getattr(self.config, 'aggregation_options')

        if self.config.single_structural_mutation:
            mutate_rate = bmr + brr + rmr + rrr + act_mr + agg_mr
            r = random.random()
            if r < bmr / mutate_rate:
                self.bias = self._clamp(self.bias + random_normal(0.0, bms), bmin, bmax)
            elif r < (bmr + brr) / mutate_rate:
                if bit == 'normal':
                    self.bias = self._clamp(random_normal(bim, bis), bmin, bmax)
                elif bit == 'uniform':
                    self.bias = random_uniform(bim, bis, bmin, bmax)
                else:
                    raise InvalidConfigurationError()
            elif r < (bmr + brr + rmr) / mutate_rate:
                self.response = self._clamp(self.response + random_normal(0.0, rms), rmin, rmax)
            elif r < (bmr + brr + rmr + rrr) / mutate_rate:
                if rit == 'normal':
                    self.response = self._clamp(random_normal(rim, ris), rmin, rmax)
                elif rit == 'uniform':
                    self.response = random_uniform(rim, ris, rmin, rmax)
                else:
                    raise InvalidConfigurationError()
            elif r < (bmr + brr + rmr + rrr + act_mr) / mutate_rate:
                # its better to keep act_mr = 0.0 if act_opt has only one choice
                # you waste a probable chance if act_length > 1 is False and act_mr != 0.0
                act_length = len(act_opt)
                if act_length > 1:
                    choices = list(range(act_length))
                    choices.remove(act_opt.index(self.activation))
                    choice_idx = random.choice(choices)
                    self.activation = act_opt[choice_idx]
            else:
                # its better to keep agg_mr = 0.0 if agg_opt has only one choice
                # you waste a probable chance if agg_length > 1 is False and agg_mr != 0.0
                agg_length = len(agg_opt)
                if agg_length > 1:
                    choices = list(range(agg_length))
                    choices.remove(agg_opt.index(self.aggregation))
                    choice_idx = random.choice(choices)
                    self.aggregation = agg_opt[choice_idx]
        else:
            if random.random() < bmr:
                self.bias = self._clamp(self.bias + random_normal(0.0, bms), bmin, bmax)

            if random.random() < brr:
                if bit == 'normal':
                    self.bias = self._clamp(random_normal(bim, bis), bmin, bmax)
                elif bit == 'uniform':
                    self.bias = random_uniform(bim, bis, bmin, bmax)
                else:
                    raise InvalidConfigurationError()

            if random.random() < rmr:
                self.response = self._clamp(self.response + random_normal(0.0, rms), rmin, rmax)

            if random.random() < rrr:
                if rit == 'normal':
                    self.response = self._clamp(random_normal(rim, ris), rmin, rmax)
                elif rit == 'uniform':
                    self.response = random_uniform(rim, ris, rmin, rmax)
                else:
                    raise InvalidConfigurationError()

            if random.random() < act_mr:
                # its better to keep act_mr = 0.0 if act_opt has only one choice
                # you waste a probable chance if act_length > 1 is False and act_mr != 0.0
                act_length = len(act_opt)
                if act_length > 1:
                    choices = list(range(act_length))
                    choices.remove(act_opt.index(self.activation))
                    choice_idx = random.choice(choices)
                    self.activation = act_opt[choice_idx]

            if random.random() < agg_mr:
                # its better to keep agg_mr = 0.0 if agg_opt has only one choice
                # you waste a probable chance if agg_length > 1 is False and agg_mr != 0.0
                agg_length = len(agg_opt)
                if agg_length > 1:
                    choices = list(range(agg_length))
                    choices.remove(agg_opt.index(self.aggregation))
                    choice_idx = random.choice(choices)
                    self.aggregation = agg_opt[choice_idx]
