"""
# ==============
# References
# ==============

[1] http://nn.cs.utexas.edu/downloads/papers/stanley.ec02.pdf

"""
import random

from evolving_networks.errors import InvalidConfigurationError
from evolving_networks.genome.genes.gene import Gene
from evolving_networks.initializers import random_normal, random_uniform
from evolving_networks.math_util import clamp, normalize


class Node(Gene):
    def __init__(self, n_id, n_type, bias, response, activation, aggregation):
        super(Node, self).__init__()
        self.id = n_id
        self.type = n_type
        self.bias = bias
        self.response = response
        self.activation = activation
        self.aggregation = aggregation

    def initialize(self, config):
        bim = getattr(config, 'bias_init_mean')
        bis = getattr(config, 'bias_init_stdev')
        bit = getattr(config, 'bias_init_type')
        bmin = getattr(config, 'bias_min_value')
        bmax = getattr(config, 'bias_max_value')

        rim = getattr(config, 'response_init_mean')
        ris = getattr(config, 'response_init_stdev')
        rit = getattr(config, 'response_init_type')
        rmin = getattr(config, 'response_min_value')
        rmax = getattr(config, 'response_max_value')

        if bit == 'normal':
            self.bias = clamp(random_normal(bim, bis), bmin, bmax)
        elif bit == 'uniform':
            self.bias = random_uniform(bim, bis, bmin, bmax)
        else:
            raise InvalidConfigurationError()

        if rit == 'normal':
            self.response = clamp(random_normal(rim, ris), rmin, rmax)
        elif rit == 'uniform':
            self.response = random_uniform(rim, ris, rmin, rmax)
        else:
            raise InvalidConfigurationError()

        self.aggregation = config.aggregation_default
        self.activation = config.activation_default_output if self.type == 'output' else config.activation_default

    def distance(self, other_node, config):
        bmin = getattr(config, 'bias_min_value')
        bmax = getattr(config, 'bias_max_value')
        rmin = getattr(config, 'response_min_value')
        rmax = getattr(config, 'response_max_value')
        bdiff_min, bdiff_max = 0.0, abs(bmin - bmax)
        rdiff_min, rdiff_max = 0.0, abs(rmin - rmax)
        biff = normalize(bdiff_min, bdiff_max, abs(self.bias - other_node.bias), 0.0, 1.0)
        rdiff = normalize(rdiff_min, rdiff_max, abs(self.response - other_node.response), 0.0, 1.0)
        actdiff = 0.0 if self.activation == other_node.activation else 1.0
        aggdiff = 0.0 if self.aggregation == other_node.aggregation else 1.0
        return normalize(0.0, 4.0, biff + rdiff + actdiff + aggdiff, 0.0,
                         1.0) * config.compatibility_weight_contribution

    def crossover(self, other_node):
        assert self.id == other_node.id  # [1][106,109]
        assert self.type == other_node.type

        bias = self.bias if random.random() < 0.5 else other_node.bias
        response = self.response if random.random() < 0.5 else other_node.response
        activation = self.activation if random.random() < 0.5 else other_node.activation
        aggregation = self.aggregation if random.random() < 0.5 else other_node.aggregation
        node = self.__class__(self.id, self.type, bias, response, activation, aggregation)
        return node

    def mutate(self, config):
        bmr = getattr(config, 'bias_mutate_rate')
        bms = getattr(config, 'bias_mutate_stdev')
        brr = getattr(config, 'bias_replace_rate')
        bim = getattr(config, 'bias_init_mean')
        bis = getattr(config, 'bias_init_stdev')
        bit = getattr(config, 'bias_init_type')
        bmin = getattr(config, 'bias_min_value')
        bmax = getattr(config, 'bias_max_value')

        rmr = getattr(config, 'response_mutate_rate')
        rms = getattr(config, 'response_mutate_stdev')
        rrr = getattr(config, 'response_replace_rate')
        rim = getattr(config, 'response_init_mean')
        ris = getattr(config, 'response_init_stdev')
        rit = getattr(config, 'response_init_type')
        rmin = getattr(config, 'response_min_value')
        rmax = getattr(config, 'response_max_value')

        act_mr = getattr(config, 'activation_mutate_rate')
        act_opt = getattr(config, 'activation_options')
        agg_mr = getattr(config, 'aggregation_mutate_rate')
        agg_opt = getattr(config, 'aggregation_options')

        if config.single_structural_mutation:
            mutate_rate = bmr + brr + rmr + rrr + act_mr + agg_mr
            r = random.random()
            if r < bmr / mutate_rate:
                self.bias = clamp(self.bias + random_normal(0.0, bms), bmin, bmax)
            elif r < (bmr + brr) / mutate_rate:
                if bit == 'normal':
                    self.bias = clamp(random_normal(bim, bis), bmin, bmax)
                elif bit == 'uniform':
                    self.bias = random_uniform(bim, bis, bmin, bmax)
                else:
                    raise InvalidConfigurationError()
            elif r < (bmr + brr + rmr) / mutate_rate:
                self.response = clamp(self.response + random_normal(0.0, rms), rmin, rmax)
            elif r < (bmr + brr + rmr + rrr) / mutate_rate:
                if rit == 'normal':
                    self.response = clamp(random_normal(rim, ris), rmin, rmax)
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
                self.bias = clamp(self.bias + random_normal(0.0, bms), bmin, bmax)

            if random.random() < brr:
                if bit == 'normal':
                    self.bias = clamp(random_normal(bim, bis), bmin, bmax)
                elif bit == 'uniform':
                    self.bias = random_uniform(bim, bis, bmin, bmax)
                else:
                    raise InvalidConfigurationError()

            if random.random() < rmr:
                self.response = clamp(self.response + random_normal(0.0, rms), rmin, rmax)

            if random.random() < rrr:
                if rit == 'normal':
                    self.response = clamp(random_normal(rim, ris), rmin, rmax)
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
