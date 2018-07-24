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


class Node(Gene):
    def __init__(self, n_id, n_type, bias, response, activation, aggregation):
        super(Node, self).__init__()
        self.id = n_id
        self.type = n_type
        self.bias = bias
        self.response = response
        self.activation = activation
        self.aggregation = aggregation

    def __str__(self):
        attributes = ['id', 'type', 'bias', 'response', 'activation', 'aggregation']
        attrib = ['{0}={1}'.format(a, getattr(self, a)) for a in attributes]
        return '{0}({1})'.format(self.__class__.__name__, ", ".join(attrib))

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

        if random.random() < 0.5:
            bias = self.bias
            response = self.response
            activation = self.activation
            aggregation = self.aggregation
        else:
            bias = other_node.bias
            response = other_node.response
            activation = other_node.activation
            aggregation = other_node.aggregation
        node = self.__class__(self.id, self.type, bias, response, activation, aggregation)
        return node

    def mutate(self, config):
        bmr = config.bias_mutate_rate
        bms = config.bias_mutate_stdev
        brr = config.bias_replace_rate
        bim = config.bias_init_mean
        bis = config.bias_init_stdev
        bit = config.bias_init_type
        bmin = config.bias_min_value
        bmax = config.bias_max_value

        rmr = config.response_mutate_rate
        rms = config.response_mutate_stdev
        rrr = config.response_replace_rate
        rim = config.response_init_mean
        ris = config.response_init_stdev
        rit = config.response_init_type
        rmin = config.response_min_value
        rmax = config.response_max_value

        act_mr = config.activation_mutate_rate
        act_opt = config.activation_options
        agg_mr = config.aggregation_mutate_rate
        agg_opt = config.aggregation_options

        if config.single_structural_mutation:
            success = False
            mutation_probs = np.array([bmr, brr, rmr, rrr, act_mr, agg_mr])
            while True:
                mutation_probs = mutation_probs / np.sum(mutation_probs)
                mut_idx = np.random.choice(range(6), 1, p=mutation_probs)[0]

                if mut_idx == 0:
                    self.bias = clamp(self.bias + random_normal(0.0, bms), bmin, bmax)
                    success = True
                elif mut_idx == 1:
                    if bit == 'normal':
                        self.bias = clamp(random_normal(bim, bis), bmin, bmax)
                    elif bit == 'uniform':
                        self.bias = random_uniform(bim, bis, bmin, bmax)
                    else:
                        raise InvalidConfigurationError()
                    success = True
                elif mut_idx == 2:
                    self.response = clamp(self.response + random_normal(0.0, rms), rmin, rmax)
                    success = True
                elif mut_idx == 3:
                    if rit == 'normal':
                        self.response = clamp(random_normal(rim, ris), rmin, rmax)
                    elif rit == 'uniform':
                        self.response = random_uniform(rim, ris, rmin, rmax)
                    else:
                        raise InvalidConfigurationError()
                    success = True
                elif mut_idx == 4:
                    nb_activations = len(act_opt)
                    if self.type != 'output' and nb_activations > 1:
                        choices = list(range(nb_activations))
                        choices.remove(act_opt.index(self.activation))
                        choice_idx = random.choice(choices)
                        self.activation = act_opt[choice_idx]
                        success = True
                else:
                    nb_aggregations = len(agg_opt)
                    if nb_aggregations > 1:
                        choices = list(range(nb_aggregations))
                        choices.remove(agg_opt.index(self.aggregation))
                        choice_idx = random.choice(choices)
                        self.aggregation = agg_opt[choice_idx]
                        success = True

                if success is True:
                    break

                mutation_probs[mut_idx] = 0.0
                if np.sum(mutation_probs) == 0.0:
                    break
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
                nb_activations = len(act_opt)
                if nb_activations > 1:
                    choices = list(range(nb_activations))
                    choices.remove(act_opt.index(self.activation))
                    choice_idx = random.choice(choices)
                    self.activation = act_opt[choice_idx]

            if random.random() < agg_mr:
                nb_aggregations = len(agg_opt)
                if nb_aggregations > 1:
                    choices = list(range(nb_aggregations))
                    choices.remove(agg_opt.index(self.aggregation))
                    choice_idx = random.choice(choices)
                    self.aggregation = agg_opt[choice_idx]
