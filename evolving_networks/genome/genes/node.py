"""
# ==============
# References
# ==============

[1] http://nn.cs.utexas.edu/downloads/papers/stanley.ec02.pdf

"""
import json
import random

import numpy as np

from evolving_networks.errors import InvalidConfigurationError
from evolving_networks.genome.genes.gene import Gene
from evolving_networks.initializers import random_normal, random_uniform
from evolving_networks.math_util import clamp


class Node(Gene):
    __params = ['id', 'type', 'bias', 'res', 'act', 'agg']

    def __init__(self):
        super(Node, self).__init__()
        self.id = None
        self.type = None
        self.bias = None
        self.res = None
        self.act = None
        self.agg = None

    def __str__(self):
        attributes = ['id', 'type', 'bias', 'res', 'act', 'agg']
        attrib = ['{0}={1}'.format(a, getattr(self, a)) for a in attributes]
        return '{0}({1})'.format(self.__class__.__name__, ", ".join(attrib))

    def __eq__(self, other):
        for p in self.__params:
            if getattr(self, p) != getattr(other, p):
                return False
        return True

    def to_json(self):
        result = dict()
        for p in self.__params:
            result[p] = getattr(self, p)
        return json.dumps(result)

    def from_json(self, node_json):
        result = json.loads(node_json)
        for p in self.__params:
            setattr(self, p, result[p])
        return self

    def initialize(self, _id, _type, bias, res, act, agg, config=None):
        self.id = _id
        self.type = _type
        self.bias = bias
        self.res = res
        self.act = act
        self.agg = agg

        if config is not None:
            bim = config.bias_init_mean
            bis = config.bias_init_stdev
            bit = config.bias_init_type
            bmin = config.bias_min_value
            bmax = config.bias_max_value

            rim = config.response_init_mean
            ris = config.response_init_stdev
            rit = config.response_init_type
            rmin = config.response_min_value
            rmax = config.response_max_value

            if self.type == 'input':
                self.bias = 0.0
                self.res = 1.0
                self.agg = 'sum'
                self.act = 'identity'
            else:
                if bit == 'normal':
                    self.bias = clamp(random_normal(bim, bis), bmin, bmax)
                elif bit == 'uniform':
                    self.bias = random_uniform(bim, bis, bmin, bmax)
                else:
                    raise InvalidConfigurationError()

                if rit == 'normal':
                    self.res = clamp(random_normal(rim, ris), rmin, rmax)
                elif rit == 'uniform':
                    self.res = random_uniform(rim, ris, rmin, rmax)
                else:
                    raise InvalidConfigurationError()

                self.agg = config.aggregation_default
                self.act = config.activation_default_output if self.type == 'output' else config.activation_default

    def crossover(self, other_node):
        assert self.id == other_node.id  # [1][106,109]
        assert self.type == other_node.type

        if random.random() < 0.5:
            bias = self.bias
            res = self.res
            act = self.act
            agg = self.agg
        else:
            bias = other_node.bias
            res = other_node.res
            act = other_node.act
            agg = other_node.agg
        node = self.__class__()
        node.initialize(self.id, self.type, bias, res, act, agg)
        return node

    def mutate(self, config):

        assert (self.type != 'input')

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
                    self.res = clamp(self.res + random_normal(0.0, rms), rmin, rmax)
                    success = True
                elif mut_idx == 3:
                    if rit == 'normal':
                        self.res = clamp(random_normal(rim, ris), rmin, rmax)
                    elif rit == 'uniform':
                        self.res = random_uniform(rim, ris, rmin, rmax)
                    else:
                        raise InvalidConfigurationError()
                    success = True
                elif mut_idx == 4:
                    nb_activations = len(act_opt)
                    if self.type != 'input' and self.type != 'output' and nb_activations > 1:
                        choices = list(range(nb_activations))
                        choices.remove(act_opt.index(self.act))
                        choice_idx = random.choice(choices)
                        self.act = act_opt[choice_idx]
                        success = True
                else:
                    nb_aggregations = len(agg_opt)
                    if nb_aggregations > 1:
                        choices = list(range(nb_aggregations))
                        choices.remove(agg_opt.index(self.agg))
                        choice_idx = random.choice(choices)
                        self.agg = agg_opt[choice_idx]
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
                self.res = clamp(self.res + random_normal(0.0, rms), rmin, rmax)

            if random.random() < rrr:
                if rit == 'normal':
                    self.res = clamp(random_normal(rim, ris), rmin, rmax)
                elif rit == 'uniform':
                    self.res = random_uniform(rim, ris, rmin, rmax)
                else:
                    raise InvalidConfigurationError()

            if random.random() < act_mr:
                nb_activations = len(act_opt)
                if self.type != 'input' and self.type != 'output' and nb_activations > 1:
                    choices = list(range(nb_activations))
                    choices.remove(act_opt.index(self.act))
                    choice_idx = random.choice(choices)
                    self.act = act_opt[choice_idx]

            if random.random() < agg_mr:
                nb_aggregations = len(agg_opt)
                if nb_aggregations > 1:
                    choices = list(range(nb_aggregations))
                    choices.remove(agg_opt.index(self.agg))
                    choice_idx = random.choice(choices)
                    self.agg = agg_opt[choice_idx]
