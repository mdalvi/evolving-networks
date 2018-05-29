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

    def crossover(self, other_gene):
        if self.config is None:
            raise InvalidConfigurationError()

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
        agg_mr = getattr(self.config, 'aggregation_mutate_rate')

        if self.config.single_structural_mutation:

            if self.config.single_structural_mutation:
                mutate_rate = bmr + brr + rmr + rrr + act_mr + agg_mr
                r = random.random()
                if r < bmr / mutate_rate:
                    self.bias = self._clamp(self.bias + random_normal(bim, bis), bmin, bmax)
                elif r < (bmr + brr) / mutate_rate:
                    if bit == 'normal':
                        self.bias = self._clamp(random_normal(bim, bis), bmin, bmax)
                    elif bit == 'uniform':
                        self.bias = random_uniform(bim, bis, bmin, bmax)
                    else:
                        raise InvalidConfigurationError()
                elif r < (bmr + brr + rmr):
                    self.response = self._clamp(self.response + random_normal(rim, ris), rmin, rmax)
                elif r < (bmr + brr + rmr + rrr):
                    if rit == 'normal':
                        self.response = self._clamp(random_normal(rim, ris), rmin, rmax)
                    elif rit == 'uniform':
                        self.response = random_uniform(rim, ris, rmin, rmax)
                    else:
                        raise InvalidConfigurationError()
                elif r < (bmr + brr + rmr + rrr + act_mr):
                    
            else:
                pass
        else:
            pass
