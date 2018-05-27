"""
# ==============
# References
# ==============

[1] http://nn.cs.utexas.edu/downloads/papers/stanley.ec02.pdf

"""

from evolving_networks.errors import InvalidConfigurationError
from evolving_networks.genes.gene import Gene


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
        bmin = getattr(self.config, 'bias_min_value')
        bmax = getattr(self.config, 'bias_max_value')

        rmr = getattr(self.config, 'response_mutate_rate')
        rms = getattr(self.config, 'response_mutate_sigma')
        rrr = getattr(self.config, 'response_replace_rate')
        rim = getattr(self.config, 'response_init_mean')
        ris = getattr(self.config, 'response_init_sigma')
        rmin = getattr(self.config, 'response_min_value')
        rmax = getattr(self.config, 'response_max_value')

        act_mr = getattr(self.config, 'activation_mutate_rate')
        agg_mr = getattr(self.config, 'aggregation_mutate_rate')

        if self.config.single_structural_mutation:
            mutate_rate  = bmr + brr + rmr + rrr + act_mr + agg_mr

        else:
            pass
