from functools import reduce
from operator import mul

from evolving_networks.math_util import mean


def product_aggregation(x):
    return reduce(mul, x, 1.0)


def max_abs_aggregation(x):
    return max(x, key=abs)


def min_abs_aggregation(x):
    return min(x, key=abs)


def sum_aggregation(x):
    return sum(x)


def max_aggregation(x):
    return max(x)


def min_aggregation(x):
    return min(x)


def mean_aggregation(x):
    return mean(x)


class InvalidAggregationError(TypeError):
    pass


class AggregationFunctionSet(object):
    def __init__(self):
        self.functions = {}
        self.add('product', product_aggregation)
        self.add('sum', sum_aggregation)
        self.add('max', max_aggregation)
        self.add('min', min_aggregation)
        self.add('maxabs', max_abs_aggregation)
        self.add('minabs', min_abs_aggregation)
        self.add('mean', mean_aggregation)

    def add(self, name, func):
        self.functions[name] = func

    def get(self, name):
        f = self.functions.get(name)
        if f is None:
            raise InvalidAggregationError("NO SUCH AGGREGATION FUNCTION FOUND [{}]".format(name))
        return f

    def is_valid(self, name):
        return name in self.functions
