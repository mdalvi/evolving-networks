from functools import reduce
from operator import mul

from evolving_networks.errors import InvalidAggregationError
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


class AggregationFunctionSet(object):
    def __init__(self):
        self.functions = {}
        self._add('product', product_aggregation)
        self._add('sum', sum_aggregation)
        self._add('max', max_aggregation)
        self._add('min', min_aggregation)
        self._add('maxabs', max_abs_aggregation)
        self._add('minabs', min_abs_aggregation)
        self._add('mean', mean_aggregation)

    def _add(self, name, func):
        self.functions[name] = func

    def get(self, name):
        f = self.functions.get(name)
        if f is None:
            raise InvalidAggregationError("NO SUCH AGGREGATION FUNCTION FOUND [{}]".format(name))
        return f

    def is_valid(self, name):
        return name in self.functions
