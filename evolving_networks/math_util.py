import math
import random


def variance(values):
    values = list(values)
    m = mean(values)
    return sum((v - m) ** 2 for v in values) / len(values)


def stdev(values):
    return math.sqrt(variance(values))


def mean(values):
    values = list(values)
    return sum(values) / len(values)


def clamp(value, min_value, max_val):
    return max(min(value, max_val), min_value)


def normalize(act_min, act_max, val, norm_min=-1.0, norm_max=1.0):
    return ((val - act_min) / (act_max - act_min)) * (norm_max - norm_min) + norm_min


def probabilistic_round(value):
    integer_part = math.floor(value)
    fractional_part = value - integer_part
    return int(integer_part + 1.0 if random.random() < fractional_part else integer_part)


stat_functions = {'min': min, 'max': max, 'mean': mean, 'stdev': stdev}
