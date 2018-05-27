import random


def random_normal(mean, sigma):
    return random.gauss(mean, sigma)


def random_uniform(mean, sigma, min_val, max_val):
    return random.uniform(max(min_val, (mean - (2 * sigma))), min(max_val, (mean + (2 * sigma))))
