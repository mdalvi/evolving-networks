def mean(values):
    values = list(values)
    return sum(values) / len(values)


def clamp(value, min_value, max_val):
    return max(min(value, max_val), min_value)


def normalize(act_min, act_max, val, norm_min=-1.0, norm_max=1.0):
    return ((val - act_min) / (act_max - act_min)) * (norm_max - norm_min) + norm_min
