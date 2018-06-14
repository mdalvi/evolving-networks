def build_essential_dict(node_keys, connections):
    essentials = {}
    for node_key in node_keys:
        e = [[node_key]]
        while True:
            essential = set(
                source_id for (source_id, target_id) in connections if target_id in e[-1] and source_id not in e[-1])
            if not essential:
                break
            e.append(list(essential))
        essentials[node_key] = e
    return essentials


def activation_recursion(essentials_dict, key, activation_path):
    value_list = essentials_dict[key]
    if len(value_list) == 1:
        activation_path.append(value_list[0][0])
    else:
        for value_set in reversed(value_list):
            for value_key in value_set:
                if key == value_key:
                    activation_path.append(value_key)
                    break
                activation_recursion(essentials_dict, value_key, activation_path)
