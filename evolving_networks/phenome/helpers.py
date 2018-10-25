def calc_required_acyclic_depth(node_ids_all, enabled_connections):
    """
    Calculates the required dependency depth for every node in network
    :return: Node dependency dictionary e.g: {0: [[0]], 1: [[1]], 2: [[2], [0, 1]], 3: [[3], [0, 1, 2]]}
    """
    required = {}
    for n_id in node_ids_all:
        depth_list = [[n_id]]
        while True:
            essential = set(
                source_id for (_, source_id, target_id) in enabled_connections if
                target_id in depth_list[-1] and source_id not in depth_list[-1])
            if not essential:
                break
            depth_list.append(list(essential))
        required[n_id] = depth_list
    return required


def calc_neural_acyclic_path(depth, n_id, path):
    """
    The function calculates exact order of neuron activations for specified n_id (output id)
    :return: Ordered activation list e.g: [0, 1, 2, 3]
    """
    depth_list = depth[n_id]

    if len(depth_list) == 1:
        # We have reached the end of dependency chain
        if depth_list[0][0] not in path:
            path.append(depth_list[0][0])
    else:
        for depth_set in reversed(depth_list):
            for depth_id in depth_set:
                if n_id == depth_id:
                    if depth_id not in path:
                        path.append(depth_id)
                    break
                calc_neural_acyclic_path(depth, depth_id, path)


def calc_required_cyclic_depth(node_ids_all, enabled_connections):
    required = {}
    for n_id in node_ids_all:
        depth_list = [[n_id]]
        while True:
            essential = set(
                source_id for (_, source_id, target_id) in enabled_connections if
                target_id in depth_list[-1] and source_id not in depth_list[-1] and source_id != n_id)
            if not essential or list(essential) in depth_list:
                break
            depth_list.append(list(essential))
        required[n_id] = depth_list
    return required


def calc_neural_cyclic_path(depth, n_id, path, searching_for):
    if n_id in searching_for:
        return
    searching_for.add(n_id)
    depth_list = depth[n_id]
    if len(depth_list) == 1:
        searching_for.remove(n_id)
        if depth_list[0][0] not in path:
            path.append(depth_list[0][0])
    else:
        for depth_set in reversed(depth_list):
            for depth_id in depth_set:
                if n_id == depth_id:
                    if depth_id not in path:
                        path.append(depth_id)
                    break
                calc_neural_cyclic_path(depth, depth_id, path, searching_for)
