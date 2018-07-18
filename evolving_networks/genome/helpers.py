"""
# ==============
# References
# ==============

[1] https://github.com/CodeReclaimers/neat-python/blob/master/neat/graphs.py

"""


def is_cyclic(connections, source_id, target_id):  # [1]
    if source_id == target_id:
        return True

    visited = {target_id}
    while True:
        num_added = 0
        for connection in connections.values():
            a, b = connection.source_id, connection.target_id
            if a in visited and b not in visited:
                if b == source_id:
                    return True

                visited.add(b)
                num_added += 1

        if num_added == 0:
            return False
