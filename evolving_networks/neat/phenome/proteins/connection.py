from evolving_networks.neat.phenome.proteins.protein import Protein


class Connection(Protein):
    def __init__(self, c_id, source_id, target_id, weight, enabled):
        super(Connection, self).__init__()
        self.id = c_id
        self.source_id = source_id
        self.target_id = target_id
        self.weight = weight
        self.enabled = enabled
