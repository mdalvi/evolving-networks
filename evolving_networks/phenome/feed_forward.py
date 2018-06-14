from evolving_networks.phenome.helpers import build_essential_dict
from evolving_networks.phenome.phenome import Phenome


class FeedForwardNetwork(Phenome):
    def __init__(self, genome, config):
        super(FeedForwardNetwork, self).__init__()
        self._create(genome, config)

    def _create(self, genome, config):
        enabled_connections = [(conn.source_id, conn.target_id) for conn in genome.connections.values() if conn.enabled]
        essentials = build_essential_dict(genome.all_keys, enabled_connections)
        print(essentials)

    def activate(self):
        pass
