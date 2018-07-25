from evolving_networks.complexity_regulation.factory import Factory


class Blended(Factory):
    def __init__(self, config):
        super(Blended, self).__init__()
        self._mode = 'blending'
        self.node_add_rate = config.genome.node_add_rate
        self.conn_add_rate = config.genome.conn_add_rate
        self.node_delete_rate = config.genome.node_delete_rate
        self.conn_delete_rate = config.genome.conn_delete_rate
        self.off_spring_asexual_rate = config.species.off_spring_asexual_rate

    @property
    def mode(self):
        return self._mode

    def determine_mode(self, statistics):
        pass
