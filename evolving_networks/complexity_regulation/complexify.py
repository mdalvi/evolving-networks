from evolving_networks.complexity_regulation.factory import Factory


class Complexify(Factory):
    def __init__(self, config):
        super(Complexify, self).__init__()
        self._mode = 'complexifying'
        self.node_add_rate = config.genome.node_add_rate
        self.conn_add_rate = config.genome.conn_add_rate
        self.node_delete_rate = 0.0
        self.conn_delete_rate = 0.0
        self.off_spring_asexual_rate = config.species.off_spring_asexual_rate

    @property
    def mode(self):
        return self._mode

    def determine_mode(self, statistics):
        pass
