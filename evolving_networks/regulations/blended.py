from evolving_networks.regulations.factory import Factory


class Blended(Factory):
    def __init__(self, config):
        super(Blended, self).__init__()
        self.mode = 'blending'
        self.node_add_rate = config.genome.node_add_rate
        self.conn_add_rate = config.genome.conn_add_rate
        self.node_delete_rate = config.genome.node_delete_rate
        self.conn_delete_rate = config.genome.conn_delete_rate

    def determine_mode(self, **kwargs):
        pass

    def __str__(self):
        return "Blended complexity regulation with mode {0}".format(self.mode)
