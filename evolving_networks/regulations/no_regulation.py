from evolving_networks.regulations.factory import Factory


class NoRegulation(Factory):
    def __init__(self, config):
        super(NoRegulation, self).__init__()
        self.mode = 'no_regulation'
        self.node_add_rate = config.genome.node_add_rate
        self.conn_add_rate = config.genome.conn_add_rate
        self.node_delete_rate = config.genome.node_delete_rate
        self.conn_delete_rate = config.genome.conn_delete_rate

    def determine_mode(self, **kwargs):
        pass

    def __str__(self):
        return "None complexity regulation with mode {0}".format(self.mode)
