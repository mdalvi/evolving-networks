from evolving_networks.regulations.factory import Factory


class NoRegulation(Factory):
    def __init__(self, config):
        super(NoRegulation, self).__init__()
        self.mode = 'no_regulation'

    def determine_mode(self, statistics):
        pass

    def __str__(self):
        return "None complexity regulation with mode {0}".format(self.mode)
