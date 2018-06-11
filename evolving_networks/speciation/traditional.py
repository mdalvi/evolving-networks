from evolving_networks.speciation.factory import Factory


class Traditional(Factory):
    def __init__(self):
        super(Factory, self).__init__()
        self.species = dict()

    def speciate(self):
        pass
