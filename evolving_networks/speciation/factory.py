from itertools import count


class Factory(object):
    def __init__(self):
        self.species = dict()
        self._specie_indexer = count(0)

    def speciate(self):
        raise NotImplementedError()
