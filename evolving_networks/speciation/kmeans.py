from evolving_networks.speciation.factory import Factory


class KMeans(Factory):
    def __init__(self):
        super(Factory, self).__init__()
        self.species = dict()
        self._genome_to_species = {}

    def speciate(self, population, generation, config):
        raise NotImplementedError()

    def get_species_id(self, genome_id):
        return self._genome_to_species[genome_id]

    def get_species(self, genome_id):
        specie_id = self._genome_to_species[genome_id]
        return self.species[specie_id]
