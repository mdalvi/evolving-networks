from evolving_networks.speciation.factory import Factory
from evolving_networks.speciation.helpers import genomic_distance
from evolving_networks.speciation.species import Species


class Traditional(Factory):
    def __init__(self):
        super(Traditional, self).__init__()
        self.species = dict()
        self._genome_to_species = {}

    def speciate(self, population, generation, config):
        # TODO: Dynamic threshold
        compatibility_threshold = config.species.compatibility_threshold

        unspeciated = set(population.keys())
        representatives, members = {}, {}

        # TODO: Optimize here
        # Because we are looping over species and removing close candidates from unspeciated, this method may not give optimal replacements for given representatives.
        for specie_id, specie in self.species.items():
            specie_distances = []
            for genome_id in unspeciated:
                genome = population[genome_id]
                d = genomic_distance(specie.representative, genome, config)
                specie_distances.append((d, genome))

            _, new_representative = min(specie_distances, key=lambda x: x[0])
            new_representative_id = new_representative.id
            representatives[specie_id] = new_representative_id
            members[specie_id] = [new_representative_id]
            unspeciated.remove(new_representative_id)

        while unspeciated:
            genome_id = unspeciated.pop()
            genome = population[genome_id]

            specie_distances = []
            for specie_id, representative_id in representatives.items():
                representative = population[representative_id]
                d = genomic_distance(representative, genome, config)
                if d < compatibility_threshold:
                    specie_distances.append((d, specie_id))

            if specie_distances:
                _, specie_id = min(specie_distances, key=lambda x: x[0])
                members[specie_id].append(genome_id)
            else:
                specie_id = next(self._specie_indexer)
                representatives[specie_id] = genome_id
                members[specie_id] = [genome_id]

        self._genome_to_species = {}
        for specie_id, representative_id in representatives.items():
            s = self.species.get(specie_id)
            if s is None:
                s = Species(specie_id, generation, config.species)

            specie_members ={}
            for genome_id in members[specie_id]:
                self._genome_to_species[genome_id] = specie_id
                specie_members[genome_id] = population[genome_id]

            s.representative = population[representative_id]
            s.members = specie_members
            self.species[specie_id] = s

    def get_species_id(self, genome_id):
        return self._genome_to_species[genome_id]

    def get_species(self, genome_id):
        specie_id = self._genome_to_species[genome_id]
        return self.species[specie_id]
