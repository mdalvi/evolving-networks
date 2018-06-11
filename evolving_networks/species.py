from itertools import count

class SpeciesSet(object):
    def __init__(self, config):
        self.config = config
        self.species = dict()
        self._specie_indexer = count(0)

    def _calculate_distance(self, genome_1, genome_2, config):
        return genome_1.distance(genome_2, config)

    def speciate(self, population, generation, config):
        # TODO: Dynamic threshold
        compatibility_threshold = config.species.compatibility_threshold

        unspeciated = set(population.keys())
        representatives, members = {},{}

        # Because we are looping over species and removing similar candidates from unspeciated, this method may not give optimal replacements for given representatives.
        for specie_id, specie in self.species.items():
            candidates = []
            for gid in unspeciated:
                g = population[gid]
                d = distances(s.representative, g)
                candidates.append((d, g))

            # The new representative is the genome closest to the current representative.
            ignored_rdist, new_rep = min(candidates, key=lambda x: x[0])
            new_rid = new_rep.key
            new_representatives[sid] = new_rid
            new_members[sid] = [new_rid]
            unspeciated.remove(new_rid)

        while unspeciated:
            genome_id = unspeciated.pop()
            genome = population[genome_id]

            specie_distances = []
            for specie_id, representative_id in representatives.items():
                representative = population[representative_id]
                d = self._calculate_distance(representative, genome)
                if d < compatibility_threshold:
                    specie_distances.append((d, specie_id))

            if specie_distances:
                _, specie_id = min(specie_distances, key=lambda x: x[0])
                members[specie_id].append(genome_id)
            else:
                specie_id = next(self._specie_indexer)
                representatives[specie_id] = genome_id
                members[specie_id] = [genome_id]

        for specie_id, representative_id in representatives.items():
            s = self.species.get(specie_id)
            if s is None:
                s = Species(specie_id, generation)
            specie_members = members[specie_id]
            s.members = specie_members
            self.species[specie_id] = s
