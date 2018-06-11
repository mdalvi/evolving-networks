class Species(object):
    def __init__(self, specie_id, generation):
        self.specie_id = specie_id
        self.generation = generation
        self.members = dict()
