from evolving_networks.errors import InvalidAttributeValueError
from evolving_networks.math_util import stat_functions


class Species(object):
    def __init__(self, specie_id, generation, config):
        self.specie_id = specie_id
        self.created = generation
        self.last_improved = generation

        self._members = None
        self._representative = None
        self.fitness_history = []
        self.is_stagnant = False
        self.adjusted_fitness = None
        self.fitness_criterion = stat_functions.get(config.fitness_criterion)

    @property
    def fitness(self):
        if self.fitness_criterion is None:
            raise InvalidAttributeValueError('NO FITNESS CRITERION FOUND FOR SPECIES [{}]'.format(self.specie_id))

        return self.fitness_criterion([genome.fitness for genome in self.members.values()])

    @property
    def representative(self):
        if self._representative is None:
            raise InvalidAttributeValueError('NO REPRESENTATIVE FOUND FOR SPECIES [{}]'.format(self.specie_id))
        return self._representative

    @representative.setter
    def representative(self, value):
        self._representative = value

    @property
    def members(self):
        if self._members is None:
            raise InvalidAttributeValueError('NO MEMBERS FOUND FOR SPECIES [{}]'.format(self.specie_id))
        return self._members

    @members.setter
    def members(self, value):
        self._members = value
