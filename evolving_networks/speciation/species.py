from evolving_networks.errors import InvalidAttributeValueError


class Species(object):
    def __init__(self, specie_id, generation):
        self.specie_id = specie_id
        self.created = generation
        self._members = None
        self._representative = None

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
