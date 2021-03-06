import configparser
import json
import os

from evolving_networks.errors import InvalidConfigurationError, InvalidConfigurationFile


class ConfigParameter(object):
    def __init__(self, name, _type, default=None):
        self.name = name
        self.type = _type
        self.default = default

    def parse(self, section, config_parser):
        if self.type == int:
            return config_parser.getint(section, self.name)
        if self.type == bool:
            return config_parser.getboolean(section, self.name)
        if self.type == float:
            return config_parser.getfloat(section, self.name)
        if self.type == list:
            v = config_parser.get(section, self.name)
            return v.split(" ")
        if self.type == str:
            return config_parser.get(section, self.name)

        raise InvalidConfigurationError("UNEXPECTED CONFIGURATION TYPE [{}]".format(repr(self.type)))


class Config(object):
    _params = ['neat', 'node', 'connection', 'genome', 'species', 'reproduction']

    def __init__(self):
        self.neat = None
        self.node = None
        self.connection = None
        self.genome = None
        self.species = None
        self.reproduction = None

    def initialize(self, filename):

        if not os.path.isfile(filename):
            raise InvalidConfigurationFile("NO SUCH CONFIGURATION FILE FOUND [{}]".format(os.path.abspath(filename)))

        config_parser = configparser.ConfigParser()
        with open(filename, encoding="utf8") as f:
            config_parser.read_file(f)

        if not config_parser.has_section('NEAT'):
            raise configparser.NoSectionError("NO SUCH 'NEAT' SECTION FOUND IN CONFIGURATION FILE")

        self.neat = DefaultNEATConfig(config_parser)

        if not config_parser.has_section('DefaultNode'):
            raise configparser.NoSectionError("NO SUCH 'DefaultNode' SECTION FOUND IN CONFIGURATION FILE")

        self.node = DefaultNodeConfig(config_parser)

        if not config_parser.has_section('DefaultConnection'):
            raise configparser.NoSectionError("NO SUCH 'DefaultConnection' SECTION FOUND IN CONFIGURATION FILE")

        self.connection = DefaultConnectionConfig(config_parser)

        if not config_parser.has_section('DefaultGenome'):
            raise configparser.NoSectionError("NO SUCH 'DefaultGenome' SECTION FOUND IN CONFIGURATION FILE")

        self.genome = DefaultGenomeConfig(config_parser)

        if not config_parser.has_section('DefaultSpecies'):
            raise configparser.NoSectionError("NO SUCH 'DefaultSpecies' SECTION FOUND IN CONFIGURATION FILE")

        self.species = DefaultSpeciesConfig(config_parser)

        if not config_parser.has_section('DefaultReproduction'):
            raise configparser.NoSectionError("NO SUCH 'DefaultReproduction' SECTION FOUND IN CONFIGURATION FILE")

        self.reproduction = DefaultReproductionConfig(config_parser)

    def __eq__(self, other):
        for p in self._params:
            if getattr(self, p) != getattr(other, p):
                return False
        return True

    def to_json(self):
        result = dict()
        for p in self._params:
            result[p] = getattr(self, p).to_json()
        return json.dumps(result)

    def from_json(self, config_json):
        result = json.loads(config_json)
        for k, v in result.items():
            if k == 'neat':
                setattr(self, k, DefaultNEATConfig().from_json(v))
            elif k == 'node':
                setattr(self, k, DefaultNodeConfig().from_json(v))
            elif k == 'connection':
                setattr(self, k, DefaultConnectionConfig().from_json(v))
            elif k == 'genome':
                setattr(self, k, DefaultGenomeConfig().from_json(v))
            elif k == 'species':
                setattr(self, k, DefaultSpeciesConfig().from_json(v))
            elif k == 'reproduction':
                setattr(self, k, DefaultReproductionConfig().from_json(v))
            else:
                raise InvalidConfigurationError()
        return self


class DefaultReproductionConfig(object):
    _params = [ConfigParameter('species_elitism', int)]

    def __init__(self, config_parser=None):
        if config_parser is not None:
            for parameter in self._params:
                setattr(self, parameter.name, parameter.parse('DefaultReproduction', config_parser))

    def __eq__(self, other):
        for p in self._params:
            if getattr(self, p.name) != getattr(other, p.name):
                return False
        return True

    def to_json(self):
        result = {}
        for p in self._params:
            result[p.name] = getattr(self, p.name)
        return json.dumps(result)

    def from_json(self, config_json):
        result = json.loads(config_json)
        for k, v in result.items():
            setattr(self, k, v)
        return self


class DefaultSpeciesConfig(object):
    _params = [ConfigParameter('compatibility_threshold', float), ConfigParameter('fitness_criterion', str),
               ConfigParameter('max_stagnation', int), ConfigParameter('elitism', float),
               ConfigParameter('off_spring_asexual_rate', float), ConfigParameter('survivor_rate', float),
               ConfigParameter('inter_species_mating_rate', float), ConfigParameter('specie_clusters', int)]

    def __init__(self, config_parser=None):
        if config_parser is not None:
            for parameter in self._params:
                setattr(self, parameter.name, parameter.parse('DefaultSpecies', config_parser))

    def __eq__(self, other):
        for p in self._params:
            if getattr(self, p.name) != getattr(other, p.name):
                return False
        return True

    def to_json(self):
        result = {}
        for p in self._params:
            result[p.name] = getattr(self, p.name)
        return json.dumps(result)

    def from_json(self, config_json):
        result = json.loads(config_json)
        for k, v in result.items():
            setattr(self, k, v)
        return self


class DefaultGenomeConfig(object):
    _params = [ConfigParameter('num_inputs', int), ConfigParameter('num_hidden', int),
               ConfigParameter('num_outputs', int), ConfigParameter('initial_connection', str),
               ConfigParameter('partial_connection_rate', float), ConfigParameter('feed_forward', bool),
               ConfigParameter('node_add_rate', float), ConfigParameter('node_delete_rate', float),
               ConfigParameter('conn_add_rate', float), ConfigParameter('conn_delete_rate', float),
               ConfigParameter('single_structural_mutation', bool),
               ConfigParameter('compatibility_disjoint_contribution', float),
               ConfigParameter('compatibility_excess_contribution', float),
               ConfigParameter('compatibility_weight_contribution', float)]

    def __init__(self, config_parser=None):
        if config_parser is not None:
            for parameter in self._params:
                setattr(self, parameter.name, parameter.parse('DefaultGenome', config_parser))

    def __eq__(self, other):
        for p in self._params:
            if getattr(self, p.name) != getattr(other, p.name):
                return False
        return True

    def to_json(self):
        result = {}
        for p in self._params:
            result[p.name] = getattr(self, p.name)
        return json.dumps(result)

    def from_json(self, config_json):
        result = json.loads(config_json)
        for k, v in result.items():
            setattr(self, k, v)
        return self


class DefaultConnectionConfig(object):
    _params = [ConfigParameter('weight_mutate_rate', float), ConfigParameter('weight_mutate_stdev', float),
               ConfigParameter('weight_replace_rate', float), ConfigParameter('enabled_default', bool),
               ConfigParameter('enabled_mutate_rate', float), ConfigParameter('weight_init_mean', float),
               ConfigParameter('weight_init_stdev', float), ConfigParameter('weight_init_type', str),
               ConfigParameter('weight_min_value', float), ConfigParameter('weight_max_value', float),
               ConfigParameter('single_structural_mutation', bool)]

    def __init__(self, config_parser=None):
        if config_parser is not None:
            for parameter in self._params:
                setattr(self, parameter.name, parameter.parse('DefaultConnection', config_parser))

    def __eq__(self, other):
        for p in self._params:
            if getattr(self, p.name) != getattr(other, p.name):
                return False
        return True

    def to_json(self):
        result = {}
        for p in self._params:
            result[p.name] = getattr(self, p.name)
        return json.dumps(result)

    def from_json(self, config_json):
        result = json.loads(config_json)
        for k, v in result.items():
            setattr(self, k, v)
        return self


class DefaultNodeConfig(object):
    _params = [ConfigParameter('bias_mutate_rate', float), ConfigParameter('bias_mutate_stdev', float),
               ConfigParameter('bias_replace_rate', float), ConfigParameter('bias_init_mean', float),
               ConfigParameter('bias_init_stdev', float), ConfigParameter('bias_init_type', str),
               ConfigParameter('bias_min_value', float), ConfigParameter('bias_max_value', float),
               ConfigParameter('response_mutate_rate', float), ConfigParameter('response_mutate_stdev', float),
               ConfigParameter('response_replace_rate', float), ConfigParameter('response_init_mean', float),
               ConfigParameter('response_init_stdev', float), ConfigParameter('response_init_type', str),
               ConfigParameter('response_min_value', float), ConfigParameter('response_max_value', float),
               ConfigParameter('activation_default', str), ConfigParameter('activation_default_output', str),
               ConfigParameter('activation_mutate_rate', float), ConfigParameter('activation_options', list),
               ConfigParameter('aggregation_default', str), ConfigParameter('aggregation_mutate_rate', float),
               ConfigParameter('aggregation_options', list), ConfigParameter('single_structural_mutation', bool)]

    def __init__(self, config_parser=None):
        if config_parser is not None:
            for parameter in self._params:
                setattr(self, parameter.name, parameter.parse('DefaultNode', config_parser))

    def __eq__(self, other):
        for p in self._params:
            if getattr(self, p.name) != getattr(other, p.name):
                return False
        return True

    def to_json(self):
        result = {}
        for p in self._params:
            result[p.name] = getattr(self, p.name)
        return json.dumps(result)

    def from_json(self, config_json):
        result = json.loads(config_json)
        for k, v in result.items():
            setattr(self, k, v)
        return self


class DefaultNEATConfig(object):
    _params = [ConfigParameter('population_size', int), ConfigParameter('fitness_criterion', str),
               ConfigParameter('no_fitness_termination', bool), ConfigParameter('fitness_threshold', float),
               ConfigParameter('phased_complexity_type', str),
               ConfigParameter('phased_complexity_threshold', float),
               ConfigParameter('phase_fitness_plateau_threshold', int),
               ConfigParameter('phase_simplification_generations_threshold', int)]

    def __init__(self, config_parser=None):
        if config_parser is not None:
            for parameter in self._params:
                setattr(self, parameter.name, parameter.parse('NEAT', config_parser))

    def __eq__(self, other):
        for p in self._params:
            if getattr(self, p.name) != getattr(other, p.name):
                return False
        return True

    def to_json(self):
        result = {}
        for p in self._params:
            result[p.name] = getattr(self, p.name)
        return json.dumps(result)

    def from_json(self, config_json):
        result = json.loads(config_json)
        for k, v in result.items():
            setattr(self, k, v)
        return self
