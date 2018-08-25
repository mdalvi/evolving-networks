import configparser
import os

from evolving_networks.errors import InvalidConfigurationError, InvalidConfigurationFile


class ConfigParameter(object):
    def __init__(self, name, type, default=None):
        self.name = name
        self.type = type
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
    def __init__(self, filename):

        if not os.path.isfile(filename):
            raise InvalidConfigurationFile("NO SUCH CONFIGURATION FILE FOUND [{}]".format(os.path.abspath(filename)))

        config_parser = configparser.ConfigParser()
        with open(filename, encoding="utf8") as f:
            config_parser.read_file(f)

        if not config_parser.has_section('PyTorch'):
            raise configparser.NoSectionError("NO SUCH 'NEAT' SECTION FOUND IN CONFIGURATION FILE")

        self.pytorch = DefaultPyTorchConfig(config_parser)

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


class DefaultReproductionConfig(object):
    __params = [ConfigParameter('species_elitism', int)]

    def __init__(self, config_parser):
        for parameter in self.__params:
            setattr(self, parameter.name, parameter.parse('DefaultReproduction', config_parser))


class DefaultSpeciesConfig(object):
    __params = [ConfigParameter('compatibility_threshold', float), ConfigParameter('fitness_criterion', str),
                ConfigParameter('max_stagnation', int), ConfigParameter('elitism', float),
                ConfigParameter('off_spring_asexual_rate', float), ConfigParameter('survivor_rate', float),
                ConfigParameter('inter_species_mating_rate', float), ConfigParameter('specie_clusters', int)]

    def __init__(self, config_parser):
        for parameter in self.__params:
            setattr(self, parameter.name, parameter.parse('DefaultSpecies', config_parser))


class DefaultGenomeConfig(object):
    __params = [ConfigParameter('compatibility_weight_contribution', float)]

    def __init__(self, config_parser):
        for parameter in self.__params:
            setattr(self, parameter.name, parameter.parse('DefaultGenome', config_parser))


class DefaultConnectionConfig(object):
    __params = [ConfigParameter('weight_mutate_rate', float), ConfigParameter('weight_mutate_stdev', float),
                ConfigParameter('weight_replace_rate', float), ConfigParameter('weight_init_mean', float),
                ConfigParameter('weight_init_stdev', float), ConfigParameter('weight_init_type', str),
                ConfigParameter('weight_min_value', float), ConfigParameter('weight_max_value', float),
                ConfigParameter('single_structural_mutation', bool)]

    def __init__(self, config_parser):
        for parameter in self.__params:
            setattr(self, parameter.name, parameter.parse('DefaultConnection', config_parser))


class DefaultPyTorchConfig(object):
    __params = [ConfigParameter('population_size', int), ConfigParameter('fitness_criterion', str),
                ConfigParameter('no_fitness_termination', bool), ConfigParameter('fitness_threshold', float)]

    def __init__(self, config_parser):
        for parameter in self.__params:
            setattr(self, parameter.name, parameter.parse('PyTorch', config_parser))
