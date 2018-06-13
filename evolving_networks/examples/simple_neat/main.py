from evolving_networks.config import Config
from evolving_networks.population import Population
from evolving_networks.reproduction.traditional import Traditional as TraditionalReproduction
from evolving_networks.speciation.traditional import Traditional as TraditionalSpeciation


def main():
    config = Config(filename='config/config_1.ini')
    reproduction_factory = TraditionalReproduction()
    speciation_factory = TraditionalSpeciation()
    population = Population(reproduction_factory, speciation_factory)
    population.initialize(config)


if __name__ == "__main__":
    main()
