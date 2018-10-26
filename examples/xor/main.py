from datetime import datetime

from evolving_networks.activations import Activations
from evolving_networks.aggregations import Aggregations
from evolving_networks.configurations.config import Config
from evolving_networks.phenome.feed_forward import FeedForwardNetwork
from evolving_networks.population import Population
from evolving_networks.regulations.blended import Blended as BlendedComplexityRegulation
from evolving_networks.reporting import reporter, stdout, statistics
from evolving_networks.reproduction.traditional import Traditional as TraditionalReproduction
from evolving_networks.speciation.traditional import Traditional as TraditionalSpeciation

# 2-input XOR inputs and expected outputs.
xor_inputs = [(0.0, 0.0), (0.0, 1.0), (1.0, 0.0), (1.0, 1.0)]
xor_outputs = [(0.0,), (1.0,), (1.0,), (0.0,)]


def main():
    config = Config()
    config.initialize('config/config_1.ini')
    reproduction_factory = TraditionalReproduction()
    speciation_factory = TraditionalSpeciation()
    regulation_factory = BlendedComplexityRegulation(config)
    reporting_factory = reporter.Reporter()
    reporting_factory.add_report(stdout.StdOut())
    reporting_factory.add_report(statistics.Statistics())
    population = Population(reproduction_factory, speciation_factory, regulation_factory, reporting_factory)
    population.initialize(evaluate, config)
    population.fit()
    best_genome = population.best_genome

    print(population.best_genome)
    filename = 'save/best_solution_{0}.json'.format(datetime.now().strftime('%Y%m%d%H%M%S'))
    with open(filename, 'w') as f:
        f.write(best_genome.to_json())


def evaluate(genomes, config):
    for genome_id, genome in genomes:
        genome.fitness = 4.0
        network = FeedForwardNetwork(genome, config)
        network.initialize(Activations(), Aggregations())
        network.reset(hard=True)
        for x, y in zip(xor_inputs, xor_outputs):
            output = network.activate(x)
            genome.fitness -= (output[0] - y[0]) ** 2
            genome.is_damaged = network.is_damaged


if __name__ == "__main__":
    main()
