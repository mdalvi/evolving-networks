import matplotlib.pyplot as plt

from evolving_networks.activations import Activations
from evolving_networks.aggregations import Aggregations
from evolving_networks.neat.configurations.config import Config
from evolving_networks.neat.phenome.feed_forward import FeedForwardNetwork
from evolving_networks.neat.population import Population
from evolving_networks.neat.reproduction.traditional import Traditional as TraditionalReproduction
from evolving_networks.regulations.blended import Blended as BlendedComplexityRegulation
from evolving_networks.reporting import reporter, stdout
from evolving_networks.speciation.traditional import Traditional as TraditionalSpeciation

# 2-input XOR inputs and expected outputs.
xor_inputs = [(0.0, 0.0), (0.0, 1.0), (1.0, 0.0), (1.0, 1.0)]
xor_outputs = [(0.0,), (1.0,), (1.0,), (0.0,)]


def main():
    config = Config(filename='config/config_1.ini')
    reproduction_factory = TraditionalReproduction()
    speciation_factory = TraditionalSpeciation()
    regulation_factory = BlendedComplexityRegulation(config)
    reporting_factory = reporter.Reporter()
    reporting_factory.add_report(stdout.StdOut())
    population = Population(reproduction_factory, speciation_factory, regulation_factory, reporting_factory)
    population.initialize(evaluate, config)
    history = population.fit()
    print(population.best_genome)
    visualize(population, history)


def visualize(population, history):
    plt.plot(range(population.generation), history.max_fitness, 'r-', label="Max Fitness")
    plt.plot(range(population.generation), history.mean_fitness, 'r:', label="Mean Fitness")
    plt.plot(range(population.generation), history.mean_species_fitness, 'b--', label="Mean Species Best")
    plt.xlabel("Generations")
    plt.ylabel("Fitness")
    plt.grid()
    plt.legend(loc="best")
    plt.show()
    plt.close()

    plt.plot(range(population.generation), history.max_complexity, 'g-', label="Max Complexity")
    plt.plot(range(population.generation), history.mean_complexity, 'g:', label="Mean Complexity")
    plt.xlabel("Generations")
    plt.ylabel("Fitness")
    plt.grid()
    plt.legend(loc="best")
    plt.show()
    plt.close()


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
