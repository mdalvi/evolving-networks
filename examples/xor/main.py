import matplotlib.pyplot as plt

from evolving_networks.activations.activations import Activations
from evolving_networks.aggregations import Aggregations
from evolving_networks.config import Config
from evolving_networks.phenome.feed_forward import FeedForwardNetwork
from evolving_networks.population import Population
from evolving_networks.reproduction.traditional import Traditional as TraditionalReproduction
from evolving_networks.speciation.traditional import Traditional as TraditionalSpeciation

# 2-input XOR inputs and expected outputs.
xor_inputs = [(0.0, 0.0), (0.0, 1.0), (1.0, 0.0), (1.0, 1.0)]
xor_outputs = [(0.0,), (1.0,), (1.0,), (0.0,)]


def main():
    config = Config(filename='config/config_2.ini')
    reproduction_factory = TraditionalReproduction()
    speciation_factory = TraditionalSpeciation()
    population = Population(reproduction_factory, speciation_factory)
    population.initialize(evaluate, config)
    history = population.fit()
    print(population.best_genome)

    plt.plot(range(population.generation), history.max_fitness, 'r-', label="Max Fitness")
    plt.plot(range(population.generation), history.mean_fitness, 'r:', label="Mean Fitness")
    plt.plot(range(population.generation), history.mean_species_best_fitness, 'b--', label="Mean Species Best")
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
        ff_network = FeedForwardNetwork(genome, config)
        ff_network.initialize(Activations(), Aggregations())
        for x, y in zip(xor_inputs, xor_outputs):
            output = ff_network.activate(x)

            genome.fitness -= (output[0] - y[0]) ** 2
            genome.is_damaged = ff_network.is_damaged

    fit_list = [genome.fitness for _, genome in genomes]
    print(sorted(fit_list, reverse=True))


if __name__ == "__main__":
    main()
