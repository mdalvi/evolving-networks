from evolving_networks.config import Config
from evolving_networks.population import Population
from evolving_networks.reproduction.traditional import Traditional as TraditionalReproduction
from evolving_networks.speciation.traditional import Traditional as TraditionalSpeciation
from evolving_networks.phenome.feed_forward import FeedForwardNetwork

def main():
    # 2-input XOR inputs and expected outputs.
    xor_inputs = [(0.0, 0.0), (0.0, 1.0), (1.0, 0.0), (1.0, 1.0)]
    xor_outputs = [(0.0,), (1.0,), (1.0,), (0.0,)]

    config = Config(filename='config/config_1.ini')
    reproduction_factory = TraditionalReproduction()
    speciation_factory = TraditionalSpeciation()
    population = Population(reproduction=reproduction_factory, speciation=speciation_factory)
    population.initialize(config)
    population.fit(evaluate, config)


def evaluate(genomes, config):
    for genome_id, genome in genomes:
        genome.fitness = 4.0
        ff_network = FeedForwardNetwork(genome, config)
        # for x, y in zip(xor_inputs, xor_outputs):
        #     output = net.activate(x)
        #     genome.fitness -= (output[0] - y[0]) ** 2


if __name__ == "__main__":
    main()
