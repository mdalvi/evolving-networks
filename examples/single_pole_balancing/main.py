import matplotlib.pyplot as plt

from evolving_networks.activations.activations import Activations
from evolving_networks.aggregations import Aggregations
from evolving_networks.complexity_regulation.blended import Blended as BlendedComplexityRegulation
from evolving_networks.complexity_regulation.complexify import Complexify as ComplexifyComplexityRegulation
from evolving_networks.config import Config
from evolving_networks.phenome.feed_forward import FeedForwardNetwork
from evolving_networks.population import Population
from evolving_networks.reproduction.traditional import Traditional as TraditionalReproduction
from evolving_networks.speciation.traditional import Traditional as TraditionalSpeciation
from evolving_networks.speciation.agglomerative import Agglomerative as AgglomerativeSpeciation
from evolving_networks.speciation.kmeans import KMeans as KMeansSpeciation

from examples.single_pole_balancing import cart_pole
runs_per_net = 5
simulation_seconds = 60.0

def main():
    config = Config(filename='config/config_1.ini')
    reproduction_factory = TraditionalReproduction()
    speciation_factory = AgglomerativeSpeciation()
    complexity_regulation_factory = BlendedComplexityRegulation(config)
    population = Population(reproduction_factory, speciation_factory, complexity_regulation_factory)
    population.initialize(evaluate, config)
    history = population.fit()
    print(population.best_genome)


def evaluate(genomes, config):
    for genome_id, genome in genomes:
        ff_network = FeedForwardNetwork(genome, config)
        ff_network.initialize(Activations(), Aggregations())

        fitnesses = []
        for runs in range(runs_per_net):
            sim = cart_pole.CartPole()

            # Run the given simulation for up to num_steps time steps.
            fitness = 0.0
            while sim.t < simulation_seconds:
                inputs = sim.get_scaled_state()
                action = ff_network.activate(inputs)

                # Apply action to the simulated cart-pole
                force = cart_pole.discrete_actuator_force(action)
                sim.step(force)

                # Stop if the network fails to keep the cart within the position or angle limits.
                # The per-run fitness is the number of time steps the network can balance the pole
                # without exceeding these limits.
                if abs(sim.x) >= sim.position_limit or abs(sim.theta) >= sim.angle_limit_radians:
                    break

                fitness = sim.t
            fitnesses.append(fitness)
        genome.fitness = min(fitnesses)

    fit_list = [genome.fitness for _, genome in genomes]
    print(sorted(fit_list, reverse=True))


if __name__ == "__main__":
    main()
