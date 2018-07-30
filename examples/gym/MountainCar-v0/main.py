"""
# ==============
# References
# ==============

[1] https://gym.openai.com/docs/
[2] https://gym.openai.com/envs/CartPole-v1/

"""
import concurrent.futures
import time

import gym

from evolving_networks.activations.activations import Activations
from evolving_networks.aggregations import Aggregations
from evolving_networks.complexity_regulation.phased import Phased as PhasedComplexityRegulation
from evolving_networks.complexity_regulation.blended import Blended as BlendedComplexityRegulation
from evolving_networks.config import Config
from evolving_networks.math_util import mean
from evolving_networks.phenome.feed_forward import FeedForwardNetwork
from evolving_networks.population import Population
from evolving_networks.reproduction.traditional import Traditional as TraditionalReproduction
from evolving_networks.speciation.traditional import Traditional as TraditionalSpeciation

gym.logger.set_level(40)


class ParallelEvaluator(object):
    def __init__(self, num_workers, eval_function):
        self.evaluations = 0
        self.num_workers = num_workers
        self.eval_function = eval_function

    def evaluate(self, genomes, config):
        t0 = time.time()
        max_fitness = float('-Infinity')
        process_data = [(g_id, genome, config) for g_id, genome in genomes]
        with concurrent.futures.ProcessPoolExecutor(max_workers=self.num_workers) as executor:  # [1]
            for fitness, (g_id, genome) in zip(executor.map(self.eval_function, process_data), genomes):

                if fitness > max_fitness:
                    max_fitness = fitness

                genome.fitness = fitness

        self.evaluations += 1
        print('Iter [{0}], Time [{1} secs], Max Fitness [{2}]'.format(self.evaluations, round(time.time() - t0),
                                                                      max_fitness))


def evaluate(attributes):
    g_id, genome, config = attributes
    ff_network = FeedForwardNetwork(genome, config)
    ff_network.initialize(Activations(), Aggregations())

    fitness = []
    env = gym.make('MountainCar-v0')  # [1], [2]
    for e_idx in range(100):
        observation = env.reset()
        episode_reward = 0
        while True:
            action = ff_network.activate(observation.tolist())
            action = action.index(max(action))
            observation, reward, done, info = env.step(action)
            episode_reward += reward
            if done:
                break

        fitness.append(episode_reward)
    return mean(fitness)


def main():
    config = Config(filename='config/config_1.ini')
    reproduction_factory = TraditionalReproduction()
    speciation_factory = TraditionalSpeciation()
    complexity_regulation_factory = PhasedComplexityRegulation(config)
    population = Population(reproduction_factory, speciation_factory, complexity_regulation_factory)
    parallel_evaluator = ParallelEvaluator(num_workers=4, eval_function=evaluate)
    population.initialize(parallel_evaluator.evaluate, config)
    hist = population.fit()
    best_genome = population.best_genome
    print(best_genome)

    # Champion solution
    env = gym.make('MountainCar-v0')  # [1]
    ff_network = FeedForwardNetwork(best_genome, config)
    ff_network.initialize(Activations(), Aggregations())

    fitness = []
    for e_idx in range(100):
        observation = env.reset()
        episode_reward = 0
        while True:
            if e_idx % 10 == 0:
                env.render()
                time.sleep(0.075)
            action = ff_network.activate(observation.tolist())
            action = action.index(max(action))
            observation, reward, done, info = env.step(action)
            episode_reward += reward
            if done:
                break
        fitness.append(episode_reward)
    print("Examination mean fitness [{}]".format(mean(fitness)))


if __name__ == "__main__":
    main()
