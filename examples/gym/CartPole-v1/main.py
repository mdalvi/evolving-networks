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

from evolving_networks.activations import Activations
from evolving_networks.aggregations import Aggregations
from evolving_networks.configurations.config import Config
from evolving_networks.math_util import mean
from evolving_networks.phenome.feed_forward import FeedForwardNetwork
from evolving_networks.population import Population
from evolving_networks.regulations.phased import Phased as PhasedComplexityRegulation
from evolving_networks.reporting import reporter, stdout, statistics
from evolving_networks.reproduction.traditional import Traditional as TraditionalReproduction
from evolving_networks.speciation.traditional import Traditional as TraditionalSpeciation

gym.logger.set_level(40)


class ParallelEvaluator(object):
    def __init__(self, num_workers, eval_function):
        self.evaluations = 0
        self.num_workers = num_workers
        self.eval_function = eval_function

    def evaluate(self, genomes, config):
        max_fitness = float('-Infinity')
        process_data = [(g_id, genome, config) for g_id, genome in genomes]
        with concurrent.futures.ProcessPoolExecutor() as executor:  # [1]
            for fitness, (g_id, genome) in zip(executor.map(self.eval_function, process_data), genomes):

                if fitness > max_fitness:
                    max_fitness = fitness

                genome.fitness = fitness

        self.evaluations += 1


def evaluate(attributes):
    g_id, genome, config = attributes
    network = FeedForwardNetwork(genome, config)
    network.initialize(Activations(), Aggregations())

    fitness = []
    env = gym.make('CartPole-v1')  # [1], [2]
    for e_idx in range(100):
        network.reset(hard=True)
        observation = env.reset()
        episode_reward = 0
        while True:
            action = network.activate(observation.tolist())[0]
            action = 0 if action < 0.5 else 1
            observation, reward, done, info = env.step(action)
            episode_reward += reward
            if done:
                break
        fitness.append(episode_reward)
    return mean(fitness)


def main():
    config = Config()
    config.initialize('config/config_1.ini')
    reproduction_factory = TraditionalReproduction()
    speciation_factory = TraditionalSpeciation()
    regulation_factory = PhasedComplexityRegulation(config)
    reporting_factory = reporter.Reporter()
    reporting_factory.add_report(stdout.StdOut())
    reporting_factory.add_report(statistics.Statistics())
    population = Population(reproduction_factory, speciation_factory, regulation_factory, reporting_factory)
    parallel_evaluator = ParallelEvaluator(num_workers=4, eval_function=evaluate)
    population.initialize(parallel_evaluator.evaluate, config)
    population.fit()
    best_genome = population.best_genome
    print(best_genome)

    # Champion solution
    env = gym.make('CartPole-v1')  # [1]
    network = FeedForwardNetwork(best_genome, config)
    network.initialize(Activations(), Aggregations())

    fitness = []
    for e_idx in range(100):
        network.reset(hard=True)
        observation = env.reset()
        episode_reward = 0
        while True:
            if e_idx % 10 == 0:
                env.render()
                time.sleep(0.075)
            action = network.activate(observation.tolist())[0]
            action = 0 if action < 0.5 else 1
            observation, reward, done, info = env.step(action)
            episode_reward += reward
            if done:
                break
        fitness.append(episode_reward)
    print("Examination mean fitness [{}]".format(mean(fitness)))


if __name__ == "__main__":
    main()
