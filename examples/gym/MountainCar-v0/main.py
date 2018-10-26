"""
# ==============
# References
# ==============

[1] https://gym.openai.com/docs/
[2] https://gym.openai.com/envs/CartPole-v1/

"""
import concurrent.futures

import gym

from evolving_networks.configurations.config import Config
# from evolving_networks.pytorch.phenome.feed_forward import FeedForwardNetwork
from evolving_networks.phenome.recurrent import RecurrentNetwork
from evolving_networks.population import Population
from evolving_networks.reproduction.traditional import Traditional as TraditionalReproduction
from evolving_networks.activations import Activations
from evolving_networks.aggregations import Aggregations
from evolving_networks.math_util import mean
from evolving_networks.regulations.no_regulation import NoRegulation
from evolving_networks.reporting import reporter, stdout, statistics
from evolving_networks.speciation.traditional_fixed import TraditionalFixed as TraditionalFixedSpeciation

gym.logger.set_level(40)


class ParallelEvaluator(object):
    def __init__(self, num_workers, eval_function):
        self.num_workers = num_workers
        self.eval_function = eval_function

    def evaluate(self, genomes, config):
        process_data = [(g_id, genome, config) for g_id, genome in genomes]
        with concurrent.futures.ProcessPoolExecutor(max_workers=self.num_workers) as executor:  # [1]
            for fitness, (g_id, genome) in zip(executor.map(self.eval_function, process_data), genomes):
                genome.fitness = fitness


def evaluate(attributes):
    g_id, genome, config = attributes
    network = RecurrentNetwork(genome, config)
    network.initialize(Activations(), Aggregations())

    fitness = []
    env = gym.make('MountainCar-v0')  # [1], [2]
    for e_idx in range(10):
        episode_reward = 0
        observation = env.reset()
        network.reset(hard=True)
        while True:
            action = network.activate(observation.tolist())
            action = action.index(max(action))
            observation, reward, done, info = env.step(action)
            episode_reward += reward
            if done:
                break
        fitness.append(episode_reward)
    return mean(fitness)


def main():
    config = Config()
    config.initialize('config/config_2.ini')
    reproduction_factory = TraditionalReproduction()
    speciation_factory = TraditionalFixedSpeciation()
    regulation_factory = NoRegulation(config)
    reporting_factory = reporter.Reporter()
    reporting_factory.add_report(stdout.StdOut())
    reporting_factory.add_report(statistics.Statistics())
    population = Population(reproduction_factory, speciation_factory, regulation_factory, reporting_factory)
    parallel_evaluator = ParallelEvaluator(num_workers=4, eval_function=evaluate)
    population.initialize(parallel_evaluator.evaluate, config)
    population.fit()
    best_genome = population.best_genome
    print(best_genome)

    env = gym.make('MountainCar-v0')  # [1]
    env = gym.wrappers.Monitor(env, 'results', force=True)
    network = RecurrentNetwork(best_genome, config)
    network.initialize(Activations(), Aggregations())

    fitness = []
    for e_idx in range(10):
        episode_reward = 0
        observation = env.reset()
        network.reset(hard=True)
        while True:
            env.render()
            action = network.activate(observation.tolist())
            action = action.index(max(action))
            observation, reward, done, info = env.step(action)
            episode_reward += reward
            if done:
                break
        fitness.append(episode_reward)
    env.close()
    env.env.close()
    fitness_reward = mean(fitness)
    print("Examination mean fitness [{}]".format(fitness_reward))


if __name__ == "__main__":
    main()
