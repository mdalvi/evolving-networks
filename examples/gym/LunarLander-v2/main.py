"""
# ==============
# References
# ==============

[1] https://gym.openai.com/docs/
[2] https://gym.openai.com/envs/CartPole-v1/

"""
import concurrent.futures

import gym

from evolving_networks.activations.activations import Activations
from evolving_networks.aggregations import Aggregations
from evolving_networks.complexity_regulation.phased import Phased as PhasedComplexityRegulation
from evolving_networks.config import Config
from evolving_networks.math_util import mean
from evolving_networks.phenome.feed_forward import FeedForwardNetwork
from evolving_networks.population import Population
from evolving_networks.reproduction.traditional import Traditional as TraditionalReproduction
from evolving_networks.speciation.traditional_fixed import TraditionalFixed as TraditionalFixedSpeciation

gym.logger.set_level(40)


class ParallelEvaluator(object):
    def __init__(self, num_workers, eval_function):
        self.evaluations = 0
        self.num_workers = num_workers
        self.eval_function = eval_function

    def evaluate(self, genomes, config):
        max_fitness = float('-Infinity')

        process_data = [(g_id, genome, config) for g_id, genome in genomes]
        with concurrent.futures.ProcessPoolExecutor(max_workers=self.num_workers) as executor:  # [1]
            for fitness, (g_id, genome) in zip(executor.map(self.eval_function, process_data), genomes):

                if fitness > max_fitness:
                    max_fitness = fitness

                genome.fitness = fitness

        self.evaluations += 1


def evaluate(attributes):
    g_id, genome, config = attributes
    recur_network = FeedForwardNetwork(genome, config)
    recur_network.initialize(Activations(), Aggregations())

    fitness = []
    env = gym.make('LunarLander-v2')  # [1], [2]
    for e_idx in range(10):
        episode_reward = 0
        observation = env.reset()
        recur_network.reset(hard=True)
        while True:
            action = recur_network.activate(observation.tolist())
            action = action.index(max(action))
            observation, reward, done, info = env.step(action)
            episode_reward += reward
            if done:
                break
        fitness.append(episode_reward)
    return mean(fitness)


def main():
    config = Config(filename='config/config_2.ini')
    reproduction_factory = TraditionalReproduction()
    speciation_factory = TraditionalFixedSpeciation()
    complexity_regulation_factory = PhasedComplexityRegulation(config)
    population = Population(reproduction_factory, speciation_factory, complexity_regulation_factory)
    parallel_evaluator = ParallelEvaluator(num_workers=4, eval_function=evaluate)
    population.initialize(parallel_evaluator.evaluate, config)

    while True:
        population.fit(10)
        best_genome = population.best_genome
        print(best_genome)

        env = gym.make('LunarLander-v2')  # [1]
        recur_network = FeedForwardNetwork(best_genome, config)
        recur_network.initialize(Activations(), Aggregations())

        fitness = []
        for e_idx in range(100):
            episode_reward = 0
            observation = env.reset()
            recur_network.reset(hard=True)
            while True:
                if e_idx % 10 == 0:
                    env.render()
                action = recur_network.activate(observation.tolist())
                action = action.index(max(action))
                observation, reward, done, info = env.step(action)
                episode_reward += reward
                if done:
                    break
            fitness.append(episode_reward)
        env.close()
        fitness_reward = mean(fitness)
        print("Examination mean fitness [{}]".format(fitness_reward))

        if fitness_reward >= config.neat.fitness_threshold:
            break


if __name__ == "__main__":
    main()
