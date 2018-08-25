"""
# ==============
# References
# ==============

[1] https://gym.openai.com/docs/
[2] https://gym.openai.com/envs/CartPole-v1/

"""
import concurrent.futures

import gym
import torch.nn as nn
import torch.nn.functional as F

from evolving_networks.math_util import mean
from evolving_networks.pytorch.configurations.config import Config
from evolving_networks.pytorch.phenome.feed_forward import FeedForwardNetwork
# from evolving_networks.pytorch.phenome.recurrent import RecurrentNetwork
from evolving_networks.pytorch.population import Population
from evolving_networks.pytorch.reproduction.traditional import Traditional as TraditionalReproduction
from evolving_networks.regulations.no_regulation import NoRegulation
from evolving_networks.reporting import reporter, stdout
from evolving_networks.speciation.traditional_fixed import TraditionalFixed as TraditionalFixedSpeciation

gym.logger.set_level(40)


class FeedForwardModel(nn.Module):
    def __init__(self):
        super(FeedForwardModel, self).__init__()
        self.fc1 = nn.Linear(in_features=8, out_features=8)
        self.fc2 = nn.Linear(in_features=8, out_features=8)
        self.fc3 = nn.Linear(in_features=8, out_features=4)

    def forward(self, x):
        fc1_output = F.relu(self.fc1(x))
        fc2_output = F.relu(self.fc2(fc1_output))
        fc3_output = F.softmax(self.fc3(fc2_output), dim=0)
        return fc3_output


class RecurrentModel(nn.Module):
    def __init__(self):
        super(RecurrentModel, self).__init__()
        self.rnn = nn.RNN(input_size=8, hidden_size=8, num_layers=1, nonlinearity='relu')
        self.fc1 = nn.Linear(in_features=8, out_features=4)

        self.hidden = None

    def forward(self, x):
        rnn_output, self.hidden = self.rnn(x, self.hidden)
        fc1_output = F.softmax(self.fc1(rnn_output), dim=2)
        return fc1_output

    def reset(self):
        self.hidden = None


class ParallelEvaluator(object):
    def __init__(self, num_workers, eval_function):
        self.evaluations = 0
        self.num_workers = num_workers
        self.eval_function = eval_function

    def evaluate(self, genomes, config):
        process_data = [(g_id, genome, config) for g_id, genome in genomes]
        with concurrent.futures.ProcessPoolExecutor(max_workers=self.num_workers) as executor:  # [1]
            for fitness, (g_id, genome) in zip(executor.map(self.eval_function, process_data), genomes):
                genome.fitness = fitness
        self.evaluations += 1


def evaluate(attributes):
    g_id, genome, config = attributes
    network = FeedForwardNetwork(genome, config)
    network.initialize(FeedForwardModel)

    fitness = []
    env = gym.make('LunarLander-v2')  # [1], [2]
    for e_idx in range(10):
        e_step = 0
        episode_reward = 0
        observation = env.reset()
        network.reset(hard=True)
        while True:
            action = network.activate(observation.tolist())
            action = action.index(max(action))
            observation, reward, done, info = env.step(action)
            episode_reward += reward
            e_step += 1
            if done:
                break
        fitness.append(episode_reward)
    return mean(fitness)


def main():
    config = Config(filename='config/config_3.ini')
    reproduction_factory = TraditionalReproduction(FeedForwardModel)
    speciation_factory = TraditionalFixedSpeciation()
    regulation_factory = NoRegulation(config)
    reporting_factory = reporter.Reporter()
    reporting_factory.add_report(stdout.StdOut())
    population = Population(reproduction_factory, speciation_factory, regulation_factory, reporting_factory)
    parallel_evaluator = ParallelEvaluator(num_workers=4, eval_function=evaluate)
    population.initialize(parallel_evaluator.evaluate, config)

    while True:
        population.fit(25)
        best_genome = population.best_genome
        print(best_genome)

        env = gym.make('LunarLander-v2')  # [1]
        env = gym.wrappers.Monitor(env, 'results', force=True)
        network = FeedForwardNetwork(best_genome, config)
        network.initialize(FeedForwardModel)

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

        if fitness_reward >= config.pytorch.fitness_threshold:
            break


if __name__ == "__main__":
    main()
