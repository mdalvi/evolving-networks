import time

from tabulate import tabulate

from evolving_networks.math_util import mean
from evolving_networks.reporting.report import Report


class Statistics(Report):
    def __init__(self):
        super(Statistics, self).__init__()
        self.generations = []
        self.best_genome = None
        self.current_best = None
        self.generation_start_time = None

        self.max_complexity = []
        self.min_complexity = []
        self.mean_complexity = []
        self.elapsed_generation_time = []

    def start_generation(self, generation):
        self.generation_start_time = time.time()
        self.generations.append(generation)

    def end_generation(self):
        self.elapsed_generation_time.append(time.time() - self.generation_start_time)

    def found_solution(self, best_genome, generation):
        pass

    def pre_speciation(self):
        pass

    def post_speciation(self, speciation, regulation, complexity):
        self.mean_complexity.append(mean(complexity))
        self.min_complexity.append(min(complexity))
        self.max_complexity.append(max(complexity))

        details = {'Id': [], 'Size': [], 'Fitness': [], 'AdjFitness': [], 'LastImproved': []}
        for s_id, specie in speciation.species.items():
            details['Id'].append(s_id)
            details['Size'].append(len(specie))
            details['Fitness'].append(specie.fitness)
            details['AdjFitness'].append(specie.adjusted_fitness)
            details['LastImproved'].append(specie.last_improved)

        if self.best_genome is None or speciation.best_genome.fitness > self.best_genome.fitness:
            self.best_genome = speciation.best_genome
        self.current_best = speciation.best_genome

        print("\nTotal number of species {0}".format(len(speciation.species)))
        print("Lifetime best fitness {0} "
              "@ complexity {1}".format(self.best_genome.fitness, self.best_genome.complexity))
        print("Current best fitness {0}"
              " @ complexity {1}".format(self.current_best.fitness, self.current_best.complexity))

        print(tabulate(details, headers="keys", numalign="right"))
        print(regulation)

    def pre_evaluation(self):
        pass

    def post_evaluation(self):
        pass

    def pre_reproduction(self):
        pass

    def post_reproduction(self):
        pass
