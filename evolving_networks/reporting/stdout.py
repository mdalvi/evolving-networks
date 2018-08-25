import time

from evolving_networks.reporting.reporter import Report


class StdOut(Report):
    def __init__(self):
        super(StdOut, self).__init__()
        self.generation = 0
        self.generation_start_time = time.time()
        self.speciation_start_time = time.time()
        self.evaluation_start_time = time.time()
        self.reproduction_start_time = time.time()

    def start_generation(self, generation):
        self.generation = generation
        self.generation_start_time = time.time()
        print("\n****** Generation {0} ******\n".format(self.generation))

    def end_generation(self):
        print("Elapsed generation time: {0:.2f} sec ".format(time.time() - self.generation_start_time))

    def found_solution(self, best_genome, generation):
        print(best_genome)
        print("***** Found solution at generation {0}".format(generation))

    def pre_speciation(self):
        self.speciation_start_time = time.time()

    def post_speciation(self):
        print("Elapsed speciation time: {0:.2f} sec ".format(time.time() - self.speciation_start_time))

    def pre_evaluation(self):
        self.evaluation_start_time = time.time()

    def post_evaluation(self):
        print("Elapsed evaluation time: {0:.2f} sec ".format(time.time() - self.evaluation_start_time))

    def pre_reproduction(self):
        self.reproduction_start_time = time.time()

    def post_reproduction(self):
        print("Elapsed reproduction time: {0:.2f} sec ".format(time.time() - self.reproduction_start_time))
