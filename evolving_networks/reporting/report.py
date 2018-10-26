class Report(object):
    def __init__(self):
        pass

    def start_generation(self, generation):
        pass

    def end_generation(self):
        pass

    def found_solution(self, best_genome, generation):
        pass

    def pre_speciation(self):
        pass

    def post_speciation(self, speciation, regulation, complexity):
        pass

    def pre_evaluation(self):
        pass

    def post_evaluation(self):
        pass

    def pre_reproduction(self):
        pass

    def post_reproduction(self):
        pass
