class Reporter(object):
    def __init__(self):
        self.reporters = []

    def add_report(self, report):
        self.reporters.append(report)

    def remove_report(self, report):
        self.reporters.remove(report)

    def start_generation(self, generation):
        for r in self.reporters:
            r.start_generation(generation)

    def end_generation(self):
        for r in self.reporters:
            r.end_generation()

    def found_solution(self, best_genome, generation):
        for r in self.reporters:
            r.found_solution(best_genome, generation)

    def pre_speciation(self):
        for r in self.reporters:
            r.pre_speciation()

    def post_speciation(self, speciation, regulation, complexity):
        for r in self.reporters:
            r.post_speciation(speciation, regulation, complexity)

    def pre_evaluation(self):
        for r in self.reporters:
            r.pre_evaluation()

    def post_evaluation(self):
        for r in self.reporters:
            r.post_evaluation()

    def pre_reproduction(self):
        for r in self.reporters:
            r.pre_reproduction()

    def post_reproduction(self):
        for r in self.reporters:
            r.post_reproduction()
