class Genome(object):
    def __init__(self, genome_id, config):
        self.id = genome_id
        self.nodes = {}
        self.connections = {}
        self.fitness = None
        self.config = config

        for _ in range(self.config.num_inputs):
            pass

    def crossover(self):
        pass

    def mutate(self):
        pass

    def create_node(self):
        pass
