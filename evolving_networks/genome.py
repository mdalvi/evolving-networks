class Genome(object):
    def __init__(self, genome_id):
        self.id = genome_id
        self.nodes = {}
        self.connections = {}
        self.fitness = None
