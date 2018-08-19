from evolving_networks.neat.phenome.proteins.protein import Protein


class Node(Protein):
    def __init__(self, node_id, node_type, bias, response, activation, aggregation):
        super(Node, self).__init__()
        self.id = node_id
        self.type = node_type
        self.bias = bias
        self.response = response
        self.activation = activation
        self.aggregation = aggregation

        self.incoming = []
        self.outgoing = 0.0

        self.fired = False
        self.activated = False

    def activate(self, inputs):
        if inputs:
            self.outgoing, self.fired = self.activation.activate((self.aggregation(inputs) * self.response) + self.bias)
        else:
            self.outgoing = 0.0
            self.fired = False
        self.activated = True
