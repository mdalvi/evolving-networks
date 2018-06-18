from evolving_networks.phenome.proteins.protein import Protein


class Node(Protein):
    def __init__(self, node_id, node_type, bias, response, activation, aggregation):
        super(Node, self).__init__()
        self.id = node_id
        self.type = node_type
        self.bias = bias
        self.response = response
        self.activation = activation
        self.aggregation = aggregation

        self.incoming_inputs = []
        self.output = 0.0
        self.is_activated = False
        self.is_fired = False

    def activate(self, inputs):
        if len(inputs) == 0:
            self.output, self.is_fired = 0.0, True
        else:
            self.output, self.is_fired = self.activation.activate(
                (self.aggregation(inputs) * self.response) + self.bias)
        self.is_activated = True
