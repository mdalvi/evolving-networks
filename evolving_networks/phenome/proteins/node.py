from evolving_networks.phenome.proteins.protein import Protein


class Node(Protein):
    def __init__(self, node_id, node_type, bias, response, activation, aggregation):
        super(Node, self).__init__()
        self.id = node_id
        self.node_type = node_type
        self.bias = bias
        self.response = response
        self.activation = activation
        self.aggregation = aggregation

        
