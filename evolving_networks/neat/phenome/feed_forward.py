from evolving_networks.neat.phenome.helpers import calc_required_acyclic_depth, calc_neural_acyclic_path
from evolving_networks.neat.phenome.phenome import Phenome
from evolving_networks.neat.phenome.proteins.node import Node


class FeedForwardNetwork(Phenome):
    def __init__(self, genome, config):
        super(FeedForwardNetwork, self).__init__()
        self.genome = genome
        self.config = config

        # Node to type mappings
        self.node_to_type = {}

        # Neural activation pathways of output nodes
        self.neuronal_paths = {}

        # Node protein collections
        self.nodes = {'input': {}, 'hidden': {}, 'output': {}}

        # Flag for genome removal
        self.is_damaged = False

    def initialize(self, activations, aggregations):
        # Set of nodes mandatory for output activation
        required_nodes = set()

        # The complete list of required (enabled) connections as (id, source_id, target_id)
        enabled_connections = [(connection.id, connection.source_id, connection.target_id) for connection in
                               self.genome.connections.values() if connection.enabled]

        depth = calc_required_acyclic_depth(self.genome.node_ids['all'], enabled_connections)
        for n_id in self.genome.node_ids['output']:
            path = []
            calc_neural_acyclic_path(depth, n_id, path)
            required_nodes.update(path)
            self.neuronal_paths[n_id] = path

        # Create required node proteins
        for n_id in required_nodes:
            # Node gene
            g_node = self.genome.nodes[n_id]
            activation = activations.get(g_node.activation)
            aggregation = aggregations.get(g_node.aggregation)

            # Node protein
            p_node = Node(n_id, g_node.type, g_node.bias, g_node.response, activation, aggregation)

            # A list of incoming weighted signals
            incoming = []

            for (c_id, source_id, target_id) in enabled_connections:

                # If target is required node then we have incoming connection dependency
                if target_id == n_id:
                    incoming.append((source_id, self.genome.connections[c_id].weight))

            p_node.incoming = incoming
            self.nodes[p_node.type][n_id] = p_node
            self.node_to_type[n_id] = p_node.type

    def activate(self, inputs):
        if len(self.genome.node_ids['input']) != len(inputs):
            raise RuntimeError("Unexpected number of inputs")

        # Assigning incoming to input node proteins
        for p_node, input_val in zip(self.nodes['input'].values(), inputs):
            # Special case incoming format for input nodes
            p_node.incoming = [input_val]

        # Feed forward reset
        self.reset()

        try:
            # Activating ordered neural pathways
            for path in self.neuronal_paths.values():
                for n_id in path:
                    # Get node protein
                    p_node = self.nodes[self.node_to_type[n_id]][n_id]

                    # Only do if node isn't activated already
                    if p_node.activated is False:

                        # Special case activation for input nodes
                        if p_node.type == 'input':
                            p_node.activate(p_node.incoming)
                        else:
                            # Creating weighted incoming signals
                            incoming = [self.nodes[self.node_to_type[i_id]][i_id].outgoing * weight for (i_id, weight)
                                        in
                                        p_node.incoming]
                            p_node.activate(incoming)
        except (OverflowError, ValueError):
            self.is_damaged = True
            return [0.0 for _ in self.nodes['output'].values()]
        return [p_node.outgoing for p_node in self.nodes['output'].values()]

    def reset(self, hard=False):
        for node_dict in self.nodes.values():
            for p_node in node_dict.values():
                p_node.outgoing = 0.0
                p_node.activated = False
