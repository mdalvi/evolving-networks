from evolving_networks.phenome.helpers import build_essential_nodes, activation_recursion
from evolving_networks.phenome.phenome import Phenome
from evolving_networks.phenome.proteins.node import Node


class FeedForwardNetwork(Phenome):
    def __init__(self, genome, config):
        super(FeedForwardNetwork, self).__init__()
        self.genome = genome
        self.config = config

        self.activation_paths = {}
        self.e_nodes = {}  # essential nodes
        self.nodes = {'input': dict(), 'hidden': dict(), 'output': dict()}

    def initialize(self, act_func_set, agg_func_set):
        activation_node_set = set()
        enabled_connections = [(conn.id, conn.source_id, conn.target_id) for conn in self.genome.connections.values() if
                               conn.enabled]
        essential_nodes = build_essential_nodes(self.genome.all_keys, enabled_connections)
        for o_key in self.genome.output_keys:
            activation_path = []
            activation_recursion(essential_nodes, o_key, activation_path)
            activation_node_set.update(activation_path)
            self.activation_paths[o_key] = activation_path

        for node_id, node in self.genome.nodes.items():
            self.nodes[node.type][node_id] = Node(node_id, node.type, node.bias, node.response,
                                                  act_func_set.get(node.activation), agg_func_set.get(node.aggregation))
            if node_id in activation_node_set:
                incoming_inputs = []
                for (c_id, s_id, t_id) in enabled_connections:
                    if t_id == node_id:
                        incoming_inputs.append((s_id, self.genome.connections[c_id].weight))
                self.nodes[node.type][node_id].incoming_inputs = incoming_inputs
                self.e_nodes[node_id] = node.type

    def activate(self, inputs):
        if len(self.genome.input_keys) != len(inputs):
            raise RuntimeError(
                "UNEXPECTED NUMBER OF INPUTS [{}] vs [{}]".format(len(inputs), len(self.genome.input_keys)))

        # Assign incoming to input nodes
        for node, ip in zip(self.nodes['input'].values(), inputs):
            node.incoming_inputs = [ip]

        # Reset essential activation nodes
        for node_id, node_type in self.e_nodes.items():
            self.nodes[node_type][node_id].output = 0.0
            self.nodes[node_type][node_id].is_activated = False
            self.nodes[node_type][node_id].is_fired = False

        # Activate nodes based on activation paths
        for activation_path in self.activation_paths.values():
            for key in activation_path:
                node = self.nodes[self.e_nodes[key]][key]
                if node.is_activated is False:
                    if node.type == 'input':
                        node.activate(node.incoming_inputs)
                    else:
                        node.activate(
                            [self.nodes[self.e_nodes[incoming]][incoming].output * weight for (incoming, weight) in
                             node.incoming_inputs])
        return [node.output for node in self.nodes['output'].values()]

    def reset(self):
        # Hard reset
        for node_dict in self.nodes.values():
            for node in node_dict.values():
                node.output, node.is_activated, node.is_fired = 0.0, False, False
