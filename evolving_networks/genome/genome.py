import random
from itertools import count

from evolving_networks.errors import InvalidConfigurationError
from evolving_networks.genome.genes.connection import Connection
from evolving_networks.genome.genes.node import Node
from evolving_networks.math_util import normalize


class Genome(object):
    _innovation_indexer = count(0)
    _innovation_archive = {}

    def __init__(self, g_id, config):
        self.id = g_id
        self.nodes = {}
        self.connections = {}
        self.fitness = None
        self.config = config

        self.all_keys = set()
        self.input_keys = set()
        self.hidden_keys = set()
        self.output_keys = set()
        self._node_indexer = count(0)

    def distance(self, other_genome, config):
        node_distance, connection_distance = 0.0, 0.0
        if self.nodes or other_genome.nodes:
            nb_max_nodes = max(len(self.nodes), len(other_genome.nodes))
            matching_nodes = list(set(self.nodes.keys()) & set(other_genome.nodes.keys()))
            disjoint_excess_nodes = list(set(self.nodes.keys()) ^ set(other_genome.nodes.keys()))

            for n_id in matching_nodes:
                node_distance += self.nodes[n_id].distance(other_genome.nodes[n_id], config.node)

            if len(matching_nodes) > 0:
                node_distance = node_distance / len(matching_nodes)

            node_distance += normalize(0.0, nb_max_nodes, len(disjoint_excess_nodes), 0.0,
                                       1.0) * config.node.compatibility_disjoint_contribution
            node_distance = node_distance / 2.0
            assert 0.0 <= node_distance <= 1.0

        if self.connections or other_genome.connections:
            nb_max_connections = max(len(self.connections), len(other_genome.connections))
            matching_connections = list(set(self.connections.keys()) & set(other_genome.connections.keys()))
            disjoint_excess_connections = list(set(self.connections.keys()) ^ set(other_genome.connections.keys()))

            for c_id in matching_connections:
                connection_distance += self.connections[c_id].distance(other_genome.connections[c_id],
                                                                       config.connection)
            if len(matching_connections) > 0:
                connection_distance = connection_distance / len(matching_connections)

            connection_distance += normalize(0.0, nb_max_connections, len(disjoint_excess_connections), 0.0,
                                             1.0) * config.connection.compatibility_disjoint_contribution
            connection_distance = connection_distance / 2.0
            assert 0.0 <= connection_distance <= 1.0

        genomic_distance = (node_distance + connection_distance) / 2.0
        assert 0.0 <= genomic_distance <= 1.0
        return genomic_distance

    def initialize(self, node_config, connection_config):
        for _ in range(self.config.num_inputs):
            n_id = next(self._node_indexer)
            assert n_id not in self.nodes
            self._create_node(n_id, 'input', node_config)
            self.input_keys.add(n_id)
            self.all_keys.add(n_id)

        for _ in range(self.config.num_hidden):
            n_id = next(self._node_indexer)
            assert n_id not in self.nodes
            self._create_node(n_id, 'hidden', node_config)
            self.hidden_keys.add(n_id)
            self.all_keys.add(n_id)

        for _ in range(self.config.num_outputs):
            n_id = next(self._node_indexer)
            assert n_id not in self.nodes
            self._create_node(n_id, 'output', node_config)
            self.output_keys.add(n_id)
            self.all_keys.add(n_id)

        if self.config.initial_connection == 'fs_neat_no_hidden':
            source_id = random.choice(self.input_keys)
            for target_id in self.output_keys:
                self._create_connection(source_id, target_id, connection_config)

        elif self.config.initial_connection == 'fs_neat_hidden':
            source_id = random.choice(self.input_keys)
            for target_id in set().union(self.hidden_keys, self.output_keys):
                self._create_connection(source_id, target_id, connection_config)

        elif self.config.initial_connection == 'full_no_direct':
            for source_id, target_id in self._compute_full_connectors(False):
                self._create_connection(source_id, target_id, connection_config)

        elif self.config.initial_connection == 'full_direct':
            for source_id, target_id in self._compute_full_connectors(True):
                self._create_connection(source_id, target_id, connection_config)

        elif self.config.initial_connection == 'partial_no_direct':
            connectors = self._compute_full_connectors(False)
            random.shuffle(connectors)
            connections_to_add = int(round(len(connectors) * self.config.partial_connection_rate))
            for source_id, target_id in connectors[:connections_to_add]:
                self._create_connection(source_id, target_id, connection_config)

        elif self.config.initial_connection == 'partial_direct':
            connectors = self._compute_full_connectors(True)
            random.shuffle(connectors)
            connections_to_add = int(round(len(connectors) * self.config.partial_connection_rate))
            for source_id, target_id in connectors[:connections_to_add]:
                self._create_connection(source_id, target_id, connection_config)

        elif self.config.initial_connection == 'unconnected':
            pass

        else:
            raise InvalidConfigurationError(
                'UNEXPECTED CONFIGURATION VALUE [{}]'.format(self.config.initial_connection))

    def _compute_full_connectors(self, direct):
        connectors = []
        if self.hidden_keys:
            for source_id in self.input_keys:
                for target_id in self.hidden_keys:
                    connectors.append((source_id, target_id))
            for source_id in self.hidden_keys:
                for target_id in self.output_keys:
                    connectors.append((source_id, target_id))
        if direct or (not self.hidden_keys):
            for source_id in self.input_keys:
                for target_id in self.output_keys:
                    connectors.append((source_id, target_id))

        # TODO: Recurrent networks
        # For recurrent genomes, include node self-connections.
        # if not self.config.feed_forward:
        #     for recurrent_id in set().union(self.hidden_keys, self.output_keys):
        #         connectors.append((recurrent_id, recurrent_id))

        return connectors

    def _create_node(self, n_id, node_type, config):
        node = Node(n_id, node_type)
        node.initialize(config)
        self.nodes[n_id] = node

    def _create_connection(self, source_id, target_id, config):
        if (source_id, target_id) in self.__class__._innovation_archive:
            c_id = self.__class__._innovation_archive[(source_id, target_id)]
        else:
            c_id = next(self.__class__._innovation_indexer)
            self.__class__._innovation_archive[(source_id, target_id)] = c_id

        assert c_id not in self.connections
        connection = Connection(c_id, source_id, target_id)
        connection.initialize(config)
        self.connections[c_id] = connection

    def crossover(self):
        pass

    def mutate(self):
        pass
