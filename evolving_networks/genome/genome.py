import random
from itertools import count

from evolving_networks.errors import InvalidConfigurationError
from evolving_networks.genome.genes.connection import Connection
from evolving_networks.genome.genes.node import Node
from evolving_networks.math_util import normalize


class Genome(object):
    _innovation_indexer = count(0)
    _innovation_archive = {}

    def __init__(self, g_id, generation, config):
        self.node_indexer = count(0)
        self.node_indexer_cntr = 0

        self.id = g_id
        self.config = config
        self.birth_generation = generation

        self.nodes = {}
        self.connections = {}
        self.fitness = 0.0
        self.adjusted_fitness = 0.0

        self.all_node_ids = set()
        self.input_node_ids = set()
        self.hidden_node_ids = set()
        self.output_node_ids = set()

    def __lt__(self, other):
        if self.fitness == other.fitness:
            return self.birth_generation < other.birth_generation
        return self.fitness < other.fitness

    def __le__(self, other):
        if self.fitness == other.fitness:
            return self.birth_generation <= other.birth_generation
        return self.fitness <= other.fitness

    def __gt__(self, other):
        if self.fitness == other.fitness:
            return self.birth_generation > other.birth_generation
        return self.fitness > other.fitness

    def __ge__(self, other):
        if self.fitness == other.fitness:
            return self.birth_generation >= other.birth_generation
        return self.fitness >= other.fitness

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

    def crossover_sexual(self, parent_1, parent_2, config):
        if parent_1.adjusted_fitness > parent_2.adjusted_fitness:
            p1, p2 = parent_1, parent_2
        elif parent_2.adjusted_fitness > parent_1.adjusted_fitness:
            p1, p2 = parent_2, parent_1
        else:
            if random.random() < 0.5:
                p1, p2 = parent_1, parent_2
            else:
                p1, p2 = parent_2, parent_1

        for node in p1.nodes.values():
            if node.type == 'input' or node.type == 'output':
                assert node.id not in self.nodes
                self._create_node(node.id, node.type, node.bias, node.response, node.activation, node.aggregation)
                self.all_node_ids.add(node.id)
                if node.type == 'input':
                    self.input_node_ids.add(node.id)
                elif node.type == 'output':
                    self.output_node_ids.add(node.id)
                else:
                    raise InvalidConfigurationError('Unexpected configuration value [{}]'.format(node.type))



    def crossover_asexual(self, parent_1):
        for node in parent_1.nodes.values():
            assert node.id not in self.nodes
            self._create_node(node.id, node.type, node.bias, node.response, node.activation, node.aggregation)
            self.all_node_ids.add(node.id)
            if node.type == 'input':
                self.input_node_ids.add(node.id)
            elif node.type == 'hidden':
                self.hidden_node_ids.add(node.id)
            elif node.type == 'output':
                self.output_node_ids.add(node.id)
            else:
                raise InvalidConfigurationError('Unexpected configuration value [{}]'.format(node.type))

        for connection in parent_1.connections.values():
            self._create_connection(connection.source_id, connection.target_id, connection.weight, connection.enabled)

        self.fitness = parent_1.fitness
        self.adjusted_fitness = parent_1.adjusted_fitness
        self.node_indexer = count(parent_1.node_indexer_cntr + 1)
        self.node_indexer_cntr = parent_1.node_indexer_cntr

    def _next_node_id(self):
        n_id = next(self.node_indexer)
        self.node_indexer_cntr = n_id
        assert n_id not in self.nodes
        return n_id

    def initialize(self, node_config, connection_config):
        for _ in range(self.config.num_inputs):
            n_id = self._next_node_id()
            self._create_node(n_id, 'input', config=node_config)
            self.input_node_ids.add(n_id)
            self.all_node_ids.add(n_id)

        for _ in range(self.config.num_hidden):
            n_id = self._next_node_id()
            self._create_node(n_id, 'hidden', config=node_config)
            self.hidden_node_ids.add(n_id)
            self.all_node_ids.add(n_id)

        for _ in range(self.config.num_outputs):
            n_id = self._next_node_id()
            self._create_node(n_id, 'output', config=node_config)
            self.output_node_ids.add(n_id)
            self.all_node_ids.add(n_id)

        if self.config.initial_connection == 'fs_neat_no_hidden':
            source_id = random.choice(self.input_node_ids)
            for target_id in self.output_node_ids:
                self._create_connection(source_id, target_id, config=connection_config)

        elif self.config.initial_connection == 'fs_neat_hidden':
            source_id = random.choice(self.input_node_ids)
            for target_id in set().union(self.hidden_node_ids, self.output_node_ids):
                self._create_connection(source_id, target_id, config=connection_config)

        elif self.config.initial_connection == 'full_no_direct':
            for source_id, target_id in self._compute_full_connectors(False):
                self._create_connection(source_id, target_id, config=connection_config)

        elif self.config.initial_connection == 'full_direct':
            for source_id, target_id in self._compute_full_connectors(True):
                self._create_connection(source_id, target_id, config=connection_config)

        elif self.config.initial_connection == 'partial_no_direct':
            connectors = self._compute_full_connectors(False)
            random.shuffle(connectors)
            connections_to_add = int(round(len(connectors) * self.config.partial_connection_rate))
            for source_id, target_id in connectors[:connections_to_add]:
                self._create_connection(source_id, target_id, config=connection_config)

        elif self.config.initial_connection == 'partial_direct':
            connectors = self._compute_full_connectors(True)
            random.shuffle(connectors)
            connections_to_add = int(round(len(connectors) * self.config.partial_connection_rate))
            for source_id, target_id in connectors[:connections_to_add]:
                self._create_connection(source_id, target_id, config=connection_config)

        elif self.config.initial_connection == 'unconnected':
            pass

        else:
            raise InvalidConfigurationError(
                'Unexpected configuration value [{}]'.format(self.config.initial_connection))

    def _compute_full_connectors(self, direct):
        # TODO: Ensure function does not create cyclic connections unknowingly
        connectors = []
        if self.hidden_node_ids:
            for source_id in self.input_node_ids:
                for target_id in self.hidden_node_ids:
                    connectors.append((source_id, target_id))
            for source_id in self.hidden_node_ids:
                for target_id in self.output_node_ids:
                    connectors.append((source_id, target_id))
        if direct or (not self.hidden_node_ids):
            for source_id in self.input_node_ids:
                for target_id in self.output_node_ids:
                    connectors.append((source_id, target_id))

        # TODO: Recurrent networks
        # For recurrent genomes, include node self-connections.
        # if not self.config.feed_forward:
        #     for recurrent_id in set().union(self.hidden_node_ids, self.output_node_ids):
        #         connectors.append((recurrent_id, recurrent_id))

        return connectors

    def _create_node(self, n_id, n_type, bias=None, response=None, activation=None, aggregation=None, config=None):
        node = Node(n_id, n_type, bias, response, activation, aggregation)
        if config is not None:
            node.initialize(config)
        self.nodes[n_id] = node

    def _create_connection(self, source_id, target_id, weight=None, enabled=None, config=None):
        if (source_id, target_id) in self.__class__._innovation_archive:
            c_id = self.__class__._innovation_archive[(source_id, target_id)]
        else:
            c_id = next(self.__class__._innovation_indexer)
            self.__class__._innovation_archive[(source_id, target_id)] = c_id

        assert c_id not in self.connections
        connection = Connection(c_id, source_id, target_id, weight, enabled)
        if config is not None:
            connection.initialize(config)
        self.connections[c_id] = connection

    def mutate(self):
        pass
