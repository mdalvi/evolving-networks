import random
from itertools import count

from evolving_networks.errors import InvalidConfigurationError, InvalidConditionalError
from evolving_networks.genome.genes.connection import Connection
from evolving_networks.genome.genes.node import Node
from evolving_networks.genome.helpers import is_cyclic
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

        self.node_ids = {'all': set(), 'input': set(), 'hidden': set(), 'output': set()}

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
        fitness_case = 'unequal'
        if parent_1.adjusted_fitness > parent_2.adjusted_fitness:
            p1, p2 = parent_1, parent_2
        elif parent_2.adjusted_fitness > parent_1.adjusted_fitness:
            p1, p2 = parent_2, parent_1
        else:
            fitness_case = 'equal'
            if random.random() < 0.5:
                p1, p2 = parent_1, parent_2
            else:
                p1, p2 = parent_2, parent_1

        connection_set_1 = p1.connections
        connection_set_2 = p2.connections

        c_list_1 = sorted(list(connection_set_1.keys()))
        c_list_2 = sorted(list(connection_set_2.keys()))

        if len(c_list_1) < len(c_list_2):
            c_list_1.extend([-1] * (max(len(c_list_1), len(c_list_2)) - min(len(c_list_1), len(c_list_2))))
        else:
            c_list_2.extend([-1] * (max(len(c_list_1), len(c_list_2)) - min(len(c_list_1), len(c_list_2))))

        matched_connections, unmatched_connections = [], []
        for x, y in zip(c_list_1, c_list_2):
            if x == y or x == -1 or y == -1:
                matched_connections.append((x, y))
            else:
                unmatched_connections.append((x, -1))
                unmatched_connections.append((-1, y))

        required_nodes = set()
        for c1_idx, c2_idx in matched_connections:
            c1 = connection_set_1[c1_idx]
            c2 = connection_set_2[c2_idx]
            new_c = c1.crossover(c2)
            assert new_c.id not in self.connections
            self.connections[new_c.id] = new_c
            required_nodes.add(new_c.source_id)
            required_nodes.add(new_c.target_id)

        for c1_idx, c2_idx in unmatched_connections:
            if fitness_case == 'equal':
                if random.random < 0.5:
                    if c1_idx != -1:
                        c = connection_set_1[c1_idx]
                        if config.genome.feed_forward:
                            if is_cyclic(self.connections, c.source_id, c.target_id) is True:
                                continue
                        self._create_connection(c.source_id, c.target_id, c.weight, c.enabled)
                        required_nodes.add(c.source_id)
                        required_nodes.add(c.target_id)
                else:
                    if c2_idx != -1:
                        c = connection_set_2[c2_idx]
                        if config.genome.feed_forward:
                            if is_cyclic(self.connections, c.source_id, c.target_id) is True:
                                continue
                        self._create_connection(c.source_id, c.target_id, c.weight, c.enabled)
                        required_nodes.add(c.source_id)
                        required_nodes.add(c.target_id)
            else:
                if c1_idx != -1:
                    c = connection_set_1[c1_idx]
                    if config.genome.feed_forward:
                        if is_cyclic(self.connections, c.source_id, c.target_id) is True:
                            continue
                    self._create_connection(c.source_id, c.target_id, c.weight, c.enabled)
                    required_nodes.add(c.source_id)
                    required_nodes.add(c.target_id)

        node_set_1 = p1.nodes
        node_set_2 = p2.nodes

        n_list_1 = sorted(list(node_set_1.keys()))
        n_list_2 = sorted(list(node_set_2.keys()))

        if len(n_list_1) < len(n_list_1):
            n_list_1.extend([-1] * (max(len(n_list_1), len(n_list_2)) - min(len(n_list_1), len(n_list_2))))
        else:
            n_list_2.extend([-1] * (max(len(n_list_1), len(n_list_2)) - min(len(n_list_1), len(n_list_2))))

        node_pairs = []
        for x, y in zip(n_list_1, n_list_2):
            if x == y or x == -1 or y == -1:
                node_pairs.append((x, y))
            else:
                node_pairs.append((x, -1))
                node_pairs.append((-1, y))

        for n1_idx, n2_idx in node_pairs:
            if n1_idx == n2_idx:
                n1 = node_set_1[n1_idx]
                n2 = node_set_2[n2_idx]
                assert n1.type == n2.type
                if n1_idx in required_nodes or n1.type == 'input' or n1.type == 'output':
                    new_n = n1.crossover(n2)
                    assert new_n.id not in self.nodes
                    self.nodes[new_n.id] = new_n
                    self.node_ids['all'].add(new_n.id)
                    self.node_ids[new_n.type].add(new_n.id)
            elif n1_idx != n2_idx and n1_idx != -1 and n2_idx == -1:
                n = node_set_1[n1_idx]
                if n1_idx in required_nodes or n.type == 'input' or n.type == 'output':
                    assert n.id not in self.nodes
                    self._create_node(n.id, n.type, n.bias, n.response, n.activation, n.aggregation)
                    self.node_ids['all'].add(n.id)
                    self.node_ids[n.type].add(n.id)
            elif n1_idx != n2_idx and n2_idx != -1 and n1_idx == -1:
                n = node_set_2[n2_idx]
                if n2_idx in required_nodes or n.type == 'input' or n.type == 'output':
                    assert n.id not in self.nodes
                    self._create_node(n.id, n.type, n.bias, n.response, n.activation, n.aggregation)
                    self.node_ids['all'].add(n.id)
                    self.node_ids[n.type].add(n.id)
            else:
                raise InvalidConditionalError()

        self.node_indexer = count(max(parent_1.node_indexer_cntr, parent_2.node_indexer_cntr) + 1)
        self.node_indexer_cntr = max(parent_1.node_indexer_cntr, parent_2.node_indexer_cntr)

    def crossover_asexual(self, parent_1):
        for node in parent_1.nodes.values():
            assert node.id not in self.nodes
            self._create_node(node.id, node.type, node.bias, node.response, node.activation, node.aggregation)
            self.node_ids['all'].add(node.id)
            self.node_ids[node.type].add(node.id)

        for connection in parent_1.connections.values():
            self._create_connection(connection.source_id, connection.target_id, connection.weight, connection.enabled)

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
            self.node_ids['input'].add(n_id)
            self.node_ids['all'].add(n_id)

        for _ in range(self.config.num_hidden):
            n_id = self._next_node_id()
            self._create_node(n_id, 'hidden', config=node_config)
            self.node_ids['hidden'].add(n_id)
            self.node_ids['all'].add(n_id)

        for _ in range(self.config.num_outputs):
            n_id = self._next_node_id()
            self._create_node(n_id, 'output', config=node_config)
            self.node_ids['output'].add(n_id)
            self.node_ids['all'].add(n_id)

        if self.config.initial_connection == 'fs_neat_no_hidden':
            source_id = random.choice(self.node_ids['input'])
            for target_id in self.node_ids['output']:
                self._create_connection(source_id, target_id, config=connection_config)

        elif self.config.initial_connection == 'fs_neat_hidden':
            source_id = random.choice(self.node_ids['input'])
            for target_id in set().union(self.node_ids['hidden'], self.node_ids['output']):
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
        if self.node_ids['hidden']:
            for source_id in self.node_ids['input']:
                for target_id in self.node_ids['hidden']:
                    connectors.append((source_id, target_id))
            for source_id in self.node_ids['hidden']:
                for target_id in self.node_ids['output']:
                    connectors.append((source_id, target_id))
        if direct or (not self.node_ids['hidden']):
            for source_id in self.node_ids['input']:
                for target_id in self.node_ids['output']:
                    connectors.append((source_id, target_id))

        # TODO: Recurrent networks
        # For recurrent genomes, include node self-connections.
        # if not self.config.feed_forward:
        #     for recurrent_id in set().union(self.node_ids['hidden'], self.node_ids['output']):
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
