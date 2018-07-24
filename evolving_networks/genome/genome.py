"""
# ==============
# References
# ==============

[1] https://github.com/CodeReclaimers/neat-python/blob/master/neat/graphs.py
[2] http://nn.cs.utexas.edu/downloads/papers/stanley.ec02.pdf

"""

import random
from itertools import count

import numpy as np

from evolving_networks.errors import InvalidConfigurationError, InvalidConditionalError
from evolving_networks.genome.genes.connection import Connection
from evolving_networks.genome.genes.node import Node
from evolving_networks.math_util import normalize, probabilistic_round


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

        self._connectors = set()
        self._cyclic_connectors = set()
        self._acyclic_connectors = set()
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

    def __str__(self):
        s = "Id: {0}\nFitness: {1}".format(self.id, self.fitness)
        s += "\nNodes:"
        for n_id, node in self.nodes.items():
            s += "\n\t{0} {1!s}".format(n_id, node)
        s += "\nConnections:"
        connections = list(self.connections.values())
        connections.sort()
        for connection in connections:
            s += "\n\t{0!s}".format(connection)
        return s

    def _distance(self, other_genome, config):
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
            nb_max_connections = len(self.connections) + len(other_genome.connections)
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

    def distance(self, other_genome, config):
        dist = 0.0
        c1 = 1.0
        c2 = 1.0
        c3 = 0.4

        conn_1_cnt = len(self.connections)
        conn_2_cnt = len(other_genome.connections)
        if conn_1_cnt != 0 and conn_2_cnt != 0:
            c_list_1 = sorted(list(self.connections.keys()))
            c_list_2 = sorted(list(other_genome.connections.keys()))

            nb_disjoint, nb_excess = 0, 0
            excess_threshold = min(max(c_list_1), max(c_list_2))
            for x in c_list_1:
                if x not in c_list_2:
                    if x <= excess_threshold:
                        nb_disjoint += 1
                    else:
                        nb_excess += 1
            for x in c_list_2:
                if x not in c_list_1:
                    if x <= excess_threshold:
                        nb_disjoint += 1
                    else:
                        nb_excess += 1

            dist += (c1 * normalize(0, max(len(c_list_1), len(c_list_2)), nb_disjoint, 0.0, 1.0))
            dist += (c2 * normalize(0, max(len(c_list_1), len(c_list_2)), nb_excess, 0.0, 1.0))
        elif conn_1_cnt == 0 and conn_2_cnt == 0:
            dist += 0.0
        else:
            dist += (c2 * 1.0)

        c_dist = 0.0
        if conn_1_cnt != 0 or conn_2_cnt != 0:
            matched_connections = list(set(self.connections.keys()) & set(other_genome.connections.keys()))
            for c_id in matched_connections:
                c_dist += self.connections[c_id].distance(other_genome.connections[c_id], config.connection)

            if len(matched_connections) > 0:
                c_dist = c_dist / len(matched_connections)

        dist += (c3 * c_dist)
        # dist = (dist / 3.0)
        # assert 0.0 <= dist <= 1.0
        return dist

    def mutate(self, config):

        node_add_rate = config.genome.node_add_rate
        node_delete_rate = config.genome.node_delete_rate
        conn_add_rate = config.genome.conn_add_rate
        conn_delete_rate = config.genome.conn_delete_rate

        if config.genome.single_structural_mutation:
            success = False
            mutation_probs = np.array([node_add_rate, node_delete_rate, conn_add_rate, conn_delete_rate])
            while True:
                mutation_probs = mutation_probs / np.sum(mutation_probs)
                mut_idx = np.random.choice(range(4), 1, p=mutation_probs)[0]

                if mut_idx == 0:
                    success = self.mutate_add_node(config)
                elif mut_idx == 1:
                    success = self.mutate_delete_node()
                elif mut_idx == 2:
                    success = self.mutate_add_connection(config)
                else:
                    success = self.mutate_delete_connection()

                if success is True:
                    break

                mutation_probs[mut_idx] = 0.0
                if np.sum(mutation_probs) == 0.0:
                    break
        else:
            if random.random() < node_add_rate:
                self.mutate_add_node(config)

            if random.random() < node_delete_rate:
                self.mutate_delete_node()

            if random.random() < conn_add_rate:
                self.mutate_add_connection(config)

            if random.random() < conn_delete_rate:
                self.mutate_delete_connection()

        mutate_nodes, mutate_connections = [], []
        nb_mutate_nodes = probabilistic_round(random.random() * len(self.nodes))
        nb_mutate_connections = probabilistic_round(random.random() * len(self.connections))

        if nb_mutate_nodes > 0:
            mutate_nodes = np.random.choice(list(self.nodes.keys()), nb_mutate_nodes)
        if nb_mutate_connections > 0 and len(self.connections) > 0:
            mutate_connections = np.random.choice(list(self.connections.keys()), nb_mutate_connections)

        for n_id in mutate_nodes:
            self.nodes[n_id].mutate(config.node)
        for c_id in mutate_connections:
            self.connections[c_id].mutate(config.connection)

    def mutate_add_node(self, config):
        if len(self.connections) == 0:
            return False

        connection = random.choice(list(self.connections.values()))
        n_id = self._next_node_id()
        self._create_node(n_id, 'hidden', config=config.node)
        self.node_ids['hidden'].add(n_id)
        self.node_ids['all'].add(n_id)

        self._compute_probable_connectors(n_id)

        connection.enabled = False
        source_id = connection.source_id
        target_id = connection.target_id
        self._create_connection(source_id, n_id, 1.0, True)  # [2] pg.108
        self._create_connection(n_id, target_id, connection.weight, True)  # [2] pg.108
        return True

    def mutate_delete_node(self):
        if len(self.node_ids['hidden']) < 1:
            return False

        n_id = random.choice(list(self.node_ids['hidden']))
        connections_to_delete = set()
        for c_id, connection in self.connections.items():
            if connection.source_id == n_id or connection.target_id == n_id:
                connections_to_delete.add(c_id)

        for c_id in connections_to_delete:
            self._connectors.remove((self.connections[c_id].source_id, self.connections[c_id].target_id))
            del self.connections[c_id]

        self.node_ids['all'].remove(n_id)
        self.node_ids['hidden'].remove(n_id)
        del self.nodes[n_id]

        self._cyclic_connectors = {(x, y) for x, y in self._cyclic_connectors if x != n_id and y != n_id}
        self._acyclic_connectors = {(x, y) for x, y in self._acyclic_connectors if x != n_id and y != n_id}
        return True

    def mutate_add_connection(self, config):
        if config.genome.feed_forward:
            possible_connectors = self._acyclic_connectors - self._connectors
        else:
            possible_connectors = self._cyclic_connectors - self._connectors

        if len(possible_connectors) == 0:
            return False

        while len(possible_connectors) > 0:
            connector = random.choice(list(possible_connectors))

            if config.genome.feed_forward:
                if self._is_cyclic(connector[0], connector[1]) is True:
                    possible_connectors.remove(connector)
                    continue

            self._create_connection(connector[0], connector[1], config=config.connection)
            return True
        return False

    def mutate_delete_connection(self):
        if len(self.connections) < 2:
            return False

        connection = random.choice(list(self.connections.values()))
        c_id = connection.id
        source_id = connection.source_id
        target_id = connection.target_id
        self._connectors.remove((source_id, target_id))
        del self.connections[c_id]

        if source_id == target_id:
            if self._is_redundant_node(source_id):
                self.node_ids['all'].remove(source_id)
                self.node_ids['hidden'].remove(source_id)
                del self.nodes[source_id]

                self._cyclic_connectors = {(x, y) for x, y in self._cyclic_connectors if
                                           x != source_id and y != source_id}
                self._acyclic_connectors = {(x, y) for x, y in self._acyclic_connectors if
                                            x != source_id and y != source_id}
        else:
            if self._is_redundant_node(source_id):
                self.node_ids['all'].remove(source_id)
                self.node_ids['hidden'].remove(source_id)
                del self.nodes[source_id]

                self._cyclic_connectors = {(x, y) for x, y in self._cyclic_connectors if
                                           x != source_id and y != source_id}
                self._acyclic_connectors = {(x, y) for x, y in self._acyclic_connectors if
                                            x != source_id and y != source_id}
            if self._is_redundant_node(target_id):
                self.node_ids['all'].remove(target_id)
                self.node_ids['hidden'].remove(target_id)
                del self.nodes[target_id]

                self._cyclic_connectors = {(x, y) for x, y in self._cyclic_connectors if
                                           x != target_id and y != target_id}
                self._acyclic_connectors = {(x, y) for x, y in self._acyclic_connectors if
                                            x != target_id and y != target_id}
        return True

    def _is_redundant_node(self, n_id):
        if self.nodes[n_id].type == 'input' or self.nodes[n_id].type == 'output':
            return False

        for connector in self._connectors:
            if connector[0] == n_id or connector[1] == n_id:
                return False

        return True

    def _is_cyclic(self, source_id, target_id):  # [1]
        if source_id == target_id:
            return True

        visited = {target_id}
        while True:
            num_added = 0
            for connection in self.connections.values():
                a, b = connection.source_id, connection.target_id
                if a in visited and b not in visited:
                    if b == source_id:
                        return True

                    visited.add(b)
                    num_added += 1

            if num_added == 0:
                return False

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

        matched_connections, unmatched_connections = set(), set()
        for x in c_list_1:
            if x in c_list_2:
                matched_connections.add((x, x))
            else:
                unmatched_connections.add((x, -1))

        for y in c_list_2:
            if y in c_list_1:
                matched_connections.add((y, y))
            else:
                unmatched_connections.add((-1, y))

        required_nodes = set()
        for c1_idx, c2_idx in matched_connections:
            c1 = connection_set_1[c1_idx]
            c2 = connection_set_2[c2_idx]
            new_c = c1.crossover(c2)
            assert new_c.id not in self.connections
            self.connections[new_c.id] = new_c
            self._connectors.add((new_c.source_id, new_c.target_id))
            required_nodes.add(new_c.source_id)
            required_nodes.add(new_c.target_id)

        for c1_idx, c2_idx in unmatched_connections:
            if fitness_case == 'equal':
                if random.random() < 0.5:
                    if c1_idx != -1:
                        c = connection_set_1[c1_idx]
                        if config.genome.feed_forward:
                            if self._is_cyclic(c.source_id, c.target_id) is True:
                                continue
                        self._create_connection(c.source_id, c.target_id, c.weight, c.enabled)
                        required_nodes.add(c.source_id)
                        required_nodes.add(c.target_id)
                else:
                    if c2_idx != -1:
                        c = connection_set_2[c2_idx]
                        if config.genome.feed_forward:
                            if self._is_cyclic(c.source_id, c.target_id) is True:
                                continue
                        self._create_connection(c.source_id, c.target_id, c.weight, c.enabled)
                        required_nodes.add(c.source_id)
                        required_nodes.add(c.target_id)
            else:
                if c1_idx != -1:
                    c = connection_set_1[c1_idx]
                    if config.genome.feed_forward:
                        if self._is_cyclic(c.source_id, c.target_id) is True:
                            continue
                    self._create_connection(c.source_id, c.target_id, c.weight, c.enabled)
                    required_nodes.add(c.source_id)
                    required_nodes.add(c.target_id)

        node_set_1 = p1.nodes
        node_set_2 = p2.nodes

        n_list_1 = sorted(list(node_set_1.keys()))
        n_list_2 = sorted(list(node_set_2.keys()))

        node_pairs = set()
        for x in n_list_1:
            if x in n_list_2:
                node_pairs.add((x, x))
            else:
                node_pairs.add((x, -1))

        for y in n_list_2:
            if y in n_list_1:
                node_pairs.add((y, y))
            else:
                node_pairs.add((-1, y))

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

        self._compute_probable_connectors()

        self.node_indexer = count(max(parent_1.node_indexer_cntr, parent_2.node_indexer_cntr) + 1)
        self.node_indexer_cntr = max(parent_1.node_indexer_cntr, parent_2.node_indexer_cntr)

    def crossover_asexual(self, parent_1):
        for node in parent_1.nodes.values():
            assert node.id not in self.nodes
            self._create_node(node.id, node.type, node.bias, node.response, node.activation, node.aggregation)
            self.node_ids['all'].add(node.id)
            self.node_ids[node.type].add(node.id)

        self._compute_probable_connectors()

        for connection in parent_1.connections.values():
            self._create_connection(connection.source_id, connection.target_id, connection.weight, connection.enabled)

        self.node_indexer = count(parent_1.node_indexer_cntr + 1)
        self.node_indexer_cntr = parent_1.node_indexer_cntr

    def _next_node_id(self):
        n_id = next(self.node_indexer)
        self.node_indexer_cntr = n_id
        assert n_id not in self.nodes
        return n_id

    def _compute_probable_connectors(self, n_id=None):
        if n_id is None:
            for source_id in self.node_ids['input']:
                for target_id in self.node_ids['hidden']:
                    self._cyclic_connectors.add((source_id, target_id))
                    self._acyclic_connectors.add((source_id, target_id))
                for target_id in self.node_ids['output']:
                    self._cyclic_connectors.add((source_id, target_id))
                    self._acyclic_connectors.add((source_id, target_id))

            for source_id in self.node_ids['hidden']:
                for target_id in self.node_ids['hidden']:
                    self._cyclic_connectors.add((source_id, target_id))
                    if source_id != target_id:
                        self._acyclic_connectors.add((source_id, target_id))
                for target_id in self.node_ids['output']:
                    self._cyclic_connectors.add((source_id, target_id))
                    self._acyclic_connectors.add((source_id, target_id))
        else:
            for source_id in self.node_ids['input']:
                self._cyclic_connectors.add((source_id, n_id))
                self._acyclic_connectors.add((source_id, n_id))
            for source_id in self.node_ids['hidden']:
                for target_id in self.node_ids['hidden']:
                    self._cyclic_connectors.add((source_id, target_id))
                    if source_id != target_id:
                        self._acyclic_connectors.add((source_id, target_id))
            for target_id in self.node_ids['output']:
                self._cyclic_connectors.add((n_id, target_id))
                self._acyclic_connectors.add((n_id, target_id))

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

        self._compute_probable_connectors()

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
        self._connectors.add((source_id, target_id))
