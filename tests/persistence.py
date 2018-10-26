import unittest

from evolving_networks.configurations.config import Config
from evolving_networks.genome.genome import Genome


class TestPersistence(unittest.TestCase):
    def test_genome_persistence(self):
        config = Config()
        config.initialize('persistence_config_1.ini')
        old_g = Genome(g_id=None, generation=None, config=config.genome)
        old_g.initialize(config.node, config.connection)
        old_g_json = old_g.to_json()
        new_g = Genome(g_id=None, generation=None, config=config.genome)
        new_g.from_json(old_g_json)

        assert (old_g.id == new_g.id)
        assert (old_g.fitness == new_g.fitness)
        assert (old_g.adjusted_fitness == new_g.adjusted_fitness)
        assert (old_g.birth_generation == new_g.birth_generation)
        assert (old_g.is_damaged == new_g.is_damaged)
        assert (old_g.nodes == new_g.nodes)
        assert (old_g.connections == new_g.connections)
        assert (old_g.innovation_archive == new_g.innovation_archive)
        assert (old_g.node_ids == new_g.node_ids)
        assert (old_g.node_indexer_cntr == new_g.node_indexer_cntr)
        assert (old_g._innovation_indexer_cntr == new_g._innovation_indexer_cntr)
        assert (old_g._connectors == new_g._connectors)
        assert (old_g._cyclic_connectors == new_g._cyclic_connectors)
        assert (old_g._acyclic_connectors == new_g._acyclic_connectors)

        assert (old_g.node_indexer != new_g.node_indexer)
        assert (old_g._innovation_indexer != new_g._innovation_indexer)

    def test_config_persistence(self):
        old_config = Config()
        old_config.initialize('persistence_config_1.ini')
        config_json = old_config.to_json()
        new_config = Config()
        new_config.from_json(config_json)

        assert (old_config == new_config)


if __name__ == '__main__':
    unittest.main()
