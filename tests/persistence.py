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

        assert (old_g == new_g)

    def test_config_persistence(self):
        old_config = Config()
        old_config.initialize('persistence_config_1.ini')
        config_json = old_config.to_json()
        new_config = Config()
        new_config.from_json(config_json)

        assert (old_config == new_config)


if __name__ == '__main__':
    unittest.main()
