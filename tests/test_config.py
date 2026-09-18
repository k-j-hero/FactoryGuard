import unittest

from factoryguard.config import load_config


class ConfigTests(unittest.TestCase):
    def test_topdown_profile_uses_center_anchor(self):
        config = load_config("configs/topdown.yaml")
        self.assertEqual(config["proximity"]["anchor_mode"], "center")
        self.assertTrue(config["require_forklift"])


if __name__ == "__main__":
    unittest.main()
