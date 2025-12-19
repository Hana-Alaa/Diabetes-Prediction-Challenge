import unittest
import sys
from pathlib import Path
import pandas as pd
import logging

# Ensure src is in the path
sys.path.append(str(Path(__file__).parent.parent / "src"))

from data_loader import DataLoader

class TestDataLoader(unittest.TestCase):
    def setUp(self):
        # Assuming run from root directory or tests directory
        self.root_dir = Path(__file__).parent.parent
        self.data_dir = self.root_dir / "Data"
        self.loader = DataLoader(data_dir=str(self.data_dir))
        
        # Suppress logging during tests
        logging.getLogger('data_loader').setLevel(logging.CRITICAL)

    def test_init(self):
        self.assertEqual(self.loader.data_dir, self.data_dir)

    def test_files_exist(self):
        """Test that the data loader can find the files (requires files to be present)."""
        if not self.data_dir.exists():
            self.skipTest("Data directory not found, skipping integration test.")
        
        if not (self.data_dir / "train.csv").exists():
            self.skipTest("train.csv not found, skipping integration test.")

        try:
            self.loader._check_file(self.loader.train_path)
            self.loader._check_file(self.loader.test_path)
        except FileNotFoundError:
            self.fail("DataLoader raised FileNotFoundError unexpectedly for existing files.")

    def test_load_all_shape(self):
        """Test loading returns correct shapes."""
        if not (self.data_dir / "train.csv").exists():
            self.skipTest("Data files not present.")

        train, test, sub = self.loader.load_all()
        self.assertIsInstance(train, pd.DataFrame)
        self.assertIsInstance(test, pd.DataFrame)
        self.assertIsInstance(sub, pd.DataFrame)
        
        # Basic logical checks
        self.assertGreater(len(train), 0)
        self.assertGreater(len(test), 0)
        self.assertEqual(len(test), len(sub))

if __name__ == "__main__":
    unittest.main()
