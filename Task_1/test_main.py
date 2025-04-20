import unittest
from unittest.mock import patch, MagicMock, PropertyMock
import os
from pathlib import Path
import pandas as pd
from main import main

class TestMainScript(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Create a temporary test directory structure
        cls.test_dir = Path(__file__).parent / "test_data"
        cls.test_dir.mkdir(exist_ok=True)
        (cls.test_dir / "dataset_1").mkdir(exist_ok=True)
        
        # Create dummy CSV files
        csv_data = {
            "train.csv": "x,y1,y2,y3,y4\n0,0,0,0,0\n1,1,1,1,1",
            "ideal.csv": "x,y1,y2,y3,y4\n0,0,0,0,0\n1,1,1,1,1",
            "test.csv": "x,y\n0,0\n1,1"
        }
        
        for filename, content in csv_data.items():
            with open(cls.test_dir / "dataset_1" / filename, 'w') as f:
                f.write(content)

    def setUp(self):
        # Create mock data that will be returned by our mocks
        self.mock_train_data = pd.DataFrame({
            'x': [1, 2, 3],
            'y1': [1.1, 2.1, 3.1],
            'y2': [1.2, 2.2, 3.2]
        })
        self.mock_ideal_data = pd.DataFrame({
            'x': [1, 2, 3],
            'y1': [1.0, 2.0, 3.0],
            'y2': [1.1, 2.1, 3.1]
        })
        self.mock_test_data = pd.DataFrame({
            'x': [1, 2, 3],
            'y': [1.05, 2.05, 3.05]
        })
        self.mock_mapped_data = pd.DataFrame({
            'x': [1, 2, 3],
            'y_test': [1.05, 2.05, 3.05],
            'ideal_function': ['y1', 'y1', 'y2']
        })
        self.mock_best_fits = {
            'y1': ('y1', 0.1),
            'y2': ('y2', 0.1)
        }

    @patch('main.create_best_fit_visualization')
    @patch('main.create_mapped_data_visualization')
    @patch('main.load_data')
    @patch('main.TestDataMapper')
    @patch('main.DataHandling')
    @patch('main.DatasetManager')
    def test_main_execution(self, mock_dataset_manager, mock_data_handler, 
                          mock_test_mapper, mock_load_data, 
                          mock_create_mapped, mock_create_best):
        # Configure the mocks
        mock_dataset_manager.return_value.load_data.return_value = None
        mock_dataset_manager.return_value.fetch_data.return_value = self.mock_train_data
        
        # Setup DataHandling mock
        data_handler_instance = mock_data_handler.return_value
        data_handler_instance.train_data = self.mock_train_data
        data_handler_instance.ideal_data = self.mock_ideal_data
        data_handler_instance.test_data = self.mock_test_data
        data_handler_instance.get_columns.return_value = ['y1', 'y2', 'y3', 'y4']
        
        # Setup TestDataMapper mock
        test_mapper_instance = mock_test_mapper.return_value
        test_mapper_instance.map_test_data.return_value = self.mock_mapped_data
        test_mapper_instance.find_best_fit.return_value = self.mock_best_fits
        
        # Setup load_data mock
        mock_load_data.return_value = (
            self.mock_train_data,
            self.mock_ideal_data,
            self.mock_mapped_data,
            self.mock_best_fits
        )
        
        # Run main with patched BASE_DIR and DATA_DIR
        with patch('main.BASE_DIR', self.test_dir), \
             patch('main.DATA_DIR', self.test_dir / "dataset_1"):
            main()
        
        # Verify the expected calls were made
        self.assertTrue(mock_dataset_manager.called)
        self.assertTrue(mock_data_handler.called)
        self.assertTrue(mock_test_mapper.called)

    @patch('main.DatasetManager')
    def test_file_not_found(self, mock_dataset_manager):
        # Configure mock to raise FileNotFoundError
        mock_dataset_manager.return_value.load_data.side_effect = FileNotFoundError
        
        # Run test with non-existent directory
        with patch('main.BASE_DIR', Path("/nonexistent")), \
             patch('main.DATA_DIR', Path("/nonexistent/dataset_1")):
            with self.assertRaises(FileNotFoundError):
                main()

    @classmethod
    def tearDownClass(cls):
        # Clean up test files
        for file in cls.test_dir.glob("**/*"):
            if file.is_file():
                file.unlink()
        for dir in cls.test_dir.glob("*/"):
            dir.rmdir()
        cls.test_dir.rmdir()

if __name__ == '__main__':
    unittest.main()
