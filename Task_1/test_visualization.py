import unittest
from unittest.mock import patch, MagicMock
import pandas as pd
from visualization import load_data, create_best_fit_visualization, create_mapped_data_visualization

class TestVisualization(unittest.TestCase):
    def setUp(self):
        # Create complete mock data with all expected columns
        self.train_df = pd.DataFrame({
            'x': [1, 2, 3],
            'y1': [1, 2, 3],
            'y2': [1.1, 2.1, 3.1],
            'y3': [1.2, 2.2, 3.2],  # Added missing column
            'y4': [1.3, 2.3, 3.3]   # Added missing column
        })
        
        self.ideal_df = pd.DataFrame({
            'x': [1, 2, 3],
            'y1': [1, 2, 3],
            'y2': [1.1, 2.1, 3.1],
            'y11': [1.0, 2.0, 3.0],  # Example ideal functions
            'y42': [1.1, 2.1, 3.1],
            'y41': [1.2, 2.2, 3.2],
            'y48': [1.3, 2.3, 3.3]
        })
        
        self.mapped_data = pd.DataFrame({
            'x': [1, 2, 3],
            'y_test': [1.05, 2.05, 3.05],
            'ideal_function': ['y42', 'y41', 'y48']  # Updated to match ideal_df columns
        })
        
        self.best_fits = {
            'y1': ('y42', 0.1),
            'y2': ('y41', 0.1),
            'y3': ('y11', 0.1),
            'y4': ('y48', 0.1)
        }

    @patch('visualization.DataHandling')
    @patch('visualization.TestDataMapper')
    def test_load_data(self, mock_mapper, mock_handler):
        # Setup mocks
        mock_handler.return_value.train_data = self.train_df
        mock_handler.return_value.ideal_data = self.ideal_df
        mock_mapper.return_value.map_test_data.return_value = self.mapped_data
        mock_mapper.return_value.find_best_fit.return_value = self.best_fits
        
        # Call function
        train, ideal, mapped, fits = load_data(":memory:", "train", "ideal", "test")
        
        # Verify
        self.assertEqual(len(train), 3)
        self.assertEqual(len(ideal), 3)
        self.assertEqual(len(mapped), 3)
        self.assertEqual(len(fits), 4)  # Updated to match 4 best fits

    @patch('visualization.output_file')
    @patch('visualization.figure')
    @patch('visualization.show')
    def test_create_best_fit_visualization(self, mock_show, mock_figure, mock_output):
        create_best_fit_visualization(self.train_df, self.ideal_df, self.best_fits)
        self.assertTrue(mock_output.called)
        self.assertTrue(mock_figure.called)

    @patch('visualization.output_file')
    @patch('visualization.figure')
    @patch('visualization.show')
    def test_create_mapped_visualization(self, mock_show, mock_figure, mock_output):
        create_mapped_data_visualization(self.train_df, self.ideal_df, self.mapped_data)
        self.assertTrue(mock_output.called)
        self.assertTrue(mock_figure.called)

if __name__ == '__main__':
    unittest.main()
