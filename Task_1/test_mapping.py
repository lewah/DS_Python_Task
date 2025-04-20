import unittest
import numpy as np
import pandas as pd
from unittest.mock import MagicMock, patch
from mapping import DataHandling, FunctionSelection, TestDataMapper
import sqlite3
from sqlalchemy import create_engine, text

class TestDataHandling(unittest.TestCase):
    def setUp(self):
        # Create in-memory database for testing
        self.engine = create_engine("sqlite:///:memory:")
        
        # Create test tables
        pd.DataFrame({
            'x': [1, 2, 3],
            'y1': [1.1, 2.1, 3.1],
            'y2': [1.2, 2.2, 3.2]
        }).to_sql("train_data", self.engine, index=False)
        
        pd.DataFrame({
            'x': [1, 2, 3],
            'y1': [1.0, 2.0, 3.0],
            'y2': [1.1, 2.1, 3.1]
        }).to_sql("ideal_data", self.engine, index=False)
        
        pd.DataFrame({
            'x': [1, 2, 3],
            'y': [1.05, 2.05, 3.05]
        }).to_sql("test_data", self.engine, index=False)

    def test_data_loading(self):
        # Mock the engine creation in DataHandling
        with patch('mapping.create_engine', return_value=self.engine):
            handler = DataHandling(":memory:", "train_data", "ideal_data", "test_data")
            self.assertEqual(len(handler.train_data), 3)
            self.assertEqual(len(handler.ideal_data), 3)
            self.assertEqual(len(handler.test_data), 3)

    def test_get_columns(self):
        with patch('mapping.create_engine', return_value=self.engine):
            handler = DataHandling(":memory:", "train_data", "ideal_data", "test_data")
            cols = handler.get_columns(handler.train_data, 'y')
            self.assertEqual(cols, ['y1', 'y2'])

class TestFunctionSelection(unittest.TestCase):
    def setUp(self):
        self.engine = create_engine("sqlite:///:memory:")
        
        # Create test tables
        pd.DataFrame({
            'x': [1, 2, 3],
            'y1': [1.1, 2.1, 3.1],
            'y2': [1.2, 2.2, 3.2]
        }).to_sql("train_data", self.engine, index=False)
        
        pd.DataFrame({
            'x': [1, 2, 3],
            'y1': [1.0, 2.0, 3.0],
            'y2': [1.1, 2.1, 3.1]
        }).to_sql("ideal_data", self.engine, index=False)
        
        pd.DataFrame({
            'x': [1, 2, 3],
            'y': [1.05, 2.05, 3.05]
        }).to_sql("test_data", self.engine, index=False)

    def test_find_best_fit(self):
        with patch('mapping.create_engine', return_value=self.engine):
            selector = FunctionSelection(":memory:", "train_data", "ideal_data", "test_data")
            best_fits = selector.find_best_fit()
            self.assertIsInstance(best_fits, dict)
            self.assertEqual(len(best_fits), 2)

class TestTestDataMapper(unittest.TestCase):
    def setUp(self):
        self.engine = create_engine("sqlite:///:memory:")
        
        # Create test tables
        pd.DataFrame({
            'x': [1, 2, 3],
            'y1': [1.1, 2.1, 3.1],
            'y2': [1.2, 2.2, 3.2]
        }).to_sql("train_data", self.engine, index=False)
        
        pd.DataFrame({
            'x': [1, 2, 3],
            'y1': [1.0, 2.0, 3.0],
            'y2': [1.1, 2.1, 3.1]
        }).to_sql("ideal_data", self.engine, index=False)
        
        pd.DataFrame({
            'x': [1, 2, 3],
            'y': [1.05, 2.05, 3.05]
        }).to_sql("test_data", self.engine, index=False)

    def test_compute_max_deviation(self):
        with patch('mapping.create_engine', return_value=self.engine):
            mapper = TestDataMapper(":memory:", "train_data", "ideal_data", "test_data")
            mapper.compute_max_deviation()
            self.assertGreater(len(mapper.max_deviation), 0)

    def test_map_test_data(self):
        with patch('mapping.create_engine', return_value=self.engine):
            mapper = TestDataMapper(":memory:", "train_data", "ideal_data", "test_data")
            mapper.compute_max_deviation()
            mapped = mapper.map_test_data()
            self.assertEqual(len(mapped), 3)

if __name__ == '__main__':
    unittest.main()