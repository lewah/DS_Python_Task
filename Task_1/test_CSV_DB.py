import unittest
import sqlite3
import pandas as pd
from unittest.mock import patch
from CSV_DB import DatabaseManager, DatasetManager
from sqlalchemy import create_engine, text
import os

class TestDatabaseManager(unittest.TestCase):
    def setUp(self):
        self.test_db = "test_db.db"
        self.csv_file = "test_data.csv"
        
        # Create a clean test CSV with exactly 3 rows
        pd.DataFrame({
            'x': [1, 2, 3],
            'y': [4, 5, 6]
        }).to_csv(self.csv_file, index=False)

        # Ensure the database is clean before each test
        if os.path.exists(f"{self.test_db}.db"):
            os.remove(f"{self.test_db}.db")

    def tearDown(self):
        # Clean up files
        if os.path.exists(f"{self.test_db}.db"):
            os.remove(f"{self.test_db}.db")
        if os.path.exists(self.csv_file):
            os.remove(self.csv_file)

    def test_db_creation(self):
        dbm = DatabaseManager(self.test_db)
        self.assertTrue(os.path.exists(f"{self.test_db}.db"))
        dbm.engine.dispose()  # Close the connection

    def test_csv_to_table(self):
        dbm = DatabaseManager(self.test_db)
        # Use if_exists='replace' to ensure clean table
        dbm.csv_to_table(self.csv_file, "test_table")
        
        # Verify table was created with correct row count
        with sqlite3.connect(f"{self.test_db}.db") as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT COUNT(*) FROM test_table;")
            count = cursor.fetchone()[0]
            self.assertEqual(count, 3)  # Verify exactly 3 rows
        
        dbm.engine.dispose()  # Close the connection

    def test_fetch_data(self):
        dbm = DatabaseManager(self.test_db)
        # First clear any existing data
        with dbm.engine.connect() as conn:
            conn.execute(text("DROP TABLE IF EXISTS test_table;"))
            conn.commit()
        
        # Now load fresh data
        dbm.csv_to_table(self.csv_file, "test_table")
        
        # Test fetch
        data = dbm.fetch_data("test_table")
        self.assertEqual(len(data), 3)  # Should have exactly 3 rows
        dbm.engine.dispose()  # Close the connection

class TestDatasetManager(unittest.TestCase):
    def setUp(self):
        self.test_db = "test_db_dataset.db"
        if os.path.exists(f"{self.test_db}.db"):
            os.remove(f"{self.test_db}.db")

    def tearDown(self):
        if os.path.exists(f"{self.test_db}.db"):
            os.remove(f"{self.test_db}.db")

    def test_dataset_manager_inheritance(self):
        manager = DatasetManager(self.test_db, "test.csv", "test_table")
        self.assertIsInstance(manager, DatabaseManager)
        manager.engine.dispose()  # Close the connection