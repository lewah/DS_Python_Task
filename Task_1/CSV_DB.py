import os
import sqlite3
import pandas as pd
import sqlalchemy
from sqlalchemy import create_engine, text
from sqlalchemy.exc import OperationalError, SQLAlchemyError

# Base class for database operations
# DBM - DatabaseManager

class DatabaseManager: #correct name for class
    """
    Manages SQLite database operations including creating tables from CSV files 
    and fetching data from tables.

    This class provides methods to:
    - Load CSV data into SQLite tables.
    - Process test data line-by-line for memory efficiency.
    - Retrieve data from tables with an optional row limit.

    Attributes:
        db_name (str): Name of the SQLite database.
        engine (sqlalchemy.Engine): SQLAlchemy engine for database connection.

    Methods:
        csv_to_table(csv_file: str, table_name: str):
            Loads data from a CSV file into the specified SQLite table.
            Handles test data separately by loading it line-by-line.

        fetch_data(table_name: str, limit: int = 5) -> pd.DataFrame | None:
            Retrieves a limited number of rows from a specified table.
    """
    def __init__(self, db_name: str):
        """
        Initializes the DatabaseManager class and establishes a database connection.
        Args:
            db_name (str): The name of the SQLite database (without the `.db` extension).
        """
        self.db_name = db_name
        self.engine = create_engine(f'sqlite:///{self.db_name}.db')
        
        # Force creation of database file by establishing a connection
        with self.engine.connect() as conn:
            conn.execute(text("SELECT 1"))  # Simple test query
            
    def __enter__(self):
        return self
        
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.engine.dispose()


    def csv_to_table(self, csv_file: str, table_name: str):
        """
        Loads data from a CSV file into an SQLite table.

        - If the CSV file contains "test" in its name, the data is inserted line-by-line.
        - Otherwise, the entire CSV file is loaded at once.
        Args:
            csv_file (str): Path to the CSV file.
            table_name (str): Name of the database table where data will be stored.
        Raises:
            FileNotFoundError: If the CSV file does not exist.
            sqlalchemy.exc.OperationalError: If there is an issue with the database connection.
            Exception: For any unexpected errors.
        """
        try:
            print(f"Attempting to load data from {csv_file} into table {table_name}")
            
            # Clear existing table if it exists
            with self.engine.begin() as conn:
                conn.execute(text(f"DROP TABLE IF EXISTS {table_name}"))
            
            if "test" in csv_file.lower():  # Check if 'test' is in the file name
                # Process Test Data line-by-line
                for chunk in pd.read_csv(csv_file, chunksize=1):
                    chunk.to_sql(table_name, self.engine, if_exists="append", index=False)
                print(f"Test Data: '{table_name}' loaded line-by-line.")
            else:
                # Load Training and Ideal Functions at once
                df = pd.read_csv(csv_file)
                with self.engine.begin() as conn:
                    df.to_sql(table_name, conn, if_exists='replace', index=False)
                print(f"Table '{table_name}' created and data loaded successfully.")
            return True

        except FileNotFoundError:
            print(f"Error: The file '{csv_file}' was not found.")
        except OperationalError as e:
            print(f"Database error: {e}")
        except Exception as e:
            print(f"An unexpected error occurred: {e}")
            
    # def fetch_data(self, table_name: str, limit: int = 5): # original
    def fetch_data(self, table_name: str, limit: int = None) -> pd.DataFrame | None: #new
        """
        Fetches a limited number of rows from a specified table in the SQLite database.
        Args:
            table_name (str): The name of the table to retrieve data from.
            limit (int, optional): The maximum number of rows to fetch (default is 5).
        Returns:
            pd.DataFrame | None: A DataFrame containing the retrieved data, 
            or None if the table is empty or an error occurs.
        Raises:
            sqlalchemy.exc.OperationalError: If an issue occurs while fetching data.
        """
        try:
            with self.engine.connect() as conn:
                # result = conn.execute(text(f"SELECT * FROM {table_name} LIMIT {limit}")) #original 
                result = conn.execute(text(f"SELECT * FROM {table_name};")) #new
                data = pd.DataFrame(result.fetchall(), columns=result.keys())
                return data if not data.empty else None
        except SQLAlchemyError as e: #new
            print(f"Database error: {e}")
            return None
        except OperationalError as e:
                print(f"Error fetching data from table '{table_name}': {e}")
                return None

    def close(self):
        """Explicitly close database connections"""
        self.engine.dispose()
# Handling data
class DatasetManager(DatabaseManager): #correct name for class
    """
    Manages loading datasets from CSV files into a SQLite database.
    This class extends DatabaseManager and provides functionality to 
    load a specific CSV dataset into a specified database table.

    Attributes:
        db_name (str): The name of the SQLite database.
        csv_file (str): The path to the CSV file.
        table_name (str): The name of the table to store the data.
    Methods:
        load_data():
            Loads data from the specified CSV file into the given SQLite database table.
    """
    def __init__(self, db_name: str, csv_file: str, table_name: str):
        """
        Initializes the DatasetManager class.
        Args:
            db_name (str): The name of the SQLite database.
            csv_file (str): The path to the CSV file to be loaded.
            table_name (str): The name of the table where the data will be stored.
        """
        super().__init__(db_name)
        self.csv_file = csv_file
        self.table_name = table_name

    def load_data(self):
        """
        Loads data from the CSV file into the specified SQLite database table.
        Calls the `csv_to_table` method from the `DatabaseManager` class to handle data loading.
        Raises:
            FileNotFoundError: If the CSV file is not found.
            sqlalchemy.exc.OperationalError: If a database-related error occurs.
            Exception: For any other unexpected errors.
        """
        try:
         self.csv_to_table(self.csv_file, self.table_name)
        except FileNotFoundError:
         print(f"Error: The file '{self.csv_file}' was not found.")
        except OperationalError as e:
            print(f"Database error: {e}")
        except Exception as e:
            print(f"An unexpected error occurred: {e}")
        
        # self.csv_to_table(self.csv_file, self.table_name)

