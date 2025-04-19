import sqlite3
import pandas as pd
import numpy as np
from sqlalchemy import create_engine,text
from sqlalchemy.exc import OperationalError
from CSV_DB import DatabaseManager,DatasetManager

class DataHandling:
    """
    class for handling database operations and retrieving data from an SQLite database by connecting to SQLite database, 
    fetches data from specified tables, and provides methods for extracting relevant columns.

    Attributes:
        db_path (str): Path to the SQLite database file.
        engine (sqlalchemy.Engine): SQLAlchemy engine for database connection.
        train_data (pd.DataFrame): DataFrame containing training data.
        ideal_data (pd.DataFrame): DataFrame containing ideal functions data.
        test_data (pd.DataFrame): DataFrame containing test data.

    Methods:
        get_columns(data: pd.DataFrame, prefix: str) -> list:
            Returns a list of column names from the given DataFrame that start with the specified prefix.
    """
    def __init__(self, db_path: str, train_table: str, ideal_table: str, test_table: str):
        """
        Initialize and fetch data from the database.
        Args:
            db_path (str): Path to the SQLite database file.
            train_table (str): Name of the table containing training data.
            ideal_table (str): Name of the table containing ideal functions data.
            test_table (str): Name of the table containing test data.

        Raises:
            sqlalchemy.exc.OperationalError: If there is an issue connecting to the database.
            Exception: For any unexpected errors during data fetching.
        """
        self.db_path = db_path
        self.engine = create_engine(f"sqlite:///{self.db_path}")  # Use absolute path

        
        # Debug: Print table names
        try:
            # Fetch data from the database
            self.train_data = pd.read_sql(f"SELECT * FROM {train_table}", self.engine)
            self.ideal_data = pd.read_sql(f"SELECT * FROM {ideal_table}", self.engine)
            self.test_data = pd.read_sql(f"SELECT * FROM {test_table}", self.engine)

            print("Data successfully loaded from database.")
        except OperationalError as e:
            print(f"Database error: {e}")
        except Exception as e:
            print(f"An unexpected error occurred: {e}")

        
    def get_columns(self, data, prefix):
        # what is returned here is ['y1', 'y2', 'y3', 'y4']
        return [col for col in data.columns if col.startswith(prefix)]


class FunctionSelection(DataHandling):
    """
    A class to select the best-fit ideal functions using the least squares method.

    Attributes:
        db_path (str): The path to the SQLite database.
        train_table (str): The name of the training data table.
        ideal_table (str): The name of the ideal functions table.
        test_table (str): The name of the test data table.
    """
    def __init__(self, db_path, train_table, ideal_table, test_table):
        super().__init__(db_path, train_table, ideal_table, test_table)
        self.x_train = self.train_data['x'].values
        self.y_train_cols = self.get_columns(self.train_data, 'y')
        self.y_ideal_cols = self.get_columns(self.ideal_data, 'y')

    
    def find_best_fit(self):
        """
        Identifies the four best-fitting ideal functions for the training dataset 
        by minimizing the Sum of Squared Errors (SSE).

        The method iterates over each training column (`y_train_cols`), compares it 
        with all ideal function columns (`y_ideal_cols`), and selects the ideal function 
        with the lowest Mean Squared Error (MSE). The top four best-matching functions 
        are chosen based on the lowest MSE.

        Returns:
            dict: A dictionary mapping each selected training column to its best-fit 
                ideal function and corresponding MSE value. 
                Format: {train_col: (best_ideal_col, MSE)}
                If an error occurs, returns None.

        Raises:
            Exception: Catches and prints any unexpected errors.
            
        Notes:
            - The method checks for shape mismatches and skips invalid comparisons.
            - The final selection is based on the lowest MSE values.
            - The method prints the best match for each selected training function.
        """
        try:
            best_fits = {}

            for train_col in self.y_train_cols:
                y_train = self.train_data[train_col].values
                min_sse = float('inf')
                best_match = None

                for ideal_col in self.y_ideal_cols:
                    y_ideal = self.ideal_data[ideal_col].values

                    if len(y_train) != len(y_ideal):
                        print(f"Shape mismatch: {train_col} ({len(y_train)}) vs {ideal_col} ({len(y_ideal)})")
                        continue  # Skip if shapes don't match

                    # Compute Sum of Squared Errors (SSE) and Mean Squared Error (MSE)
                    sse = np.sum((y_train - y_ideal) ** 2)
                    mse = sse / len(y_train)

                    if sse < min_sse:
                        min_sse = sse
                        best_match = ideal_col

                best_fits[train_col] = (best_match, round(min_sse / len(y_train), 4))

            # Sort by lowest MSE and pick the best 4
            best_fits_sorted = sorted(best_fits.items(), key=lambda x: x[1][1])[:4]

            # Print results
            for train_col, (best_ideal, mse) in best_fits_sorted:
                print(f"Best fit for {train_col}: {best_ideal} with MSE={mse}")

            return dict(best_fits_sorted)

        except Exception as e:
            print(f"An error occurred while finding the best fit: {e}")
            return None
    
class TestDataMapper(FunctionSelection):
    """
    Maps test data to the most suitable ideal function using the maximum deviation criterion.

    This class extends `FunctionSelection` and performs the following tasks:
    1. Computes the maximum deviation for each best-fit ideal function.
    2. Maps test data points to an ideal function if the deviation does not exceed 
       `sqrt(2) * max_deviation`.
    3. Saves the mapped results into an SQLite database.

    Attributes:
        db_path (str): Path to the SQLite database.
        train_table (str): Name of the training data table.
        ideal_table (str): Name of the ideal functions data table.
        test_table (str): Name of the test data table.
        max_deviation (dict): Stores the maximum deviation for each selected ideal function.
        selected_functions (dict): Maps training data columns to their best-fit ideal functions.

    Methods:
        compute_max_deviation():
            Computes the maximum deviation for each selected ideal function.
        map_test_data():
            Maps test data to the best-fit ideal function based on the deviation threshold.
        save_to_database(df, table_name):
            Saves the mapped results into an SQLite database.
    """
    def __init__(self, db_path, train_table, ideal_table, test_table):
        """
        Initializes the TestDataMapper class and sets up required attributes.
        Args:
            db_path (str): Path to the SQLite database.
            train_table (str): Name of the training data table.
            ideal_table (str): Name of the ideal functions data table.
            test_table (str): Name of the test data table.
        """
        super().__init__(db_path, train_table, ideal_table, test_table)
        self.max_deviation = {}
        self.selected_functions = {}
    
    def compute_max_deviation(self):
        """
        Computes the maximum deviation between training data and their corresponding 
        best-fit ideal functions.
        Uses the best-fit functions identified from `find_best_fit()` and calculates 
        the maximum absolute difference between training and ideal function values.
        Stores the computed maximum deviation for each best-fit function.
        """
        
        # Get the best fit functions using the corrected method
        best_fits = self.find_best_fit()  # This returns a dictionary {train_col: (ideal_col, mse)}
        
        for train_col, (best_func, _) in best_fits.items():
            self.selected_functions[train_col] = best_func
            
            # Compute max deviation between training and ideal function
            deviations = np.abs(self.train_data[train_col].values - self.ideal_data[best_func].values)
            self.max_deviation[best_func] = np.max(deviations)
            
            # Debug: Print deviations and max deviation for each ideal function
            print(f"Ideal Function: {best_func}, Max Deviation: {self.max_deviation[best_func]}")

    # def map_test_data(self, best_fits, max_deviations):
    def map_test_data(self):
        """
        Maps test data points to the most suitable ideal function.
        Each test data point (x, y) is compared against all selected ideal functions. 
        The function with the minimum deviation is assigned to the test point, provided 
        the deviation does not exceed `sqrt(2) * max_deviation` for that function.
        Returns:
            pd.DataFrame: A DataFrame containing mapped test data with columns:
                - 'x': X value from the test data.
                - 'y_test': Y value from the test data.
                - 'deviation(delta_y)': Absolute deviation from the assigned ideal function.
                - 'ideal_function': The best-matched ideal function.
        """
        results = []
        sqrt_2 = np.sqrt(2)
        
        for _, row in self.test_data.iterrows():
            x_test, y_test = row['x'], row['y']
            best_match = None
            min_deviation = float('inf')
            
            for ideal_func in set(self.selected_functions.values()):
                y_ideal = self.ideal_data.loc[self.ideal_data['x'] == x_test, ideal_func].values
                # print(f"x_test: {x_test}, ideal_func: {ideal_func}, y_ideal: {y_ideal}")
                # print(f"y_ideal: {y_ideal}")  
                
                if len(y_ideal) == 0:
                    continue  # Skip if x_test is not in ideal function data
                
                deviation = abs(y_test - y_ideal[0])
                threshold = sqrt_2 * self.max_deviation[ideal_func]
                
                # if deviation <= np.sqrt(2) * self.max_deviation[ideal_col]:
                if deviation <= threshold and deviation < min_deviation:
                    # if deviation < min_deviation:
                    min_deviation = deviation
                    best_match = ideal_func
            
            results.append({'x': x_test, 'y_test': y_test, 'deviation(delta_y)': min_deviation, 'ideal_function': best_match})
            
        return pd.DataFrame(results)
    
    def save_to_database(self, df, table_name):
        """
        Saves the mapped test data into the specified SQLite database table.
        Args:
            df (pd.DataFrame): The DataFrame containing the mapped test data.
            table_name (str): Name of the table where data will be stored.
        """
        df.to_sql(table_name, self.engine, if_exists="replace", index=False)
        print(f"Data saved to table: {table_name}")




