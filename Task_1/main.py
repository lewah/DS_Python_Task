import os
import cProfile # Profiling
import sqlite3
import sqlalchemy
from pathlib import Path
from sqlalchemy import create_engine, text,select
from sqlalchemy.exc import OperationalError, SQLAlchemyError
from CSV_DB import DatabaseManager,DatasetManager
from mapping import DataHandling,FunctionSelection,TestDataMapper
from visualization import load_data,create_best_fit_visualization,create_mapped_data_visualization

# for unittest
BASE_DIR = Path('/Users/ingrida.l/Desktop/MS-DS/Q1/Programing_with_python/Assignment/Task_1')
# # BASE_DIR = Path(__file__).parent  # This makes it relative to the script location for Unittest
DATA_DIR = BASE_DIR / 'dataset_1'

def main():
    # print('Hello World')
    '''Python program needs to be able to;Independently compile a SQLite database (file) ideally via sqlalchemy and load the training data 
    into a single five column spreadsheet / table in the file.  a. first column depicts the x-values of all functions'''
    #the path commented for testing purposes
    # BASE_DIR = Path('/Users/ingrida.l/Desktop/MS-DS/Q1/Programing_with_python/Assignment/Task_1')
    # DATA_DIR = BASE_DIR / 'dataset_1'

    db_name = 'Python_Assignment_v2' 
    db_path = BASE_DIR / f'{db_name}.db'  # Using Path for database path

    # print("Database path:", os.path.abspath(db_name))
    
    # Defining paths and table names
    #1. change paths and used pathlib
    datasets = [
        {'csv_file': DATA_DIR / "train.csv", 'table_name': 'Training_data'},
        {'csv_file': DATA_DIR / "ideal.csv", 'table_name': 'Ideal_functions_data'},
        {'csv_file': DATA_DIR / "test.csv", 'table_name': 'Test_data'}
    ]
    # Check if files exist
    for dataset in datasets:
        if not dataset['csv_file'].exists():
            raise FileNotFoundError(f"File {dataset['csv_file']} not found!")

    # Load all datasets this is the original
    for dataset in datasets:
        manager = DatasetManager(db_name, str(dataset['csv_file']), dataset['table_name']) # new
        manager.load_data()
        print(f"\nSample data from {dataset['table_name']} table:")
        data = manager.fetch_data(dataset['table_name'])
        if data is not None:
            print(data.head())  # Prints a few rows for readability
            
    # for unittest Load all datasets
    # for dataset in datasets:
    #     if not dataset['csv_file'].exists():
    #         raise FileNotFoundError(f"File {dataset['csv_file']} not found!")

    #     manager = DatasetManager(db_name, str(dataset['csv_file']), dataset['table_name']) 
    #     manager.load_data()
    
    # Verify tables in the database
    with sqlite3.connect(db_name) as conn:
        # conn = sqlite3.connect(f'{db_name}.db')
        cursor = conn.cursor()
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
        tables = cursor.fetchall()
        # Debug to check if the connection works returns Tables in database: [('Test_data',), ('Training_data',), ('Ideal_functions_data',)]
        print("Tables in database:", tables)
    # conn.commit()
    # conn.close()
    
    #Working with the databse
    train_table = 'Training_data'
    ideal_table = 'Ideal_functions_data'
    test_table = 'Test_data'
    output_table = "Mapped_Test_Data" 
    
    # Initialize DataHandling
    print("Initializing DataHandling...")
    # data_handler = DataHandling(db_name, train_table, ideal_table, test_table)
    data_handler = DataHandling(db_path, train_table, ideal_table, test_table)

    # # Debuging and inspect the fetched data
    # print("\nSample Training Data:")
    # print(data_handler.train_data.head())

    # print("\nSample Ideal Functions Data:")
    # print(data_handler.ideal_data.head())

    # print("\nSample Test Data:")
    # print(data_handler.test_data.head())

    # Use get_columns method to check if 
    print("\nColumns in Training Data starting with 'y':")
    y_columns = data_handler.get_columns(data_handler.train_data, 'y')
    print(y_columns) #Debug
   

    # Instantiate the TestDataMapper class
    test_mapper = TestDataMapper(db_path, train_table, ideal_table, test_table)
    # Compute the max deviations
    test_mapper.compute_max_deviation()
    # Map test data to ideal functions
    mapped_data = test_mapper.map_test_data()
    # Save results to the database
    test_mapper.save_to_database(mapped_data, output_table)
    print("Test data mapping completed and saved to database.")
    # print(test_mapper.max_deviation) #Debug
    
    # for visualisation ,Load data
    training_df, ideal_df, mapped_data,best_fits = load_data(db_path, train_table, ideal_table, test_table)
    
    # Generate visualization
    create_mapped_data_visualization(training_df, ideal_df, mapped_data)
    create_best_fit_visualization(training_df, ideal_df, best_fits)

if __name__ == '__main__':
    # cProfile.run('main()') #####Profile to identify which parts of your code are slow or consuming excessive resources
    main()