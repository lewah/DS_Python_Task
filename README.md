# DS_Python_Task

#Data Mapping and Visualization Project 

This project provides an end-to-end solution for:

1. Loading CSV data into an SQLite database
2. Mapping test data to ideal functions using statistical methods
3. Generating interactive visualizations of the relationships

## Key Features

Database Management: Efficient CSV-to-SQLite loading with line-by-line processing for large files
Data Mapping:
    Best-fit ideal function selection using Least Squares Method
    Test data mapping with deviation-based criteria
Visualization: Interactive Bokeh plots for data exploration
Testing: Comprehensive unit tests for all components
Error Handling: Robust exception handling for file operations and database connections
Project Structure

| File | Purpose |
|-----------|---------|
| main.py | Main script that orchestrates the entire workflow |
| CSV_DB.py | Database operations (CSV loading, table management) | 
| mapping.py | Core data mapping logic |
| visualization.py | Interactive Bokeh visualizations |
| test_main.py | Unit tests for main workflow |
| test_CSV_DB.py | Unit tests for database operations |
| test_mapping.py | Unit tests for data mapping |
| test_visualization.py | Unit tests for visualizations |

## Installation

Clone the repository
Install dependencies: pip install -r requirements.txt

1. Prepare Data
Place your CSV files (train.csv, ideal.csv, test.csv) in a dataset_1 directory.

2. Run the Main Script
python main.py
This will:
Create/update the SQLite database
Map test data to ideal functions
Generate two interactive visualizations:
    best_fit_visualization.html
    visualization.html

3. View Results
Open the generated HTML files in a web browser to explore the interactive visualizations.

Modify these variables in main.py if needed:
BASE_DIR = Path('/path/to/your/project')  # Root project directory
DATA_DIR = BASE_DIR / 'dataset_1'         # Directory containing CSV files
db_name = 'Python_Assignment_v2'          # Database name

## Testing

Run all unit tests:
    python -m unittest discover
Or run specific test files:
    python -m unittest test_main.py
    python -m unittest test_CSV_DB.py
    python -m unittest test_mapping.py
    python -m unittest test_visualization.py

Test data is processed line-by-line to handle large files efficiently
The script includes profiling capability (commented out in main.py)


